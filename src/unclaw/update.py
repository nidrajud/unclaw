"""Safe local git update command for the Unclaw project."""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from unclaw.errors import ConfigurationError
from unclaw.settings import resolve_project_root

OutputFunc = Callable[[str], None]
_UNSUPPORTED_GIT_STATE_MESSAGE = (
    "This checkout is in an unsupported git state for `unclaw update`. "
    "Run `git status`, resolve that state manually, then rerun `unclaw update`."
)
_DIVERGENCE_ERROR_MESSAGE = (
    "Could not determine local and remote branch divergence safely. "
    "Run `git fetch --prune` and `git status`, then update manually if needed."
)


@dataclass(frozen=True, slots=True)
class GitCommandResult:
    """One completed git command invocation."""

    returncode: int
    stdout: str
    stderr: str


class GitUpdateSafetyError(ConfigurationError):
    """Raised when automatic update cannot safely reason about git state."""


def main(project_root: Path | None = None) -> int:
    """Run the safe local update flow."""

    return run_update(project_root=project_root)


def run_update(
    *,
    project_root: Path | None = None,
    output_func: OutputFunc = print,
) -> int:
    """Fetch and fast-forward the local checkout when it is safe."""

    repo_root: Path | None = None
    branch_name: str | None = None
    upstream_ref: str | None = None
    printed_repo_header = False

    try:
        target_root = _resolve_target_root(project_root)
        repo_root = _git_repo_root(target_root)
        if repo_root is None:
            output_func(f"This folder is not a git checkout: {target_root}")
            return 1

        _ensure_no_in_progress_git_operation(repo_root)
        branch_name = _current_branch_name(repo_root)
        if branch_name is None:
            output_func(f"Repository: {repo_root}")
            output_func(
                "Cannot run `unclaw update` from a detached HEAD. Check out a branch, then rerun `unclaw update`."
            )
            return 1

        upstream_ref = _upstream_ref(repo_root)
        if upstream_ref is None:
            output_func(f"Repository: {repo_root}")
            output_func(f"Branch: {branch_name}")
            output_func(
                "This branch has no upstream tracking branch. Set one with `git branch --set-upstream-to <remote>/<branch>`, then rerun `unclaw update`."
            )
            return 1

        remote_name = _remote_name_from_upstream(upstream_ref)
        output_func(f"Repository: {repo_root}")
        output_func(f"Branch: {branch_name}")
        output_func(f"Upstream: {upstream_ref}")
        printed_repo_header = True

        fetch_result = _run_git(repo_root, "fetch", remote_name, "--prune")
        if fetch_result.returncode != 0:
            output_func(
                _format_git_failure("Could not fetch remote updates.", fetch_result)
            )
            return 1
        output_func(f"Fetched latest refs from {remote_name}.")

        ahead_count, behind_count = _branch_divergence(repo_root, upstream_ref)
        has_local_changes, status_lines = _working_tree_status(repo_root)

        if behind_count == 0 and ahead_count == 0:
            output_func(f"Already up to date on {branch_name}.")
            if has_local_changes:
                output_func("Uncommitted local changes were left untouched.")
            return 0

        if ahead_count > 0 and behind_count == 0:
            output_func(
                f"Local branch {branch_name} is ahead of {upstream_ref} by {ahead_count} commit(s). Nothing was pulled."
            )
            if has_local_changes:
                output_func("Uncommitted local changes were left untouched.")
            return 0

        if has_local_changes:
            output_func(
                "Cannot fast-forward because you have uncommitted local changes. Commit or stash them, then rerun `unclaw update`."
            )
            if status_lines:
                output_func("Local changes:")
                for line in status_lines[:10]:
                    output_func(f"  {line}")
            return 1

        if ahead_count > 0 and behind_count > 0:
            output_func(
                f"Local branch {branch_name} has diverged from {upstream_ref} (ahead {ahead_count}, behind {behind_count})."
            )
            output_func(
                "Update manually with `git status` and `git log --oneline --graph --decorate --all`."
            )
            return 1

        merge_result = _run_git(repo_root, "merge", "--ff-only", upstream_ref)
        if merge_result.returncode != 0:
            output_func(
                _format_git_failure(
                    f"Fast-forward update from {upstream_ref} failed.",
                    merge_result,
                )
            )
            return 1

        output_func(
            f"Updated {branch_name} with {behind_count} incoming commit(s) from {upstream_ref}."
        )
        return 0
    except GitUpdateSafetyError as exc:
        if repo_root is not None and not printed_repo_header:
            output_func(f"Repository: {repo_root}")
        if branch_name is not None and not printed_repo_header:
            output_func(f"Branch: {branch_name}")
        if upstream_ref is not None and not printed_repo_header:
            output_func(f"Upstream: {upstream_ref}")
        output_func(str(exc))
        return 1


def _resolve_target_root(project_root: Path | None) -> Path:
    if project_root is not None:
        return project_root.expanduser().resolve()
    try:
        return resolve_project_root(None)
    except ConfigurationError:
        return Path.cwd().resolve()


def _git_repo_root(path: Path) -> Path | None:
    result = _run_git(path, "rev-parse", "--show-toplevel")
    if result.returncode != 0:
        return None
    root_text = _read_single_output_line(
        result.stdout,
        error_message=_UNSUPPORTED_GIT_STATE_MESSAGE,
    )
    if not root_text:
        return None
    return Path(root_text).resolve()


def _current_branch_name(repo_root: Path) -> str | None:
    result = _run_git(repo_root, "branch", "--show-current")
    if result.returncode != 0:
        return None
    branch_name = _read_single_output_line(
        result.stdout,
        error_message=_UNSUPPORTED_GIT_STATE_MESSAGE,
    )
    if not branch_name:
        return None
    return branch_name


def _upstream_ref(repo_root: Path) -> str | None:
    result = _run_git(
        repo_root,
        "rev-parse",
        "--abbrev-ref",
        "--symbolic-full-name",
        "@{upstream}",
    )
    if result.returncode != 0:
        return None
    upstream_ref = _read_single_output_line(
        result.stdout,
        error_message=_UNSUPPORTED_GIT_STATE_MESSAGE,
    )
    if not upstream_ref:
        return None
    return upstream_ref


def _branch_divergence(repo_root: Path, upstream_ref: str) -> tuple[int, int]:
    result = _run_git(
        repo_root,
        "rev-list",
        "--left-right",
        "--count",
        f"HEAD...{upstream_ref}",
    )
    if result.returncode != 0:
        raise GitUpdateSafetyError(_DIVERGENCE_ERROR_MESSAGE)

    counts = result.stdout.strip().split()
    if len(counts) != 2:
        raise GitUpdateSafetyError(_DIVERGENCE_ERROR_MESSAGE)
    try:
        ahead_count = int(counts[0])
        behind_count = int(counts[1])
    except ValueError as exc:
        raise GitUpdateSafetyError(_DIVERGENCE_ERROR_MESSAGE) from exc
    if ahead_count < 0 or behind_count < 0:
        raise GitUpdateSafetyError(_DIVERGENCE_ERROR_MESSAGE)
    return ahead_count, behind_count


def _working_tree_status(repo_root: Path) -> tuple[bool, list[str]]:
    result = _run_git(repo_root, "status", "--short")
    if result.returncode != 0:
        raise GitUpdateSafetyError(_UNSUPPORTED_GIT_STATE_MESSAGE)
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    return bool(lines), lines


def _ensure_no_in_progress_git_operation(repo_root: Path) -> None:
    git_dir = _git_dir(repo_root)
    for marker_name, operation_name in (
        ("MERGE_HEAD", "merge"),
        ("rebase-merge", "rebase"),
        ("rebase-apply", "rebase"),
        ("CHERRY_PICK_HEAD", "cherry-pick"),
        ("REVERT_HEAD", "revert"),
        ("BISECT_LOG", "bisect"),
    ):
        if (git_dir / marker_name).exists():
            raise GitUpdateSafetyError(
                "Cannot run `unclaw update` while a git "
                f"{operation_name} is in progress. Run `git status`, finish or abort "
                "that operation, then rerun `unclaw update`."
            )


def _git_dir(repo_root: Path) -> Path:
    result = _run_git(repo_root, "rev-parse", "--git-dir")
    if result.returncode != 0:
        raise GitUpdateSafetyError(_UNSUPPORTED_GIT_STATE_MESSAGE)

    git_dir_text = _read_single_output_line(
        result.stdout,
        error_message=_UNSUPPORTED_GIT_STATE_MESSAGE,
    )
    if not git_dir_text:
        raise GitUpdateSafetyError(_UNSUPPORTED_GIT_STATE_MESSAGE)

    git_dir = Path(git_dir_text)
    if not git_dir.is_absolute():
        git_dir = (repo_root / git_dir).resolve()
    return git_dir


def _run_git(repo_root: Path, *args: str) -> GitCommandResult:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise GitUpdateSafetyError(
            "Could not run git safely in this checkout. Update it manually."
        ) from exc
    return GitCommandResult(
        returncode=completed.returncode,
        stdout=completed.stdout.strip(),
        stderr=completed.stderr.strip(),
    )


def _format_git_failure(message: str, result: GitCommandResult) -> str:
    detail = _first_detail_line(result.stderr or result.stdout)
    if not detail:
        detail = "git returned an unknown error."
    return f"{message} {detail}"


def _read_single_output_line(stdout: str, *, error_message: str) -> str:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not lines:
        return ""
    if len(lines) != 1:
        raise GitUpdateSafetyError(error_message)
    return lines[0]


def _remote_name_from_upstream(upstream_ref: str) -> str:
    remote_name, separator, branch_name = upstream_ref.partition("/")
    if not separator or not remote_name or not branch_name:
        raise GitUpdateSafetyError(_UNSUPPORTED_GIT_STATE_MESSAGE)
    return remote_name


def _first_detail_line(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    return lines[0]


if __name__ == "__main__":
    raise SystemExit(main())
