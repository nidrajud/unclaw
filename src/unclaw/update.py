"""Safe local git update command for the Unclaw project."""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from unclaw.errors import ConfigurationError
from unclaw.settings import resolve_project_root

OutputFunc = Callable[[str], None]


@dataclass(frozen=True, slots=True)
class GitCommandResult:
    """One completed git command invocation."""

    returncode: int
    stdout: str
    stderr: str


def main(project_root: Path | None = None) -> int:
    """Run the safe local update flow."""

    return run_update(project_root=project_root)


def run_update(
    *,
    project_root: Path | None = None,
    output_func: OutputFunc = print,
) -> int:
    """Fetch and fast-forward the local checkout when it is safe."""

    target_root = _resolve_target_root(project_root)
    repo_root = _git_repo_root(target_root)
    if repo_root is None:
        output_func(f"This folder is not a git checkout: {target_root}")
        return 1

    branch_name = _current_branch_name(repo_root)
    if branch_name is None:
        output_func(f"Repository: {repo_root}")
        output_func(
            "Cannot update automatically from a detached HEAD. Check out a branch and rerun `unclaw update`."
        )
        return 1

    upstream_ref = _upstream_ref(repo_root)
    if upstream_ref is None:
        output_func(f"Repository: {repo_root}")
        output_func(f"Branch: {branch_name}")
        output_func(
            "This branch has no upstream tracking branch. Set one with `git branch --set-upstream-to <remote>/<branch>` and rerun `unclaw update`."
        )
        return 1

    remote_name, _, _ = upstream_ref.partition("/")
    output_func(f"Repository: {repo_root}")
    output_func(f"Branch: {branch_name}")
    output_func(f"Upstream: {upstream_ref}")

    fetch_result = _run_git(repo_root, "fetch", remote_name, "--prune")
    if fetch_result.returncode != 0:
        output_func(_format_git_failure("Could not fetch remote updates.", fetch_result))
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
        output_func("Update manually with `git status` and `git log --oneline --graph --decorate --all`.")
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
    root_text = result.stdout.strip()
    if not root_text:
        return None
    return Path(root_text).resolve()


def _current_branch_name(repo_root: Path) -> str | None:
    result = _run_git(repo_root, "branch", "--show-current")
    if result.returncode != 0:
        return None
    branch_name = result.stdout.strip()
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
    upstream_ref = result.stdout.strip()
    if not upstream_ref:
        return None
    return upstream_ref


def _branch_divergence(repo_root: Path, upstream_ref: str) -> tuple[int, int]:
    result = _run_git(repo_root, "rev-list", "--left-right", "--count", f"HEAD...{upstream_ref}")
    if result.returncode != 0:
        return 0, 0

    counts = result.stdout.strip().split()
    if len(counts) != 2:
        return 0, 0
    ahead_count = int(counts[0])
    behind_count = int(counts[1])
    return ahead_count, behind_count


def _working_tree_status(repo_root: Path) -> tuple[bool, list[str]]:
    result = _run_git(repo_root, "status", "--short")
    if result.returncode != 0:
        return False, []
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    return bool(lines), lines


def _run_git(repo_root: Path, *args: str) -> GitCommandResult:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    return GitCommandResult(
        returncode=completed.returncode,
        stdout=completed.stdout.strip(),
        stderr=completed.stderr.strip(),
    )


def _format_git_failure(message: str, result: GitCommandResult) -> str:
    detail = result.stderr or result.stdout or "git returned an unknown error."
    return f"{message} {detail}"


if __name__ == "__main__":
    raise SystemExit(main())
