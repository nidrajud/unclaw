from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

import unclaw.update as update
from unclaw.update import GitCommandResult, run_update

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git is required")


def test_run_update_reports_non_git_folder(tmp_path: Path) -> None:
    project_root = tmp_path / "plain-folder"
    project_root.mkdir()
    outputs: list[str] = []

    result = run_update(project_root=project_root, output_func=outputs.append)

    assert result == 1
    assert outputs == [f"This folder is not a git checkout: {project_root.resolve()}"]


def test_run_update_fast_forwards_clean_checkout(tmp_path: Path) -> None:
    _remote_repo, seed_repo, local_repo = _create_git_checkout(tmp_path)
    _commit_file(seed_repo, "README.md", "v2\n", message="Update README")
    _git(seed_repo, "push")

    outputs: list[str] = []
    result = run_update(project_root=local_repo, output_func=outputs.append)

    assert result == 0
    assert (local_repo / "README.md").read_text(encoding="utf-8") == "v2\n"
    assert any("Fetched latest refs from origin." in line for line in outputs)
    assert any("Updated main with 1 incoming commit(s) from origin/main." in line for line in outputs)


def test_run_update_blocks_dirty_checkout_when_remote_has_updates(tmp_path: Path) -> None:
    _remote_repo, seed_repo, local_repo = _create_git_checkout(tmp_path)
    _commit_file(seed_repo, "README.md", "remote change\n", message="Remote update")
    _git(seed_repo, "push")

    (local_repo / "README.md").write_text("local dirty change\n", encoding="utf-8")
    outputs: list[str] = []
    result = run_update(project_root=local_repo, output_func=outputs.append)

    assert result == 1
    assert any("Cannot fast-forward because you have uncommitted local changes." in line for line in outputs)
    assert any("README.md" in line for line in outputs)


def test_run_update_reports_malformed_divergence_output_cleanly(
    monkeypatch,
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    outputs: list[str] = []

    def fake_run_git(repo_root_arg: Path, *args: str) -> GitCommandResult:
        assert repo_root_arg == repo_root
        command = tuple(args)
        if command == ("rev-parse", "--show-toplevel"):
            return GitCommandResult(returncode=0, stdout=str(repo_root), stderr="")
        if command == ("branch", "--show-current"):
            return GitCommandResult(returncode=0, stdout="main", stderr="")
        if command == (
            "rev-parse",
            "--abbrev-ref",
            "--symbolic-full-name",
            "@{upstream}",
        ):
            return GitCommandResult(returncode=0, stdout="origin/main", stderr="")
        if command == ("fetch", "origin", "--prune"):
            return GitCommandResult(returncode=0, stdout="", stderr="")
        if command == ("rev-list", "--left-right", "--count", "HEAD...origin/main"):
            return GitCommandResult(returncode=0, stdout="oops", stderr="")
        raise AssertionError(f"Unexpected git command: {command}")

    monkeypatch.setattr(update, "_run_git", fake_run_git)

    result = run_update(project_root=repo_root, output_func=outputs.append)

    assert result == 1
    assert "Repository: " + str(repo_root) in outputs
    assert "Branch: main" in outputs
    assert "Upstream: origin/main" in outputs
    assert (
        "Could not determine local/remote branch divergence safely. Please update manually."
        in outputs
    )


def test_run_update_reports_unsupported_upstream_state_cleanly(
    monkeypatch,
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    outputs: list[str] = []

    def fake_run_git(repo_root_arg: Path, *args: str) -> GitCommandResult:
        assert repo_root_arg == repo_root
        command = tuple(args)
        if command == ("rev-parse", "--show-toplevel"):
            return GitCommandResult(returncode=0, stdout=str(repo_root), stderr="")
        if command == ("branch", "--show-current"):
            return GitCommandResult(returncode=0, stdout="main", stderr="")
        if command == (
            "rev-parse",
            "--abbrev-ref",
            "--symbolic-full-name",
            "@{upstream}",
        ):
            return GitCommandResult(returncode=0, stdout="origin-main", stderr="")
        raise AssertionError(f"Unexpected git command: {command}")

    monkeypatch.setattr(update, "_run_git", fake_run_git)

    result = run_update(project_root=repo_root, output_func=outputs.append)

    assert result == 1
    assert "Repository: " + str(repo_root) in outputs
    assert "Branch: main" in outputs
    assert (
        "This checkout is in an unsupported git state for automatic update."
        in outputs
    )


def _create_git_checkout(tmp_path: Path) -> tuple[Path, Path, Path]:
    remote_repo = tmp_path / "remote.git"
    seed_repo = tmp_path / "seed"
    local_repo = tmp_path / "local"

    _git(tmp_path, "init", "--bare", str(remote_repo))
    _git(tmp_path, "-C", str(remote_repo), "symbolic-ref", "HEAD", "refs/heads/main")
    _git(tmp_path, "clone", str(remote_repo), str(seed_repo))
    _configure_git_identity(seed_repo)
    _git(seed_repo, "checkout", "-b", "main")
    _commit_file(seed_repo, "README.md", "v1\n", message="Initial commit")
    _git(seed_repo, "push", "-u", "origin", "main")
    _git(tmp_path, "clone", str(remote_repo), str(local_repo))
    _configure_git_identity(local_repo)
    return remote_repo, seed_repo, local_repo


def _commit_file(repo_root: Path, file_name: str, contents: str, *, message: str) -> None:
    (repo_root / file_name).write_text(contents, encoding="utf-8")
    _git(repo_root, "add", file_name)
    _git(repo_root, "commit", "-m", message)


def _configure_git_identity(repo_root: Path) -> None:
    _git(repo_root, "config", "user.name", "Unclaw Tests")
    _git(repo_root, "config", "user.email", "tests@example.com")


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
