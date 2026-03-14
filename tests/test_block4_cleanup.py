from __future__ import annotations

import re
import shutil
from pathlib import Path

import pytest

import unclaw.update as update
from unclaw import main as unclaw_main
from unclaw.core.context_builder import build_context_messages
from unclaw.core.session_manager import SessionManager
from unclaw.llm.base import LLMRole
from unclaw.schemas.chat import MessageRole
from unclaw.schemas.events import EventLevel
from unclaw.settings import load_settings
from unclaw.startup import OllamaStatus, build_startup_report
from unclaw.update import GitCommandResult, run_update

_MICROSECOND_UTC_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z"
)


def test_system_prompt_is_loaded_from_config_prompts(tmp_path: Path) -> None:
    project_root = _create_temp_project(tmp_path)
    prompt_path = project_root / "config" / "prompts" / "system.txt"
    prompt_text = "You are the test prompt. Keep replies short."
    prompt_path.write_text(prompt_text + "\n", encoding="utf-8")

    settings = load_settings(project_root=project_root)
    session_manager = SessionManager.from_settings(settings)

    try:
        session = session_manager.create_session(make_current=False)
        messages = build_context_messages(
            session_manager=session_manager,
            session_id=session.id,
            user_message="hello",
        )
    finally:
        session_manager.close()

    assert settings.system_prompt == prompt_text
    assert messages[0].role is LLMRole.SYSTEM
    assert messages[0].content == prompt_text


def test_runtime_persistence_timestamps_use_microsecond_utc_precision(
    tmp_path: Path,
) -> None:
    project_root = _create_temp_project(tmp_path)
    settings = load_settings(project_root=project_root)
    session_manager = SessionManager.from_settings(settings)

    try:
        session = session_manager.create_session(make_current=False)
        message = session_manager.add_message(
            MessageRole.USER,
            "hello",
            session_id=session.id,
        )
        event = session_manager.event_repository.add_event(
            session_id=session.id,
            event_type="test.event",
            level=EventLevel.INFO,
            message="event",
        )
    finally:
        session_manager.close()

    for value in (
        session.created_at,
        session.updated_at,
        message.created_at,
        event.created_at,
    ):
        assert _MICROSECOND_UTC_RE.fullmatch(value)


def test_switch_session_keeps_only_one_active_session_without_touching_timestamps(
    tmp_path: Path,
) -> None:
    project_root = _create_temp_project(tmp_path)
    settings = load_settings(project_root=project_root)
    session_manager = SessionManager.from_settings(settings)

    try:
        first = session_manager.create_session(title="First", make_current=False)
        second = session_manager.create_session(title="Second", make_current=False)
        third = session_manager.create_session(title="Third", make_current=False)

        activated_first = session_manager.switch_session(first.id)
        first_updated_before = activated_first.updated_at
        third_updated_before = session_manager.load_session(third.id).updated_at

        activated_third = session_manager.switch_session(third.id)
        refreshed_first = session_manager.load_session(first.id)
        refreshed_second = session_manager.load_session(second.id)
        refreshed_third = session_manager.load_session(third.id)
        sessions = session_manager.list_sessions()
    finally:
        session_manager.close()

    assert activated_third.id == third.id
    assert refreshed_first is not None and refreshed_first.is_active is False
    assert refreshed_second is not None and refreshed_second.is_active is False
    assert refreshed_third is not None and refreshed_third.is_active is True
    assert refreshed_first.updated_at == first_updated_before
    assert refreshed_third.updated_at == third_updated_before
    assert sum(session.is_active for session in sessions) == 1


def test_run_update_reports_in_progress_git_operations_cleanly(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    git_dir = repo_root / ".git"
    git_dir.mkdir(parents=True)
    (git_dir / "MERGE_HEAD").write_text("abc123\n", encoding="utf-8")
    outputs: list[str] = []

    def fake_run_git(repo_root_arg: Path, *args: str) -> GitCommandResult:
        assert repo_root_arg == repo_root
        command = tuple(args)
        if command == ("rev-parse", "--show-toplevel"):
            return GitCommandResult(returncode=0, stdout=str(repo_root), stderr="")
        if command == ("rev-parse", "--git-dir"):
            return GitCommandResult(returncode=0, stdout=".git", stderr="")
        raise AssertionError(f"Unexpected git command: {command}")

    monkeypatch.setattr(update, "_run_git", fake_run_git)

    result = run_update(project_root=repo_root, output_func=outputs.append)

    assert result == 1
    assert outputs == [
        f"Repository: {repo_root}",
        (
            "Cannot run `unclaw update` while a git merge is in progress. "
            "Run `git status`, finish or abort that operation, then rerun "
            "`unclaw update`."
        ),
    ]


def test_run_update_fetch_failure_uses_a_single_actionable_detail_line(
    monkeypatch: pytest.MonkeyPatch,
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
        if command == ("rev-parse", "--git-dir"):
            return GitCommandResult(returncode=0, stdout=".git", stderr="")
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
            return GitCommandResult(
                returncode=1,
                stdout="",
                stderr="fatal: authentication failed\nhint: check credentials",
            )
        raise AssertionError(f"Unexpected git command: {command}")

    monkeypatch.setattr(update, "_run_git", fake_run_git)

    result = run_update(project_root=repo_root, output_func=outputs.append)

    assert result == 1
    assert outputs[-1] == "Could not fetch remote updates. fatal: authentication failed"
    assert not any("hint: check credentials" in line for line in outputs)


def test_startup_report_uses_secure_by_default_telegram_access_wording(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = load_settings(project_root=_repo_root())

    monkeypatch.setattr(
        "unclaw.startup.inspect_ollama",
        lambda timeout_seconds=1.5: OllamaStatus(
            cli_path="/usr/bin/ollama",
            is_installed=True,
            is_running=False,
            model_names=(),
        ),
    )

    report = build_startup_report(
        settings,
        channel_name="telegram",
        channel_enabled=True,
        required_profile_names=(),
        telegram_allowed_chat_ids=frozenset(),
    )
    access_check = next(check for check in report.checks if check.label == "Telegram access")

    assert (
        access_check.detail
        == "Secure by default: no Telegram chats are authorized yet (`allowed_chat_ids: []`)."
    )


def test_main_help_keeps_canonical_logs_commands_and_safe_update_wording() -> None:
    help_text = unclaw_main.build_parser().format_help()

    assert "  unclaw logs\n" in help_text
    assert "  unclaw logs full\n" in help_text
    assert "Safely fetch and fast-forward this local checkout." in help_text


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _create_temp_project(tmp_path: Path) -> Path:
    project_root = tmp_path / "project"
    shutil.copytree(_repo_root() / "config", project_root / "config")
    return project_root
