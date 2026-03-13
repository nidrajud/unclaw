from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from unclaw.core.command_handler import CommandHandler, CommandStatus
from unclaw.settings import load_settings


def test_help_lists_enriched_commands_for_cli() -> None:
    handler = CommandHandler(
        settings=_load_repo_settings(),
        session_manager=SimpleNamespace(),
    )

    result = handler.handle("/help")

    assert result.status is CommandStatus.OK
    assert result.lines[0] == "Available slash commands for this channel:"
    assert "/help  Show this command list." in result.lines
    assert "/new               Start a fresh session." in result.lines
    assert "/sessions          List recent sessions." in result.lines
    assert "/use <session_id>  Switch to an existing session." in result.lines
    assert "/model                 Show the active model profile." in result.lines
    assert "/model <profile_name>  Switch model profiles." in result.lines
    assert "/think                 Show thinking mode status." in result.lines
    assert "/think on              Turn thinking mode on." in result.lines
    assert "/think off             Turn thinking mode off." in result.lines
    assert "/tools       List built-in tools." in result.lines
    assert "/read <path> Read one local file." in result.lines
    assert "/ls <path>   List one local directory." in result.lines
    assert "/fetch <url> Fetch one URL." in result.lines
    assert "/session  Show the current session state." in result.lines
    assert "/summary  Show the saved session summary." in result.lines
    assert "/exit  Leave the terminal runtime." in result.lines


def test_help_omits_exit_for_telegram() -> None:
    handler = CommandHandler(
        settings=_load_repo_settings(),
        session_manager=SimpleNamespace(),
        allow_exit=False,
    )

    result = handler.handle("/help")

    assert result.status is CommandStatus.OK
    assert "/exit  Leave the terminal runtime." not in result.lines


def _load_repo_settings():
    return load_settings(project_root=Path(__file__).resolve().parents[1])
