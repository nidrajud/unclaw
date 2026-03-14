from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from unclaw.core.command_handler import CommandHandler, CommandStatus
from unclaw.settings import load_settings


def test_help_lists_enriched_commands_for_cli() -> None:
    handler = CommandHandler(
        settings=_load_repo_settings(),
        session_manager=_build_session_manager(),
    )

    result = handler.handle("/help")

    assert result.status is CommandStatus.OK
    assert result.lines[0] == "Available slash commands for this channel:"
    assert "/new               Start a fresh session." in result.lines
    assert "/sessions          List recent sessions." in result.lines
    assert "/use <session_id>  Switch to an existing session." in result.lines
    assert "/model                 Show the active model profile." in result.lines
    assert "/model <profile_name>  Switch model profiles." in result.lines
    assert "/think                 Show thinking mode status." in result.lines
    assert "/think on              Turn thinking mode on." in result.lines
    assert "/think off             Turn thinking mode off." in result.lines
    assert "/tools            List built-in tools." in result.lines
    assert "/read <path>      Read one local file inside allowed roots." in result.lines
    assert (
        "/ls [path]        List one local directory inside allowed roots. Defaults to the current directory."
        in result.lines
    )
    assert "/fetch <url>      Fetch one public URL." in result.lines
    assert (
        "/search <query>  Search the public web, ground the answer, and append compact sources."
        in result.lines
    )
    assert "/session  Show the current session state." in result.lines
    assert "/summary  Show the saved session summary." in result.lines
    assert "/help  Show this command list with examples." in result.lines
    assert "Examples:" in result.lines
    assert "/ls ." in result.lines
    assert "/ls /home/user/project" in result.lines
    assert "/read README.md" in result.lines
    assert "/search local ai agents" in result.lines
    assert "/exit  Leave the terminal runtime." in result.lines
    assert (
        "Tip: use 'unclaw logs' or 'unclaw logs full' in another terminal."
        in result.lines
    )


def test_help_omits_exit_for_telegram() -> None:
    handler = CommandHandler(
        settings=_load_repo_settings(),
        session_manager=_build_session_manager(),
        allow_exit=False,
    )

    result = handler.handle("/help")

    assert result.status is CommandStatus.OK
    assert "/exit  Leave the terminal runtime." not in result.lines
    assert (
        "Tip: use 'unclaw logs' or 'unclaw logs full' in another terminal."
        not in result.lines
    )


def test_fast_profile_forces_thinking_off_on_startup() -> None:
    settings = _with_default_fast_and_thinking_enabled(_load_repo_settings())

    handler = CommandHandler(
        settings=settings,
        session_manager=_build_session_manager(),
    )

    assert handler.current_model_profile_name == "fast"
    assert handler.thinking_enabled is False


def test_switching_to_fast_turns_thinking_off_cleanly() -> None:
    handler = CommandHandler(
        settings=_load_repo_settings(),
        session_manager=_build_session_manager(),
        current_model_profile_name="main",
        thinking_enabled=True,
    )

    result = handler.handle("/model fast")

    assert result.status is CommandStatus.OK
    assert handler.current_model_profile_name == "fast"
    assert handler.thinking_enabled is False
    assert (
        "Thinking mode was turned off because fast mode does not support thinking."
        in result.lines
    )


def test_think_status_on_fast_explains_that_thinking_is_unsupported() -> None:
    handler = CommandHandler(
        settings=_load_repo_settings(),
        session_manager=_build_session_manager(),
        current_model_profile_name="fast",
        thinking_enabled=False,
    )

    result = handler.handle("/think")

    assert result.status is CommandStatus.OK
    assert result.lines == (
        "Thinking mode: off",
        "Fast mode does not support thinking.",
    )


def test_think_on_fails_clearly_on_fast() -> None:
    handler = CommandHandler(
        settings=_load_repo_settings(),
        session_manager=_build_session_manager(),
        current_model_profile_name="fast",
        thinking_enabled=False,
    )

    result = handler.handle("/think on")

    assert result.status is CommandStatus.ERROR
    assert handler.thinking_enabled is False
    assert result.lines == (
        "Fast mode does not support thinking. "
        "Switch to another model profile to turn it on.",
    )


def test_switching_back_to_supported_model_keeps_thinking_off() -> None:
    handler = CommandHandler(
        settings=_load_repo_settings(),
        session_manager=_build_session_manager(),
        current_model_profile_name="main",
        thinking_enabled=True,
    )

    assert handler.handle("/model fast").status is CommandStatus.OK

    result = handler.handle("/model main")

    assert result.status is CommandStatus.OK
    assert handler.current_model_profile_name == "main"
    assert handler.thinking_enabled is False


def test_ls_defaults_to_current_directory_when_no_path_is_given() -> None:
    handler = CommandHandler(
        settings=_load_repo_settings(),
        session_manager=_build_session_manager(),
    )

    result = handler.handle("/ls")

    assert result.status is CommandStatus.OK
    assert result.tool_call is not None
    assert result.tool_call.tool_name == "list_directory"
    assert result.tool_call.arguments == {"path": "."}


def test_search_uses_freeform_query_and_wires_to_search_tool() -> None:
    handler = CommandHandler(
        settings=_load_repo_settings(),
        session_manager=_build_session_manager(),
    )

    result = handler.handle("/search local-first agent runtime")

    assert result.status is CommandStatus.OK
    assert result.tool_call is not None
    assert result.tool_call.tool_name == "search_web"
    assert result.tool_call.arguments == {
        "query": "local-first agent runtime",
    }


def test_search_accepts_apostrophes_in_natural_language_queries() -> None:
    handler = CommandHandler(
        settings=_load_repo_settings(),
        session_manager=_build_session_manager(),
    )

    result = handler.handle("/search qu'est-ce qu'il s'est passe aujourd'hui ?")

    assert result.status is CommandStatus.OK
    assert result.tool_call is not None
    assert result.tool_call.tool_name == "search_web"
    assert result.tool_call.arguments == {
        "query": "qu'est-ce qu'il s'est passe aujourd'hui ?",
    }


def test_read_keeps_useful_quoted_paths_working() -> None:
    handler = CommandHandler(
        settings=_load_repo_settings(),
        session_manager=_build_session_manager(),
    )

    result = handler.handle('/read "docs/My Notes.txt"')

    assert result.status is CommandStatus.OK
    assert result.tool_call is not None
    assert result.tool_call.tool_name == "read_text_file"
    assert result.tool_call.arguments == {"path": "docs/My Notes.txt"}


def _load_repo_settings():
    return load_settings(project_root=Path(__file__).resolve().parents[1])


def _build_session_manager() -> SimpleNamespace:
    return SimpleNamespace(current_session_id="sess-current")


def _with_default_fast_and_thinking_enabled(settings):
    return replace(
        settings,
        app=replace(
            settings.app,
            default_model_profile="fast",
            thinking=replace(settings.app.thinking, default_enabled=True),
        ),
    )
