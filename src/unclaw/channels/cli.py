"""Terminal CLI entrypoint for the current Unclaw runtime."""

from __future__ import annotations

import sys

from unclaw.bootstrap import bootstrap
from unclaw.core.command_handler import CommandHandler, CommandResult, CommandStatus
from unclaw.core.session_manager import SessionManager
from unclaw.errors import UnclawError
from unclaw.schemas.chat import MessageRole
from unclaw.settings import Settings

_PLACEHOLDER_ASSISTANT_REPLY = "Runtime pipeline not implemented yet."


def main() -> int:
    """Run the interactive Unclaw CLI."""
    try:
        settings = bootstrap()
        session_manager = SessionManager.from_settings(settings)
        current_session = session_manager.ensure_current_session()
        command_handler = CommandHandler(
            settings=settings,
            session_manager=session_manager,
        )
    except UnclawError as exc:
        print(f"Failed to start Unclaw: {exc}", file=sys.stderr)
        return 1

    try:
        _print_banner(
            settings=settings,
            session_id=current_session.id,
            command_handler=command_handler,
        )
        return run_cli(session_manager=session_manager, command_handler=command_handler)
    finally:
        session_manager.close()


def run_cli(
    *,
    session_manager: SessionManager,
    command_handler: CommandHandler,
) -> int:
    """Run the interactive read-eval-print loop."""
    while True:
        try:
            user_input = input(_build_prompt(command_handler))
        except EOFError:
            print()
            print("Exiting Unclaw.")
            return 0
        except KeyboardInterrupt:
            print()
            print("Exiting Unclaw.")
            return 0

        stripped_input = user_input.strip()
        if not stripped_input:
            continue

        if stripped_input.startswith("/"):
            result = command_handler.handle(stripped_input)
            _render_command_result(result)
            if result.should_exit:
                return 0
            continue

        session = session_manager.ensure_current_session()
        session_manager.add_message(
            MessageRole.USER,
            stripped_input,
            session_id=session.id,
        )

        # Temporary placeholder until the runtime pipeline is wired in.
        assistant_reply = _PLACEHOLDER_ASSISTANT_REPLY
        session_manager.add_message(
            MessageRole.ASSISTANT,
            assistant_reply,
            session_id=session.id,
        )
        print(f"assistant> {assistant_reply}")


def _print_banner(
    *,
    settings: Settings,
    session_id: str,
    command_handler: CommandHandler,
) -> None:
    print(f"{settings.app.display_name} 🦐")
    print("Local-first agent runtime")
    print(
        f"Session: {session_id} | "
        f"Model: {command_handler.current_model_profile_name} | "
        f"Thinking: {command_handler.thinking_label}"
    )
    print("Type /help for commands.")


def _build_prompt(command_handler: CommandHandler) -> str:
    session = command_handler.session_manager.ensure_current_session()
    return (
        f"[{session.id} | model={command_handler.current_model_profile_name} | "
        f"think={command_handler.thinking_label}] you> "
    )


def _render_command_result(result: CommandResult) -> None:
    if not result.lines:
        return

    if result.status is CommandStatus.ERROR:
        first_line, *other_lines = result.lines
        print(f"Error: {first_line}")
        for line in other_lines:
            print(line)
        return

    for line in result.lines:
        print(line)


if __name__ == "__main__":
    raise SystemExit(main())
