"""Terminal CLI entrypoint for the current Unclaw runtime."""

from __future__ import annotations

import sys

from unclaw.bootstrap import bootstrap
from unclaw.core.command_handler import CommandHandler, CommandResult, CommandStatus
from unclaw.core.executor import ToolExecutor
from unclaw.core.runtime import run_user_turn
from unclaw.core.session_manager import SessionManager
from unclaw.errors import UnclawError
from unclaw.logs.event_bus import EventBus
from unclaw.logs.tracer import Tracer
from unclaw.memory import MemoryManager
from unclaw.schemas.chat import MessageRole
from unclaw.settings import Settings
from unclaw.tools.contracts import ToolDefinition, ToolResult


def main() -> int:
    """Run the interactive Unclaw CLI."""
    try:
        settings = bootstrap()
        session_manager = SessionManager.from_settings(settings)
        memory_manager = MemoryManager(session_manager=session_manager)
        current_session = session_manager.ensure_current_session()
        event_bus = EventBus()
        tracer = Tracer(
            event_bus=event_bus,
            event_repository=session_manager.event_repository,
        )
        tool_executor = ToolExecutor.with_default_tools()
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
        return run_cli(
            session_manager=session_manager,
            command_handler=command_handler,
            memory_manager=memory_manager,
            tracer=tracer,
            tool_executor=tool_executor,
        )
    finally:
        session_manager.close()


def run_cli(
    *,
    session_manager: SessionManager,
    command_handler: CommandHandler,
    memory_manager: MemoryManager,
    tracer: Tracer,
    tool_executor: ToolExecutor,
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
            memory_command_result = _handle_memory_command(
                raw_command=stripped_input,
                memory_manager=memory_manager,
            )
            if memory_command_result is not None:
                _render_command_result(memory_command_result)
                continue

            result = command_handler.handle(stripped_input)
            if result.list_tools:
                _render_tool_list(tool_executor.list_tools())
                continue
            if result.tool_call is not None:
                _render_tool_result(tool_executor.execute(result.tool_call))
                continue
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
        assistant_reply = run_user_turn(
            session_manager=session_manager,
            command_handler=command_handler,
            user_input=stripped_input,
            tracer=tracer,
        )
        print(f"assistant> {assistant_reply}")
        try:
            memory_manager.build_or_refresh_session_summary(session.id)
        except UnclawError as exc:
            print(f"Warning: could not refresh session summary: {exc}")


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


def _render_tool_list(tools: list[ToolDefinition]) -> None:
    if not tools:
        print("No built-in tools available.")
        return

    name_width = max(len("Name"), *(len(tool.name) for tool in tools))
    permission_width = max(
        len("Permission"),
        *(len(tool.permission_level.value) for tool in tools),
    )

    print("Built-in tools:")
    print(
        f"{'Name'.ljust(name_width)}  "
        f"{'Permission'.ljust(permission_width)}  "
        "Description"
    )
    for tool in tools:
        print(
            f"{tool.name.ljust(name_width)}  "
            f"{tool.permission_level.value.ljust(permission_width)}  "
            f"{tool.description}"
        )


def _render_tool_result(result: ToolResult) -> None:
    if result.success:
        print(result.output_text)
        return

    if not result.output_text:
        print(f"Error: {result.error}")
        return

    lines = result.output_text.splitlines() or [result.output_text]
    first_line, *other_lines = lines
    print(f"Error: {first_line}")
    for line in other_lines:
        print(line)


def _handle_memory_command(
    *,
    raw_command: str,
    memory_manager: MemoryManager,
) -> CommandResult | None:
    normalized = raw_command.strip()
    parts = normalized[1:].split()
    command = parts[0].lower()
    arguments = parts[1:]

    try:
        if command == "summary":
            if arguments:
                return CommandResult(
                    status=CommandStatus.ERROR,
                    lines=("Usage: /summary",),
                )

            summary_text = memory_manager.get_session_summary()
            return CommandResult(
                status=CommandStatus.OK,
                lines=("Session summary:", summary_text),
            )

        if command == "session":
            if arguments:
                return CommandResult(
                    status=CommandStatus.ERROR,
                    lines=("Usage: /session",),
                )

            state = memory_manager.get_session_state()
            lines = [
                f"Session: {state.session_id}",
                f"Title: {state.title}",
                f"Updated: {state.updated_at}",
                (
                    "Messages: "
                    f"{state.message_count} total | "
                    f"{state.user_message_count} user | "
                    f"{state.assistant_message_count} assistant"
                ),
                f"Summary: {state.summary_text}",
            ]
            if state.recent_snippets:
                lines.append("Recent snippets:")
                lines.extend(state.recent_snippets)

            return CommandResult(
                status=CommandStatus.OK,
                lines=tuple(lines),
            )
    except UnclawError as exc:
        return CommandResult(status=CommandStatus.ERROR, lines=(str(exc),))

    return None


if __name__ == "__main__":
    raise SystemExit(main())
