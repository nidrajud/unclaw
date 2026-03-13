"""Terminal CLI entrypoint for the current Unclaw runtime."""

from __future__ import annotations

import sys
from pathlib import Path

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
from unclaw.startup import (
    StartupReport,
    build_banner,
    build_startup_report,
    format_startup_report,
)
from unclaw.tools.contracts import ToolDefinition, ToolResult


def main(project_root: Path | None = None) -> int:
    """Run the interactive Unclaw CLI."""
    try:
        settings = bootstrap(project_root=project_root)
        startup_report = build_startup_report(
            settings,
            channel_name="terminal",
            channel_enabled=settings.app.channels.terminal_enabled,
            required_profile_names=(settings.app.default_model_profile,),
            optional_profile_names=tuple(
                profile_name
                for profile_name in settings.models
                if profile_name != settings.app.default_model_profile
            ),
        )
        if startup_report.has_errors:
            print(_build_preflight_banner(settings))
            print(format_startup_report(startup_report))
            return 1

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
            memory_manager=memory_manager,
        )
    except UnclawError as exc:
        print(f"Failed to start Unclaw: {exc}", file=sys.stderr)
        return 1

    try:
        _print_banner(
            settings=settings,
            session_id=current_session.id,
            command_handler=command_handler,
            startup_report=startup_report,
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
        print(f"Unclaw> {assistant_reply}")
        try:
            memory_manager.build_or_refresh_session_summary(session.id)
        except UnclawError as exc:
            print(f"Warning: could not refresh session summary: {exc}")


def _print_banner(
    *,
    settings: Settings,
    session_id: str,
    command_handler: CommandHandler,
    startup_report: StartupReport,
) -> None:
    print(
        build_banner(
            title="Unclaw terminal",
            subtitle="Local-first assistant runtime with live local model routing.",
            rows=(
                ("mode", "terminal"),
                ("session", session_id),
                (
                    "default",
                    (
                        f"{command_handler.current_model_profile_name} -> "
                        f"{command_handler.current_model_profile.model_name}"
                    ),
                ),
                ("thinking", command_handler.thinking_label),
                ("logging", settings.app.logging.mode),
            ),
        )
    )
    print(format_startup_report(startup_report))
    print("Type /help for commands. Press Ctrl-D or use /exit to leave.")


def _build_preflight_banner(settings: Settings) -> str:
    return build_banner(
        title="Unclaw terminal",
        subtitle="Local-first assistant runtime with live local model routing.",
        rows=(
            ("mode", "terminal"),
            (
                "default",
                f"{settings.app.default_model_profile} -> {settings.default_model.model_name}",
            ),
            ("logging", settings.app.logging.mode),
        ),
    )


def _build_prompt(command_handler: CommandHandler) -> str:
    session = command_handler.session_manager.ensure_current_session()
    return (
        f"unclaw[{session.id} | model={command_handler.current_model_profile_name} | "
        f"think={command_handler.thinking_label}]> "
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


if __name__ == "__main__":
    raise SystemExit(main())
