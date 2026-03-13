"""Slash command parsing and handling shared by interactive channels."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from shlex import split as shlex_split
from typing import TYPE_CHECKING

from unclaw.core.executor import resolve_builtin_tool_command
from unclaw.core.session_manager import SessionManager, SessionManagerError
from unclaw.schemas.session import SessionSummary
from unclaw.settings import ModelProfile, Settings
from unclaw.tools.contracts import ToolCall

if TYPE_CHECKING:
    from unclaw.memory import MemoryManager


class CommandStatus(StrEnum):
    """Render-friendly command outcome types."""

    OK = "ok"
    ERROR = "error"
    EXIT = "exit"


@dataclass(frozen=True, slots=True)
class CommandResult:
    """Structured result returned by the slash command handler."""

    status: CommandStatus
    lines: tuple[str, ...]
    list_tools: bool = False
    tool_call: ToolCall | None = None

    @property
    def should_exit(self) -> bool:
        return self.status is CommandStatus.EXIT


@dataclass(frozen=True, slots=True)
class ParsedCommand:
    """One parsed slash command with tokenized and raw arguments."""

    name: str
    arguments: tuple[str, ...]
    raw_arguments: str


@dataclass(slots=True)
class CommandHandler:
    """Handle slash commands without executing any model calls."""

    settings: Settings
    session_manager: SessionManager
    memory_manager: MemoryManager | None = None
    current_model_profile_name: str | None = None
    thinking_enabled: bool | None = None
    allow_exit: bool = True

    def __post_init__(self) -> None:
        if self.current_model_profile_name is None:
            self.current_model_profile_name = self.settings.app.default_model_profile
        elif self.current_model_profile_name not in self.settings.models:
            raise ValueError(
                f"Unknown model profile '{self.current_model_profile_name}'."
            )

        if self.thinking_enabled is None:
            self.thinking_enabled = self.settings.app.thinking.default_enabled

        if self.thinking_enabled and not self.current_model_profile.thinking_supported:
            self.thinking_enabled = False

    @property
    def current_model_profile(self) -> ModelProfile:
        return self.settings.models[self.current_model_profile_name]

    @property
    def thinking_label(self) -> str:
        return "on" if self.thinking_enabled else "off"

    def handle(self, raw_command: str) -> CommandResult:
        """Parse one slash command and return a structured result."""
        try:
            parsed_command = self._parse_command(raw_command)
        except ValueError as exc:
            return self._error(str(exc))

        try:
            match parsed_command.name:
                case "new":
                    return self._handle_new(parsed_command.arguments)
                case "sessions":
                    return self._handle_sessions(parsed_command.arguments)
                case "use":
                    return self._handle_use(parsed_command.arguments)
                case "summary":
                    return self._handle_summary(parsed_command.arguments)
                case "session":
                    return self._handle_session(parsed_command.arguments)
                case "model":
                    return self._handle_model(parsed_command.arguments)
                case "think":
                    return self._handle_think(parsed_command.arguments)
                case "tools":
                    return self._handle_tools(parsed_command.arguments)
                case "read":
                    return self._handle_tool_command(
                        parsed_command,
                        usage_line="/read <path>",
                    )
                case "ls":
                    return self._handle_tool_command(
                        parsed_command,
                        usage_line="/ls <path>",
                    )
                case "fetch":
                    return self._handle_tool_command(
                        parsed_command,
                        usage_line="/fetch <url>",
                    )
                case "help":
                    return self._handle_help(parsed_command.arguments)
                case "exit":
                    return self._handle_exit(parsed_command.arguments)
                case _:
                    return self._error(
                        f"Unknown command '/{parsed_command.name}'. "
                        "Use /help to list commands."
                    )
        except SessionManagerError as exc:
            return self._error(str(exc))

    def _handle_new(self, arguments: tuple[str, ...]) -> CommandResult:
        if arguments:
            return self._usage("/new")

        session = self.session_manager.create_session()
        return self._ok(
            f"Created and switched to session {session.id}.",
            f"Title: {session.title}",
        )

    def _handle_sessions(self, arguments: tuple[str, ...]) -> CommandResult:
        if arguments:
            return self._usage("/sessions")

        sessions = self.session_manager.list_sessions()
        if not sessions:
            return self._ok("No sessions found.")

        lines = ["Recent sessions:"]
        for session in sessions:
            lines.append(self._format_session_line(session))
        return self._ok(*lines)

    def _handle_use(self, arguments: tuple[str, ...]) -> CommandResult:
        if len(arguments) != 1:
            return self._usage("/use <session_id>")

        session = self.session_manager.switch_session(arguments[0])
        return self._ok(
            f"Switched to session {session.id}.",
            f"Title: {session.title}",
        )

    def _handle_summary(self, arguments: tuple[str, ...]) -> CommandResult:
        if arguments:
            return self._usage("/summary")
        if self.memory_manager is None:
            return self._error("Session memory is not available in this channel.")

        summary_text = self.memory_manager.get_session_summary()
        return self._ok("Session summary:", summary_text)

    def _handle_session(self, arguments: tuple[str, ...]) -> CommandResult:
        if arguments:
            return self._usage("/session")
        if self.memory_manager is None:
            return self._error("Session memory is not available in this channel.")

        state = self.memory_manager.get_session_state()
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

        return self._ok(*lines)

    def _handle_model(self, arguments: tuple[str, ...]) -> CommandResult:
        if not arguments:
            profile = self.current_model_profile
            return self._ok(
                f"Current model profile: {profile.name}",
                f"Provider: {profile.provider} | Model: {profile.model_name}",
            )

        if len(arguments) != 1:
            return self._usage("/model <profile_name>")

        profile_name = arguments[0]
        profile = self.settings.models.get(profile_name)
        if profile is None:
            available_profiles = ", ".join(sorted(self.settings.models))
            return self._error(
                f"Unknown model profile '{profile_name}'. "
                f"Available profiles: {available_profiles}."
            )

        self.current_model_profile_name = profile_name
        lines = [
            f"Selected model profile: {profile.name}",
            f"Provider: {profile.provider} | Model: {profile.model_name}",
        ]
        if self.thinking_enabled and not profile.thinking_supported:
            self.thinking_enabled = False
            lines.append(
                f"Thinking mode was turned off because '{profile.name}' "
                "does not support it."
            )

        return self._ok(*lines)

    def _handle_think(self, arguments: tuple[str, ...]) -> CommandResult:
        if not arguments:
            lines = [f"Thinking mode: {self.thinking_label}"]
            if not self.current_model_profile.thinking_supported:
                lines.append(
                    f"Current model profile '{self.current_model_profile.name}' "
                    "does not support thinking mode."
                )
            return self._ok(*lines)

        if len(arguments) != 1:
            return self._usage("/think <on|off>")

        value = arguments[0].lower()
        if value not in {"on", "off"}:
            return self._usage("/think <on|off>")

        if value == "on" and not self.current_model_profile.thinking_supported:
            return self._error(
                f"Model profile '{self.current_model_profile.name}' "
                "does not support thinking mode."
            )

        self.thinking_enabled = value == "on"
        return self._ok(f"Thinking mode: {self.thinking_label}")

    def _handle_tools(self, arguments: tuple[str, ...]) -> CommandResult:
        if arguments:
            return self._usage("/tools")
        return CommandResult(
            status=CommandStatus.OK,
            lines=(),
            list_tools=True,
        )

    def _handle_tool_command(
        self,
        parsed_command: ParsedCommand,
        *,
        usage_line: str,
    ) -> CommandResult:
        argument_value = self._read_freeform_argument(parsed_command)
        if argument_value is None:
            return self._usage(usage_line)

        tool_name = resolve_builtin_tool_command(parsed_command.name)
        if tool_name is None:
            return self._error(
                f"Unknown tool command '/{parsed_command.name}'. "
                "Use /help to list commands."
            )

        argument_key = "url" if parsed_command.name == "fetch" else "path"
        return CommandResult(
            status=CommandStatus.OK,
            lines=(),
            tool_call=ToolCall(
                tool_name=tool_name,
                arguments={argument_key: argument_value},
            ),
        )

    def _handle_help(self, arguments: tuple[str, ...]) -> CommandResult:
        if arguments:
            return self._usage("/help")

        lines = [
            "Available slash commands for this channel:",
            "",
            "Sessions:",
            "/new               Start a fresh session.",
            "/sessions          List recent sessions.",
            "/use <session_id>  Switch to an existing session.",
            "",
            "Models:",
            "/model                 Show the active model profile.",
            "/model <profile_name>  Switch model profiles.",
            "/think                 Show thinking mode status.",
            "/think on              Turn thinking mode on.",
            "/think off             Turn thinking mode off.",
            "",
            "Tools:",
            "/tools       List built-in tools.",
            "/read <path> Read one local file.",
            "/ls <path>   List one local directory.",
            "/fetch <url> Fetch one URL.",
            "",
            "Memory:",
            "/session  Show the current session state.",
            "/summary  Show the saved session summary.",
            "",
            "General:",
            "/help  Show this command list.",
        ]
        if self.allow_exit:
            lines.append("/exit  Leave the terminal runtime.")
        return self._ok(*lines)

    def _handle_exit(self, arguments: tuple[str, ...]) -> CommandResult:
        if not self.allow_exit:
            return self._error("The /exit command is only available in the CLI.")
        if arguments:
            return self._usage("/exit")
        return CommandResult(status=CommandStatus.EXIT, lines=("Exiting Unclaw.",))

    def _format_session_line(self, session: SessionSummary) -> str:
        marker = "*" if session.id == self.session_manager.current_session_id else " "
        return (
            f"{marker} {session.id} | {session.title} | "
            f"updated {session.updated_at}"
        )

    def _parse_command(self, raw_command: str) -> ParsedCommand:
        normalized = raw_command.strip()
        if not normalized.startswith("/"):
            raise ValueError("Commands must start with '/'.")
        if normalized == "/":
            raise ValueError("Empty command. Use /help to list available commands.")

        command_text = normalized[1:]
        command_name, _, raw_arguments = command_text.partition(" ")
        if not command_name:
            raise ValueError("Empty command. Use /help to list available commands.")

        try:
            arguments = tuple(shlex_split(raw_arguments)) if raw_arguments else ()
        except ValueError as exc:
            raise ValueError(f"Could not parse command arguments: {exc}") from exc

        return ParsedCommand(
            name=command_name.lower(),
            arguments=arguments,
            raw_arguments=raw_arguments,
        )

    def _read_freeform_argument(self, parsed_command: ParsedCommand) -> str | None:
        raw_value = parsed_command.raw_arguments.strip()
        if not raw_value:
            return None

        if len(parsed_command.arguments) == 1:
            value = parsed_command.arguments[0].strip()
            return value or None

        return raw_value

    def _ok(self, *lines: str) -> CommandResult:
        return CommandResult(status=CommandStatus.OK, lines=tuple(lines))

    def _error(self, *lines: str) -> CommandResult:
        return CommandResult(status=CommandStatus.ERROR, lines=tuple(lines))

    def _usage(self, usage_line: str) -> CommandResult:
        return self._error(f"Usage: {usage_line}")
