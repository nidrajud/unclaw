"""Slash command parsing and handling for the CLI channel."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from unclaw.core.session_manager import SessionManager, SessionManagerError
from unclaw.schemas.session import SessionSummary
from unclaw.settings import ModelProfile, Settings


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

    @property
    def should_exit(self) -> bool:
        return self.status is CommandStatus.EXIT


@dataclass(slots=True)
class CommandHandler:
    """Handle slash commands without executing any model calls."""

    settings: Settings
    session_manager: SessionManager
    current_model_profile_name: str | None = None
    thinking_enabled: bool | None = None

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
        normalized = raw_command.strip()
        if not normalized.startswith("/"):
            return self._error("Commands must start with '/'.")

        if normalized == "/":
            return self._error("Empty command. Use /help to list available commands.")

        parts = normalized[1:].split()
        command = parts[0].lower()
        arguments = parts[1:]

        try:
            match command:
                case "new":
                    return self._handle_new(arguments)
                case "sessions":
                    return self._handle_sessions(arguments)
                case "use":
                    return self._handle_use(arguments)
                case "model":
                    return self._handle_model(arguments)
                case "think":
                    return self._handle_think(arguments)
                case "help":
                    return self._handle_help(arguments)
                case "exit":
                    return self._handle_exit(arguments)
                case _:
                    return self._error(
                        f"Unknown command '/{command}'. Use /help to list commands."
                    )
        except SessionManagerError as exc:
            return self._error(str(exc))

    def _handle_new(self, arguments: list[str]) -> CommandResult:
        if arguments:
            return self._usage("/new")

        session = self.session_manager.create_session()
        return self._ok(
            f"Created and switched to session {session.id}.",
            f"Title: {session.title}",
        )

    def _handle_sessions(self, arguments: list[str]) -> CommandResult:
        if arguments:
            return self._usage("/sessions")

        sessions = self.session_manager.list_sessions()
        if not sessions:
            return self._ok("No sessions found.")

        lines = ["Recent sessions:"]
        for session in sessions:
            lines.append(self._format_session_line(session))
        return self._ok(*lines)

    def _handle_use(self, arguments: list[str]) -> CommandResult:
        if len(arguments) != 1:
            return self._usage("/use <session_id>")

        session = self.session_manager.switch_session(arguments[0])
        return self._ok(
            f"Switched to session {session.id}.",
            f"Title: {session.title}",
        )

    def _handle_model(self, arguments: list[str]) -> CommandResult:
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

    def _handle_think(self, arguments: list[str]) -> CommandResult:
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

    def _handle_help(self, arguments: list[str]) -> CommandResult:
        if arguments:
            return self._usage("/help")

        return self._ok(
            "Available commands:",
            "/new",
            "/sessions",
            "/use <session_id>",
            "/model",
            "/model <profile_name>",
            "/think",
            "/think on",
            "/think off",
            "/help",
            "/exit",
        )

    def _handle_exit(self, arguments: list[str]) -> CommandResult:
        if arguments:
            return self._usage("/exit")
        return CommandResult(status=CommandStatus.EXIT, lines=("Exiting Unclaw.",))

    def _format_session_line(self, session: SessionSummary) -> str:
        marker = "*" if session.id == self.session_manager.current_session_id else " "
        return (
            f"{marker} {session.id} | {session.title} | "
            f"updated {session.updated_at}"
        )

    def _ok(self, *lines: str) -> CommandResult:
        return CommandResult(status=CommandStatus.OK, lines=tuple(lines))

    def _error(self, *lines: str) -> CommandResult:
        return CommandResult(status=CommandStatus.ERROR, lines=tuple(lines))

    def _usage(self, usage_line: str) -> CommandResult:
        return self._error(f"Usage: {usage_line}")

