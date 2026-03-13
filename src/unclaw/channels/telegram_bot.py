"""Polling-based Telegram channel for the Unclaw runtime."""

from __future__ import annotations

import json
import logging
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import yaml

from unclaw.bootstrap import bootstrap
from unclaw.core.command_handler import CommandHandler, CommandResult, CommandStatus
from unclaw.core.executor import ToolExecutor
from unclaw.core.runtime import run_user_turn
from unclaw.core.session_manager import SessionManager
from unclaw.errors import ConfigurationError, UnclawError
from unclaw.llm.base import utc_now_iso
from unclaw.local_secrets import (
    resolve_telegram_bot_token,
    validate_telegram_token_env_var_name,
)
from unclaw.logs.event_bus import EventBus
from unclaw.logs.tracer import Tracer
from unclaw.memory import MemoryManager
from unclaw.schemas.chat import MessageRole
from unclaw.schemas.session import SessionRecord
from unclaw.settings import Settings
from unclaw.startup import build_banner, build_startup_report, format_startup_report
from unclaw.tools.contracts import ToolDefinition, ToolResult

LOGGER = logging.getLogger(__name__)
_TELEGRAM_CONFIG_FILE_NAME = "telegram.yaml"
_TELEGRAM_API_BASE_URL = "https://api.telegram.org"
_POLL_RETRY_DELAY_SECONDS = 3.0
_MESSAGE_LIMIT = 4000


@dataclass(frozen=True, slots=True)
class TelegramConfig:
    """Minimal Telegram channel configuration."""

    bot_token_env_var: str
    polling_timeout_seconds: int
    allowed_chat_ids: frozenset[int]

    def is_chat_allowed(self, chat_id: int) -> bool:
        if not self.allowed_chat_ids:
            return True
        return chat_id in self.allowed_chat_ids


class TelegramApiError(UnclawError):
    """Raised when the Telegram Bot API request fails."""


@dataclass(slots=True)
class TelegramApiClient:
    """Small synchronous Telegram Bot API client."""

    bot_token: str
    request_timeout_seconds: float = 40.0

    @property
    def api_base_url(self) -> str:
        return f"{_TELEGRAM_API_BASE_URL}/bot{self.bot_token}"

    def get_me(self) -> dict[str, Any]:
        result = self._request("getMe", {})
        if not isinstance(result, dict):
            raise TelegramApiError("Telegram returned an invalid bot profile payload.")
        return result

    def get_updates(
        self,
        *,
        offset: int | None,
        timeout_seconds: int,
    ) -> list[dict[str, Any]]:
        payload: dict[str, Any] = {
            "timeout": timeout_seconds,
            "allowed_updates": ["message"],
        }
        if offset is not None:
            payload["offset"] = offset

        result = self._request("getUpdates", payload)
        if not isinstance(result, list):
            raise TelegramApiError("Telegram returned an invalid updates payload.")
        return [update for update in result if isinstance(update, dict)]

    def send_message(self, *, chat_id: int, text: str) -> None:
        self._request(
            "sendMessage",
            {
                "chat_id": chat_id,
                "text": text,
            },
        )

    def _request(self, method: str, payload: dict[str, Any]) -> Any:
        request = Request(
            url=f"{self.api_base_url}/{method}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.request_timeout_seconds) as response:
                raw_body = response.read().decode("utf-8")
        except HTTPError as exc:
            raise TelegramApiError(
                f"Telegram API request failed with HTTP {exc.code}: "
                f"{_read_http_error_body(exc)}"
            ) from exc
        except URLError as exc:
            raise TelegramApiError(
                f"Could not reach the Telegram API: {exc.reason}"
            ) from exc
        except OSError as exc:
            raise TelegramApiError(f"Telegram API request failed: {exc}") from exc

        try:
            payload = json.loads(raw_body)
        except json.JSONDecodeError as exc:
            raise TelegramApiError("Telegram returned invalid JSON.") from exc

        if not isinstance(payload, dict):
            raise TelegramApiError("Telegram returned an invalid response payload.")
        if payload.get("ok") is not True:
            description = payload.get("description")
            if isinstance(description, str) and description.strip():
                raise TelegramApiError(description.strip())
            raise TelegramApiError("Telegram returned an unsuccessful response.")

        return payload.get("result")


@dataclass(slots=True)
class TelegramChatSessionStore:
    """Persist Telegram chat-to-session bindings in the local SQLite database."""

    connection: sqlite3.Connection

    def initialize(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS telegram_chat_sessions (
                chat_id INTEGER PRIMARY KEY,
                session_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions (id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_telegram_chat_sessions_session_id
            ON telegram_chat_sessions (session_id);
            """
        )
        self.connection.commit()

    def get_session_id(self, chat_id: int) -> str | None:
        row = self.connection.execute(
            """
            SELECT session_id
            FROM telegram_chat_sessions
            WHERE chat_id = ?
            """,
            (chat_id,),
        ).fetchone()
        if row is None:
            return None
        value = row["session_id"]
        if not isinstance(value, str) or not value.strip():
            return None
        return value

    def bind_chat(self, *, chat_id: int, session_id: str) -> None:
        timestamp = utc_now_iso()
        self.connection.execute(
            """
            INSERT INTO telegram_chat_sessions (
                chat_id,
                session_id,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?)
            ON CONFLICT(chat_id) DO UPDATE SET
                session_id = excluded.session_id,
                updated_at = excluded.updated_at
            """,
            (chat_id, session_id, timestamp, timestamp),
        )
        self.connection.commit()


@dataclass(slots=True)
class TelegramBotChannel:
    """Bridge Telegram messages into the shared local runtime."""

    settings: Settings
    config: TelegramConfig
    session_manager: SessionManager
    memory_manager: MemoryManager
    tracer: Tracer
    tool_executor: ToolExecutor
    api_client: TelegramApiClient
    session_store: TelegramChatSessionStore
    command_handlers: dict[int, CommandHandler] = field(default_factory=dict)

    def run(self) -> None:
        bot_profile = self.api_client.get_me()
        username = bot_profile.get("username", "<unknown>")
        self.tracer.trace_channel_started(
            channel_name="telegram",
            model_profile_name=self.settings.app.default_model_profile,
            extra_payload={
                "polling_timeout_seconds": self.config.polling_timeout_seconds,
                "username": username,
            },
        )
        LOGGER.info(
            "Telegram polling started for @%s with timeout=%ss",
            username,
            self.config.polling_timeout_seconds,
        )

        next_update_offset: int | None = None
        while True:
            try:
                updates = self.api_client.get_updates(
                    offset=next_update_offset,
                    timeout_seconds=self.config.polling_timeout_seconds,
                )
            except TelegramApiError as exc:
                LOGGER.error("Telegram polling failed: %s", exc)
                time.sleep(_POLL_RETRY_DELAY_SECONDS)
                continue

            for update in updates:
                update_id = update.get("update_id")
                if isinstance(update_id, int):
                    next_update_offset = update_id + 1

                try:
                    self._handle_update(update)
                except TelegramApiError as exc:
                    LOGGER.error("Telegram transport failed for one update: %s", exc)
                except Exception:
                    LOGGER.exception("Unexpected Telegram update failure.")

    def _handle_update(self, update: dict[str, Any]) -> None:
        message = update.get("message")
        if not isinstance(message, dict):
            return

        chat = message.get("chat")
        if not isinstance(chat, dict):
            return

        chat_id = chat.get("id")
        if not isinstance(chat_id, int):
            return

        if not self.config.is_chat_allowed(chat_id):
            LOGGER.warning("Ignoring message from unauthorized chat %s", chat_id)
            self._send_reply(
                chat_id=chat_id,
                text="This Unclaw bot is not allowed in this chat.",
            )
            return

        text = message.get("text")
        if not isinstance(text, str) or not text.strip():
            self._send_reply(
                chat_id=chat_id,
                text="Please send a text message or a slash command.",
            )
            return

        active_session = self._activate_chat_session(chat_id)
        command_handler = self._get_command_handler(chat_id)
        LOGGER.info(
            "Handling Telegram message for chat=%s session=%s",
            chat_id,
            active_session.id,
        )

        normalized_text = text.strip()
        self.tracer.trace_telegram_message_received(
            session_id=active_session.id,
            chat_id=chat_id,
            text_length=len(normalized_text),
            is_command=normalized_text.startswith("/"),
        )
        if normalized_text.startswith("/"):
            self._handle_command(
                chat_id=chat_id,
                text=normalized_text,
                command_handler=command_handler,
            )
            return

        self._handle_chat_turn(
            chat_id=chat_id,
            text=normalized_text,
            session_id=active_session.id,
            command_handler=command_handler,
        )

    def _activate_chat_session(self, chat_id: int) -> SessionRecord:
        mapped_session_id = self.session_store.get_session_id(chat_id)
        if mapped_session_id is not None:
            mapped_session = self.session_manager.load_session(mapped_session_id)
            if mapped_session is not None:
                active_session = self.session_manager.switch_session(mapped_session.id)
                self.tracer.trace_session_selected(
                    session_id=active_session.id,
                    title=active_session.title,
                    reason="telegram_chat",
                )
                return active_session

        session = self.session_manager.create_session(title=f"Telegram chat {chat_id}")
        self.session_store.bind_chat(chat_id=chat_id, session_id=session.id)
        self.tracer.trace_session_started(
            session_id=session.id,
            title=session.title,
            source="telegram_chat",
        )
        return session

    def _get_command_handler(self, chat_id: int) -> CommandHandler:
        handler = self.command_handlers.get(chat_id)
        if handler is not None:
            return handler

        handler = CommandHandler(
            settings=self.settings,
            session_manager=self.session_manager,
            memory_manager=self.memory_manager,
            tracer=self.tracer,
            allow_exit=False,
        )
        self.command_handlers[chat_id] = handler
        return handler

    def _handle_command(
        self,
        *,
        chat_id: int,
        text: str,
        command_handler: CommandHandler,
    ) -> None:
        normalized_command = _normalize_telegram_command(text)
        if normalized_command == "/start" or normalized_command.startswith("/start "):
            normalized_command = "/help"

        result = command_handler.handle(normalized_command)
        self._persist_chat_binding(chat_id)

        if result.list_tools:
            reply_text = _format_tool_list(self.tool_executor.list_tools())
        elif result.tool_call is not None:
            session_id = self.session_manager.current_session_id
            self.tracer.trace_tool_started(
                session_id=session_id,
                tool_name=result.tool_call.tool_name,
                arguments=result.tool_call.arguments,
            )
            tool_started_at = time.perf_counter()
            tool_result = self.tool_executor.execute(result.tool_call)
            self.tracer.trace_tool_finished(
                session_id=session_id,
                tool_name=result.tool_call.tool_name,
                success=tool_result.success,
                output_length=len(tool_result.output_text),
                error=tool_result.error,
                tool_duration_ms=_elapsed_ms(tool_started_at),
            )
            reply_text = _format_tool_result(tool_result)
        elif result.should_exit:
            reply_text = "The Telegram bot keeps running. Use /help to see commands."
        else:
            reply_text = _format_command_result(result)

        if reply_text:
            self._send_reply(chat_id=chat_id, text=reply_text)

    def _handle_chat_turn(
        self,
        *,
        chat_id: int,
        text: str,
        session_id: str,
        command_handler: CommandHandler,
    ) -> None:
        self.session_manager.add_message(
            MessageRole.USER,
            text,
            session_id=session_id,
        )
        assistant_reply = run_user_turn(
            session_manager=self.session_manager,
            command_handler=command_handler,
            user_input=text,
            tracer=self.tracer,
        )
        try:
            self.memory_manager.build_or_refresh_session_summary(session_id)
        except UnclawError as exc:
            LOGGER.warning("Could not refresh session summary for %s: %s", session_id, exc)

        self._persist_chat_binding(chat_id)
        self._send_reply(chat_id=chat_id, text=assistant_reply)

    def _persist_chat_binding(self, chat_id: int) -> None:
        current_session_id = self.session_manager.current_session_id
        if current_session_id is None:
            return
        self.session_store.bind_chat(chat_id=chat_id, session_id=current_session_id)

    def _send_reply(self, *, chat_id: int, text: str) -> None:
        for chunk in _split_message_chunks(text):
            self.api_client.send_message(chat_id=chat_id, text=chunk)


def load_telegram_config(settings: Settings) -> TelegramConfig:
    """Load and validate the Telegram channel configuration file."""

    config_path = settings.paths.config_dir / _TELEGRAM_CONFIG_FILE_NAME
    payload = _load_yaml_mapping(config_path)

    bot_token_env_var = validate_telegram_token_env_var_name(
        _read_str(payload, "bot_token_env_var")
    )
    polling_timeout_seconds = _read_int(payload, "polling_timeout_seconds", minimum=1)
    allowed_chat_ids = _read_allowed_chat_ids(payload.get("allowed_chat_ids"))

    return TelegramConfig(
        bot_token_env_var=bot_token_env_var,
        polling_timeout_seconds=polling_timeout_seconds,
        allowed_chat_ids=allowed_chat_ids,
    )


def main(project_root: Path | None = None) -> int:
    """Run the Telegram polling channel."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    session_manager: SessionManager | None = None
    try:
        settings = bootstrap(project_root=project_root)
        telegram_config = load_telegram_config(settings)
        startup_report = build_startup_report(
            settings,
            channel_name="telegram",
            channel_enabled=settings.app.channels.telegram_enabled,
            required_profile_names=(settings.app.default_model_profile,),
            optional_profile_names=tuple(
                profile_name
                for profile_name in settings.models
                if profile_name != settings.app.default_model_profile
            ),
            telegram_token_env_var=telegram_config.bot_token_env_var,
        )
        print(
            build_banner(
                title="Unclaw Telegram",
                subtitle="Local-first bot channel backed by your local model runtime.",
                rows=(
                    ("mode", "telegram"),
                    (
                        "default",
                        (
                            f"{settings.app.default_model_profile} -> "
                            f"{settings.default_model.model_name}"
                        ),
                    ),
                    ("logging", settings.app.logging.mode),
                    ("polling", f"{telegram_config.polling_timeout_seconds}s"),
                ),
            )
        )
        print(format_startup_report(startup_report))
        if startup_report.has_errors:
            return 1

        resolved_bot_token = resolve_telegram_bot_token(
            settings,
            bot_token_env_var=telegram_config.bot_token_env_var,
        )
        if resolved_bot_token is None:
            raise ConfigurationError(
                "Telegram bot token is missing. "
                "Run `unclaw onboard` and paste it into the local project secrets "
                "file, or use the advanced fallback "
                f"{telegram_config.bot_token_env_var} environment variable."
            )

        session_manager = SessionManager.from_settings(settings)
        memory_manager = MemoryManager(session_manager=session_manager)
        event_bus = EventBus()
        tracer = Tracer(
            event_bus=event_bus,
            event_repository=session_manager.event_repository,
        )
        tracer.runtime_log_path = (
            settings.paths.log_file_path if settings.app.logging.file_enabled else None
        )
        tracer.trace_model_profile_selected(
            session_id=None,
            model_profile_name=settings.app.default_model_profile,
            provider=settings.default_model.provider,
            model_name=settings.default_model.model_name,
            reason="startup",
        )
        tool_executor = ToolExecutor.with_default_tools()
        api_client = TelegramApiClient(bot_token=resolved_bot_token.value)
        session_store = TelegramChatSessionStore(session_manager.connection)
        session_store.initialize()

        bot = TelegramBotChannel(
            settings=settings,
            config=telegram_config,
            session_manager=session_manager,
            memory_manager=memory_manager,
            tracer=tracer,
            tool_executor=tool_executor,
            api_client=api_client,
            session_store=session_store,
        )
        bot.run()
        return 0
    except KeyboardInterrupt:
        print("\nStopping Unclaw Telegram bot.")
        return 0
    except UnclawError as exc:
        print(f"Failed to start Unclaw Telegram bot: {exc}", file=sys.stderr)
        return 1
    finally:
        if session_manager is not None:
            session_manager.close()


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigurationError(f"Missing Telegram configuration file: {path}") from exc
    except OSError as exc:
        raise ConfigurationError(f"Could not read Telegram configuration file: {path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Invalid YAML in Telegram configuration file: {path}") from exc

    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ConfigurationError(
            f"Telegram configuration must contain a mapping: {path}"
        )
    return payload


def _read_bool(
    payload: dict[str, Any],
    key: str,
    *,
    default: bool | None = None,
) -> bool:
    value = payload.get(key, default)
    if isinstance(value, bool):
        return value
    if value is None:
        raise ConfigurationError(f"Missing Telegram setting '{key}'.")
    raise ConfigurationError(f"Telegram setting '{key}' must be a boolean.")


def _read_str(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"Telegram setting '{key}' must be a non-empty string.")
    return value.strip()


def _read_int(
    payload: dict[str, Any],
    key: str,
    *,
    minimum: int | None = None,
) -> int:
    value = payload.get(key)
    if not isinstance(value, int):
        raise ConfigurationError(f"Telegram setting '{key}' must be an integer.")
    if minimum is not None and value < minimum:
        raise ConfigurationError(
            f"Telegram setting '{key}' must be greater than or equal to {minimum}."
        )
    return value


def _read_allowed_chat_ids(value: Any) -> frozenset[int]:
    if value is None:
        return frozenset()
    if not isinstance(value, list):
        raise ConfigurationError("Telegram setting 'allowed_chat_ids' must be a list.")

    chat_ids: set[int] = set()
    for raw_chat_id in value:
        if isinstance(raw_chat_id, bool):
            raise ConfigurationError(
                "Telegram chat ids must be integers, not boolean values."
            )
        if isinstance(raw_chat_id, int):
            chat_ids.add(raw_chat_id)
            continue
        if isinstance(raw_chat_id, str):
            stripped_value = raw_chat_id.strip()
            if not stripped_value:
                raise ConfigurationError("Telegram chat ids must not be empty strings.")
            try:
                chat_ids.add(int(stripped_value))
            except ValueError as exc:
                raise ConfigurationError(
                    "Telegram chat ids must be integers."
                ) from exc
            continue
        raise ConfigurationError("Telegram chat ids must be integers.")
    return frozenset(chat_ids)


def _normalize_telegram_command(text: str) -> str:
    command, separator, remainder = text.partition(" ")
    command_name, mention_separator, _mention = command.partition("@")
    if not mention_separator:
        return text
    return f"{command_name}{separator}{remainder}".strip()


def _format_command_result(result: CommandResult) -> str:
    if not result.lines:
        return ""

    if result.status is CommandStatus.ERROR:
        first_line, *other_lines = result.lines
        lines = [f"Error: {first_line}", *other_lines]
        return "\n".join(lines)

    return "\n".join(result.lines)


def _format_tool_list(tools: list[ToolDefinition]) -> str:
    if not tools:
        return "No built-in tools available."

    lines = ["Built-in tools:"]
    for tool in tools:
        lines.append(
            f"- {tool.name} [{tool.permission_level.value}] {tool.description}"
        )
    return "\n".join(lines)


def _format_tool_result(result: ToolResult) -> str:
    if result.success:
        return result.output_text

    if not result.output_text:
        return f"Error: {result.error}"

    lines = result.output_text.splitlines() or [result.output_text]
    first_line, *other_lines = lines
    return "\n".join([f"Error: {first_line}", *other_lines])


def _split_message_chunks(text: str, *, limit: int = _MESSAGE_LIMIT) -> list[str]:
    normalized = text.strip()
    if not normalized:
        return []

    chunks: list[str] = []
    remaining = normalized
    while len(remaining) > limit:
        split_at = remaining.rfind("\n", 0, limit + 1)
        if split_at < limit // 2:
            split_at = remaining.rfind(" ", 0, limit + 1)
        if split_at < limit // 2:
            split_at = limit

        chunks.append(remaining[:split_at].rstrip())
        remaining = remaining[split_at:].lstrip()

    chunks.append(remaining)
    return chunks


def _elapsed_ms(started_at: float) -> int:
    return max(0, round((time.perf_counter() - started_at) * 1000))


def _read_http_error_body(exc: HTTPError) -> str:
    try:
        raw_body = exc.read().decode("utf-8").strip()
    except OSError:
        return exc.reason or "Unknown Telegram error."

    if not raw_body:
        return exc.reason or "Unknown Telegram error."

    try:
        payload = json.loads(raw_body)
    except json.JSONDecodeError:
        return raw_body

    if isinstance(payload, dict):
        description = payload.get("description")
        if isinstance(description, str) and description.strip():
            return description.strip()
    return raw_body


if __name__ == "__main__":
    raise SystemExit(main())
