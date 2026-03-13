from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from unclaw.channels import telegram_bot
from unclaw.local_secrets import mask_telegram_bot_token
from unclaw.settings import load_settings

EXAMPLE_TELEGRAM_TOKEN = "123456789:AAExampleTelegramBotTokenValue"


class FakeApiClient:
    def __init__(self) -> None:
        self.sent_messages: list[tuple[int, str]] = []

    def send_message(self, *, chat_id: int, text: str) -> None:
        self.sent_messages.append((chat_id, text))


class FakeTracer:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []

    def trace_telegram_chat_rejected(self, **kwargs: object) -> None:
        self.events.append(("chat_rejected", dict(kwargs)))

    def trace_telegram_rate_limited(self, **kwargs: object) -> None:
        self.events.append(("rate_limited", dict(kwargs)))

    def trace_telegram_message_received(self, **kwargs: object) -> None:
        self.events.append(("message_received", dict(kwargs)))

    def trace_session_selected(self, **kwargs: object) -> None:
        self.events.append(("session_selected", dict(kwargs)))

    def trace_session_started(self, **kwargs: object) -> None:
        self.events.append(("session_started", dict(kwargs)))


def test_load_telegram_config_denies_all_when_allowlist_is_empty(
    tmp_path: Path,
) -> None:
    project_root = _create_temp_project(tmp_path)
    settings = load_settings(project_root=project_root)

    config = telegram_bot.load_telegram_config(settings)

    assert config.allowed_chat_ids == frozenset()
    assert config.is_chat_allowed(123456789) is False


def test_unauthorized_chat_is_rejected_without_running_model(
    monkeypatch,
    tmp_path: Path,
) -> None:
    channel, api_client, tracer = _build_channel(
        tmp_path,
        allowed_chat_ids=(),
    )

    def fail_activate(self, chat_id: int):  # type: ignore[no-untyped-def]
        raise AssertionError(f"Unauthorized chat {chat_id} should not activate a session.")

    monkeypatch.setattr(
        telegram_bot.TelegramBotChannel,
        "_activate_chat_session",
        fail_activate,
    )

    channel._handle_update(
        {
            "message": {
                "chat": {"id": 42},
                "date": 100,
                "text": "hello",
            }
        }
    )

    assert api_client.sent_messages == [(42, telegram_bot._UNAUTHORIZED_CHAT_MESSAGE)]
    assert tracer.events == [
        (
            "chat_rejected",
            {
                "chat_id": 42,
                "reason": "unauthorized",
            },
        )
    ]


def test_telegram_rate_limit_rejects_excess_burst_and_recovers(
    monkeypatch,
    tmp_path: Path,
) -> None:
    current_time = [110.0]
    channel, api_client, tracer = _build_channel(
        tmp_path,
        allowed_chat_ids=(42,),
        clock=lambda: current_time[0],
    )
    processed_messages: list[str] = []

    def fake_activate(self, chat_id: int):  # type: ignore[no-untyped-def]
        return SimpleNamespace(id=f"sess-{chat_id}", title=f"Telegram chat {chat_id}")

    def fake_get_command_handler(self, chat_id: int):  # type: ignore[no-untyped-def]
        return SimpleNamespace()

    def fake_handle_chat_turn(  # type: ignore[no-untyped-def]
        self,
        *,
        chat_id: int,
        text: str,
        session_id: str,
        command_handler,
    ) -> None:
        processed_messages.append(text)
        self._send_reply(chat_id=chat_id, text=f"ok:{text}")

    monkeypatch.setattr(
        telegram_bot.TelegramBotChannel,
        "_activate_chat_session",
        fake_activate,
    )
    monkeypatch.setattr(
        telegram_bot.TelegramBotChannel,
        "_get_command_handler",
        fake_get_command_handler,
    )
    monkeypatch.setattr(
        telegram_bot.TelegramBotChannel,
        "_handle_chat_turn",
        fake_handle_chat_turn,
    )

    for text, message_date in (
        ("one", 100),
        ("two", 101),
        ("three", 102),
        ("four", 103),
    ):
        channel._handle_update(
            {
                "message": {
                    "chat": {"id": 42},
                    "date": message_date,
                    "text": text,
                }
            }
        )

    current_time[0] = 140.0
    channel._handle_update(
        {
            "message": {
                "chat": {"id": 42},
                "date": 140,
                "text": "after",
            }
        }
    )

    assert processed_messages == ["one", "two", "three", "after"]
    assert api_client.sent_messages == [
        (42, "ok:one"),
        (42, "ok:two"),
        (42, "ok:three"),
        (42, telegram_bot._RATE_LIMITED_CHAT_MESSAGE),
        (42, "ok:after"),
    ]
    assert (
        "rate_limited",
        {
            "chat_id": 42,
            "pending_messages": 3,
            "max_pending_messages": 2,
        },
    ) in tracer.events


def test_telegram_api_errors_mask_bot_token(monkeypatch) -> None:
    client = telegram_bot.TelegramApiClient(bot_token=EXAMPLE_TELEGRAM_TOKEN)

    def fake_urlopen(request, timeout):  # type: ignore[no-untyped-def]
        raise OSError(f"request failed for {request.full_url}")

    monkeypatch.setattr(telegram_bot, "urlopen", fake_urlopen)

    with pytest.raises(telegram_bot.TelegramApiError) as exc_info:
        client.get_me()

    message = str(exc_info.value)
    assert EXAMPLE_TELEGRAM_TOKEN not in message
    assert mask_telegram_bot_token(EXAMPLE_TELEGRAM_TOKEN) in message


def test_telegram_main_uses_locally_stored_token_without_env(
    monkeypatch,
    tmp_path: Path,
) -> None:
    project_root = _create_temp_project(tmp_path)
    settings = load_settings(project_root=project_root)
    (project_root / "config" / "secrets.yaml").write_text(
        yaml.safe_dump(
            {"telegram": {"bot_token": EXAMPLE_TELEGRAM_TOKEN}},
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    captured: dict[str, str] = {}
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.setattr(telegram_bot, "bootstrap", lambda project_root=None: settings)
    monkeypatch.setattr(
        telegram_bot,
        "build_startup_report",
        lambda *args, **kwargs: SimpleNamespace(has_errors=False),
    )
    monkeypatch.setattr(telegram_bot, "format_startup_report", lambda report: "report")
    monkeypatch.setattr(telegram_bot, "build_banner", lambda **kwargs: "banner")

    class FakeSessionManager:
        def __init__(self) -> None:
            self.connection = object()
            self.event_repository = object()

        @classmethod
        def from_settings(cls, settings):  # type: ignore[no-untyped-def]
            return cls()

        def close(self) -> None:
            return None

    class FakeMemoryManager:
        def __init__(self, *, session_manager) -> None:  # type: ignore[no-untyped-def]
            self.session_manager = session_manager

    class FakeEventBus:
        pass

    class FakeTracer:
        def __init__(
            self,
            *,
            event_bus,
            event_repository,
            include_reasoning_text: bool = False,
        ) -> None:  # type: ignore[no-untyped-def]
            self.event_bus = event_bus
            self.event_repository = event_repository
            self.include_reasoning_text = include_reasoning_text

        def trace_model_profile_selected(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
            return None

    class FakeToolExecutor:
        @classmethod
        def with_default_tools(cls):  # type: ignore[no-untyped-def]
            return cls()

    class FakeTelegramApiClient:
        def __init__(
            self,
            *,
            bot_token: str,
            request_timeout_seconds: float = 40.0,
        ) -> None:
            captured["bot_token"] = bot_token
            self.bot_token = bot_token
            self.request_timeout_seconds = request_timeout_seconds

    class FakeTelegramChatSessionStore:
        def __init__(self, connection) -> None:  # type: ignore[no-untyped-def]
            self.connection = connection

        def initialize(self) -> None:
            return None

    class FakeTelegramBotChannel:
        def __init__(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
            self.kwargs = kwargs

        def run(self) -> None:
            return None

    monkeypatch.setattr(telegram_bot, "SessionManager", FakeSessionManager)
    monkeypatch.setattr(telegram_bot, "MemoryManager", FakeMemoryManager)
    monkeypatch.setattr(telegram_bot, "EventBus", FakeEventBus)
    monkeypatch.setattr(telegram_bot, "Tracer", FakeTracer)
    monkeypatch.setattr(telegram_bot, "ToolExecutor", FakeToolExecutor)
    monkeypatch.setattr(telegram_bot, "TelegramApiClient", FakeTelegramApiClient)
    monkeypatch.setattr(
        telegram_bot,
        "TelegramChatSessionStore",
        FakeTelegramChatSessionStore,
    )
    monkeypatch.setattr(telegram_bot, "TelegramBotChannel", FakeTelegramBotChannel)

    result = telegram_bot.main(project_root=project_root)

    assert result == 0
    assert captured["bot_token"] == EXAMPLE_TELEGRAM_TOKEN


def _build_channel(
    tmp_path: Path,
    *,
    allowed_chat_ids: tuple[int, ...],
    clock=lambda: 0.0,
) -> tuple[telegram_bot.TelegramBotChannel, FakeApiClient, FakeTracer]:
    project_root = _create_temp_project(tmp_path)
    settings = load_settings(project_root=project_root)
    api_client = FakeApiClient()
    tracer = FakeTracer()
    channel = telegram_bot.TelegramBotChannel(
        settings=settings,
        config=telegram_bot.TelegramConfig(
            bot_token_env_var="TELEGRAM_BOT_TOKEN",
            polling_timeout_seconds=30,
            allowed_chat_ids=frozenset(allowed_chat_ids),
        ),
        session_manager=SimpleNamespace(current_session_id=None),
        memory_manager=SimpleNamespace(),
        tracer=tracer,
        tool_executor=SimpleNamespace(list_tools=lambda: []),
        api_client=api_client,
        session_store=SimpleNamespace(),
        clock=clock,
    )
    return channel, api_client, tracer


def _create_temp_project(tmp_path: Path) -> Path:
    source_root = Path(__file__).resolve().parents[1]
    project_root = tmp_path / "project"
    shutil.copytree(source_root / "config", project_root / "config")
    return project_root
