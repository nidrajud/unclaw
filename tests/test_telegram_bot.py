from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from unclaw.channels import telegram_bot
from unclaw.core.command_handler import CommandResult, CommandStatus
from unclaw.core.session_manager import SessionManager
from unclaw.local_secrets import mask_telegram_bot_token
from unclaw.logs.event_bus import EventBus
from unclaw.logs.tracer import Tracer
from unclaw.schemas.chat import MessageRole
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

    def trace_tool_started(self, **kwargs: object) -> None:
        self.events.append(("tool_started", dict(kwargs)))

    def trace_tool_finished(self, **kwargs: object) -> None:
        self.events.append(("tool_finished", dict(kwargs)))


def test_load_telegram_config_denies_all_when_allowlist_is_empty(
    tmp_path: Path,
) -> None:
    project_root = _create_temp_project(tmp_path, allowed_chat_ids=[])
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

    assert api_client.sent_messages == [
        (42, telegram_bot._build_unauthorized_chat_message(42))
    ]
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


def test_telegram_management_commands_update_local_allowlist(
    capsys,
    tmp_path: Path,
) -> None:
    project_root = _create_temp_project(tmp_path, allowed_chat_ids=[])
    settings = load_settings(project_root=project_root)

    assert telegram_bot.main(project_root=project_root, command="allow", chat_id=42) == 0
    assert telegram_bot.load_telegram_config(settings).allowed_chat_ids == frozenset(
        {42}
    )
    assert "Authorized Telegram chat 42." in capsys.readouterr().out

    assert telegram_bot.main(project_root=project_root, command="list") == 0
    listed_output = capsys.readouterr().out
    assert "Authorized Telegram chats:" in listed_output
    assert "- 42" in listed_output

    assert telegram_bot.main(project_root=project_root, command="revoke", chat_id=42) == 0
    assert telegram_bot.load_telegram_config(settings).allowed_chat_ids == frozenset()
    assert "Revoked Telegram chat 42." in capsys.readouterr().out


def test_rejected_chat_stays_blocked_until_it_is_explicitly_allowed(
    monkeypatch,
    tmp_path: Path,
) -> None:
    project_root = _create_temp_project(tmp_path, allowed_chat_ids=[])
    settings = load_settings(project_root=project_root)

    blocked_channel, blocked_api_client, _ = _build_channel_for_settings(
        settings,
        config=telegram_bot.load_telegram_config(settings),
    )
    blocked_channel._handle_update(
        {
            "message": {
                "chat": {"id": 42},
                "date": 100,
                "text": "hello",
            }
        }
    )

    still_blocked_channel, still_blocked_api_client, _ = _build_channel_for_settings(
        settings,
        config=telegram_bot.load_telegram_config(settings),
    )
    still_blocked_channel._handle_update(
        {
            "message": {
                "chat": {"id": 42},
                "date": 101,
                "text": "hello again",
            }
        }
    )

    assert blocked_api_client.sent_messages == [
        (42, telegram_bot._build_unauthorized_chat_message(42))
    ]
    assert still_blocked_api_client.sent_messages == [
        (42, telegram_bot._build_unauthorized_chat_message(42))
    ]

    assert telegram_bot.main(project_root=project_root, command="allow", chat_id=42) == 0

    allowed_channel, allowed_api_client, _ = _build_channel_for_settings(
        settings,
        config=telegram_bot.load_telegram_config(settings),
    )

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

    allowed_channel._handle_update(
        {
            "message": {
                "chat": {"id": 42},
                "date": 102,
                "text": "hello after allow",
            }
        }
    )

    assert allowed_api_client.sent_messages == [(42, "ok:hello after allow")]


def test_telegram_allow_latest_uses_logged_rejected_chat(
    capsys,
    tmp_path: Path,
) -> None:
    project_root = _create_temp_project(tmp_path, allowed_chat_ids=[])
    settings = load_settings(project_root=project_root)
    session_manager = SessionManager.from_settings(settings)
    tracer = Tracer(
        event_bus=EventBus(),
        event_repository=session_manager.event_repository,
    )

    tracer.trace_telegram_chat_rejected(chat_id=77, reason="unauthorized")
    session_manager.close()

    assert telegram_bot.main(project_root=project_root, command="allow-latest") == 0
    assert telegram_bot.load_telegram_config(settings).allowed_chat_ids == frozenset(
        {77}
    )
    assert "77" in capsys.readouterr().out


def test_telegram_allow_command_normalizes_duplicates_and_preserves_config_values(
    tmp_path: Path,
) -> None:
    project_root = _create_temp_project(tmp_path, allowed_chat_ids=[])
    telegram_config_path = project_root / "config" / "telegram.yaml"
    telegram_config_path.write_text(
        yaml.safe_dump(
            {
                "bot_token_env_var": "TELEGRAM_BOT_TOKEN",
                "polling_timeout_seconds": 30,
                "notes": "keep me",
                "allowed_chat_ids": ["42", 42],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    assert telegram_bot.main(project_root=project_root, command="allow", chat_id=42) == 0

    payload = yaml.safe_load(telegram_config_path.read_text(encoding="utf-8"))
    assert payload["bot_token_env_var"] == "TELEGRAM_BOT_TOKEN"
    assert payload["polling_timeout_seconds"] == 30
    assert payload["notes"] == "keep me"
    assert payload["allowed_chat_ids"] == [42]


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
        def with_default_tools(cls, settings=None):  # type: ignore[no-untyped-def]
            del settings
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


def test_telegram_tool_traces_use_explicit_chat_session_id(
    monkeypatch,
    tmp_path: Path,
) -> None:
    project_root = _create_temp_project(tmp_path, allowed_chat_ids=[42])
    settings = load_settings(project_root=project_root)
    api_client = FakeApiClient()
    tracer = FakeTracer()
    bound_sessions: list[tuple[int, str]] = []
    session_manager = SimpleNamespace(current_session_id="stale-session")

    channel = telegram_bot.TelegramBotChannel(
        settings=settings,
        config=telegram_bot.TelegramConfig(
            bot_token_env_var="TELEGRAM_BOT_TOKEN",
            polling_timeout_seconds=30,
            allowed_chat_ids=frozenset({42}),
        ),
        session_manager=session_manager,
        memory_manager=SimpleNamespace(),
        tracer=tracer,
        tool_executor=SimpleNamespace(
            list_tools=lambda: [],
            execute=lambda _tool_call: SimpleNamespace(
                tool_name="read_text_file",
                success=True,
                output_text="tool output",
                error=None,
            ),
        ),
        api_client=api_client,
        session_store=SimpleNamespace(
            bind_chat=lambda *, chat_id, session_id: bound_sessions.append(
                (chat_id, session_id)
            )
        ),
    )

    monkeypatch.setattr(
        telegram_bot.TelegramBotChannel,
        "_activate_chat_session",
        lambda self, chat_id: SimpleNamespace(
            id=f"chat-session-{chat_id}",
            title=f"Telegram chat {chat_id}",
        ),
    )
    monkeypatch.setattr(
        telegram_bot.TelegramBotChannel,
        "_get_command_handler",
        lambda self, chat_id: SimpleNamespace(
            handle=lambda _text: CommandResult(
                status=CommandStatus.OK,
                lines=(),
                tool_call=SimpleNamespace(
                    tool_name="read_text_file",
                    arguments={"path": "README.md"},
                ),
            )
        ),
    )

    channel._handle_update(
        {
            "message": {
                "chat": {"id": 42},
                "date": 100,
                "text": "/read README.md",
            }
        }
    )

    tool_started = next(
        payload for event_name, payload in tracer.events if event_name == "tool_started"
    )
    tool_finished = next(
        payload for event_name, payload in tracer.events if event_name == "tool_finished"
    )
    assert tool_started["session_id"] == "chat-session-42"
    assert tool_started["tool_name"] == "read_text_file"
    assert tool_started["arguments"] == {"path": "README.md"}
    assert tool_finished["session_id"] == "chat-session-42"
    assert tool_finished["tool_name"] == "read_text_file"
    assert tool_finished["success"] is True
    assert tool_finished["output_length"] == 11
    assert isinstance(tool_finished["tool_duration_ms"], int)
    assert bound_sessions == [(42, "chat-session-42")]


def test_telegram_tool_results_are_persisted_for_follow_up_grounding(
    monkeypatch,
    tmp_path: Path,
) -> None:
    project_root = _create_temp_project(tmp_path, allowed_chat_ids=[42])
    settings = load_settings(project_root=project_root)
    session_manager = SessionManager.from_settings(settings)
    api_client = FakeApiClient()
    tracer = FakeTracer()

    try:
        session = session_manager.create_session(title="Telegram chat 42")
        channel = telegram_bot.TelegramBotChannel(
            settings=settings,
            config=telegram_bot.TelegramConfig(
                bot_token_env_var="TELEGRAM_BOT_TOKEN",
                polling_timeout_seconds=30,
                allowed_chat_ids=frozenset({42}),
            ),
            session_manager=session_manager,
            memory_manager=SimpleNamespace(),
            tracer=tracer,
            tool_executor=SimpleNamespace(
                list_tools=lambda: [],
                execute=lambda _tool_call: SimpleNamespace(
                    tool_name="search_web",
                    success=True,
                    output_text=(
                        "Search query: latest news about Ollama\n"
                        "Summary:\n"
                        "- I searched 2 public result(s) and read 2 top source(s) directly.\n"
                    ),
                    error=None,
                ),
            ),
            api_client=api_client,
            session_store=SimpleNamespace(bind_chat=lambda **kwargs: None),
        )

        monkeypatch.setattr(
            telegram_bot.TelegramBotChannel,
            "_activate_chat_session",
            lambda self, chat_id: session_manager.load_session(session.id),
        )
        monkeypatch.setattr(
            telegram_bot.TelegramBotChannel,
            "_get_command_handler",
            lambda self, chat_id: SimpleNamespace(
                handle=lambda _text: CommandResult(
                    status=CommandStatus.OK,
                    lines=(),
                    tool_call=SimpleNamespace(
                        tool_name="search_web",
                        arguments={"query": "latest news about Ollama"},
                    ),
                )
            ),
        )

        channel._handle_update(
            {
                "message": {
                    "chat": {"id": 42},
                    "date": 100,
                    "text": "/search latest news about Ollama",
                }
            }
        )

        messages = session_manager.list_messages(session.id)
        assert messages[-1].role is MessageRole.TOOL
        assert "Tool: search_web" in messages[-1].content
        assert "Outcome: success" in messages[-1].content
        assert "Search query: latest news about Ollama" in messages[-1].content
    finally:
        session_manager.close()


def test_telegram_command_result_session_id_overrides_global_current_session(
    monkeypatch,
    tmp_path: Path,
) -> None:
    project_root = _create_temp_project(tmp_path, allowed_chat_ids=[42])
    settings = load_settings(project_root=project_root)
    api_client = FakeApiClient()
    tracer = FakeTracer()
    bound_sessions: list[tuple[int, str]] = []

    channel = telegram_bot.TelegramBotChannel(
        settings=settings,
        config=telegram_bot.TelegramConfig(
            bot_token_env_var="TELEGRAM_BOT_TOKEN",
            polling_timeout_seconds=30,
            allowed_chat_ids=frozenset({42}),
        ),
        session_manager=SimpleNamespace(current_session_id="global-session"),
        memory_manager=SimpleNamespace(),
        tracer=tracer,
        tool_executor=SimpleNamespace(list_tools=lambda: []),
        api_client=api_client,
        session_store=SimpleNamespace(
            bind_chat=lambda *, chat_id, session_id: bound_sessions.append(
                (chat_id, session_id)
            )
        ),
    )

    monkeypatch.setattr(
        telegram_bot.TelegramBotChannel,
        "_activate_chat_session",
        lambda self, chat_id: SimpleNamespace(
            id=f"chat-session-{chat_id}",
            title=f"Telegram chat {chat_id}",
        ),
    )
    monkeypatch.setattr(
        telegram_bot.TelegramBotChannel,
        "_get_command_handler",
        lambda self, chat_id: SimpleNamespace(
            handle=lambda _text: CommandResult(
                status=CommandStatus.OK,
                lines=("Switched.",),
                session_id="new-chat-session",
            )
        ),
    )

    channel._handle_update(
        {
            "message": {
                "chat": {"id": 42},
                "date": 100,
                "text": "/use other-session",
            }
        }
    )

    assert bound_sessions == [(42, "new-chat-session")]
    assert api_client.sent_messages == [(42, "Switched.")]


def _build_channel(
    tmp_path: Path,
    *,
    allowed_chat_ids: tuple[int, ...],
    clock=lambda: 0.0,
) -> tuple[telegram_bot.TelegramBotChannel, FakeApiClient, FakeTracer]:
    project_root = _create_temp_project(
        tmp_path,
        allowed_chat_ids=list(allowed_chat_ids),
    )
    settings = load_settings(project_root=project_root)
    return _build_channel_for_settings(
        settings,
        config=telegram_bot.TelegramConfig(
            bot_token_env_var="TELEGRAM_BOT_TOKEN",
            polling_timeout_seconds=30,
            allowed_chat_ids=frozenset(allowed_chat_ids),
        ),
        clock=clock,
    )


def _build_channel_for_settings(
    settings,
    *,
    config,
    clock=lambda: 0.0,
):
    api_client = FakeApiClient()
    tracer = FakeTracer()
    channel = telegram_bot.TelegramBotChannel(
        settings=settings,
        config=config,
        session_manager=SimpleNamespace(
            current_session_id=None,
            add_message=lambda *args, **kwargs: None,
        ),
        memory_manager=SimpleNamespace(),
        tracer=tracer,
        tool_executor=SimpleNamespace(list_tools=lambda: []),
        api_client=api_client,
        session_store=SimpleNamespace(bind_chat=lambda **kwargs: None),
        clock=clock,
    )
    return channel, api_client, tracer


def _create_temp_project(
    tmp_path: Path,
    *,
    allowed_chat_ids: list[int] | None = None,
) -> Path:
    source_root = Path(__file__).resolve().parents[1]
    project_root = tmp_path / "project"
    shutil.copytree(source_root / "config", project_root / "config")
    if allowed_chat_ids is not None:
        telegram_config_path = project_root / "config" / "telegram.yaml"
        payload = yaml.safe_load(telegram_config_path.read_text(encoding="utf-8"))
        assert isinstance(payload, dict)
        payload["allowed_chat_ids"] = allowed_chat_ids
        telegram_config_path.write_text(
            yaml.safe_dump(payload, sort_keys=False),
            encoding="utf-8",
        )
    return project_root
