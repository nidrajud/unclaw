from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

import yaml

from unclaw.channels import telegram_bot
from unclaw.settings import load_settings

EXAMPLE_TELEGRAM_TOKEN = "123456789:AAExampleTelegramBotTokenValue"


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
        def __init__(self, *, event_bus, event_repository) -> None:  # type: ignore[no-untyped-def]
            self.event_bus = event_bus
            self.event_repository = event_repository

        def trace_model_profile_selected(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
            return None

    class FakeToolExecutor:
        @classmethod
        def with_default_tools(cls):  # type: ignore[no-untyped-def]
            return cls()

    class FakeTelegramApiClient:
        def __init__(self, *, bot_token: str, request_timeout_seconds: float = 40.0) -> None:
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


def _create_temp_project(tmp_path: Path) -> Path:
    source_root = Path(__file__).resolve().parents[1]
    project_root = tmp_path / "project"
    shutil.copytree(source_root / "config", project_root / "config")
    return project_root
