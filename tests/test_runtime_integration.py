from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

import yaml

from unclaw.core.capabilities import (
    build_runtime_capability_context,
    build_runtime_capability_summary,
)
from unclaw.core.command_handler import CommandHandler
from unclaw.core.research_flow import run_search_then_answer
from unclaw.core.runtime import run_user_turn
from unclaw.core.session_manager import SessionManager
from unclaw.llm.base import LLMMessage, LLMResponse, LLMRole
from unclaw.logs.event_bus import EventBus
from unclaw.logs.tracer import TraceEvent, Tracer
from unclaw.schemas.chat import MessageRole
from unclaw.settings import load_settings
from unclaw.tools.contracts import ToolResult
from unclaw.tools.registry import ToolRegistry
from unclaw.tools.web_tools import FETCH_URL_TEXT_DEFINITION


def test_run_user_turn_persists_reply_and_emits_runtime_events(
    monkeypatch,
    tmp_path: Path,
) -> None:
    project_root = _create_temp_project(tmp_path)
    settings = load_settings(project_root=project_root)
    session_manager = SessionManager.from_settings(settings)
    event_bus = EventBus()
    published_events: list[TraceEvent] = []
    event_bus.subscribe(published_events.append)
    tracer = Tracer(
        event_bus=event_bus,
        event_repository=session_manager.event_repository,
    )
    tracer.runtime_log_path = settings.paths.log_file_path
    command_handler = CommandHandler(
        settings=settings,
        session_manager=session_manager,
        memory_manager=SimpleNamespace(),
    )
    captured: dict[str, object] = {}

    class FakeOllamaProvider:
        provider_name = "ollama"

        def __init__(
            self,
            *,
            base_url: str = "http://127.0.0.1:11434",
            default_timeout_seconds: float = 60.0,
        ) -> None:
            del base_url, default_timeout_seconds

        def chat(  # type: ignore[no-untyped-def]
            self,
            profile,
            messages,
            *,
            timeout_seconds=None,
            thinking_enabled=False,
            content_callback=None,
        ):
            del timeout_seconds
            captured["profile_name"] = profile.name
            captured["messages"] = list(messages)
            captured["thinking_enabled"] = thinking_enabled
            if content_callback is not None:
                content_callback("Local reply")
            return LLMResponse(
                provider="ollama",
                model_name=profile.model_name,
                content="Local reply",
                created_at="2026-03-13T12:00:00Z",
                finish_reason="stop",
                reasoning="short reasoning",
            )

        def is_available(self, *, timeout_seconds=None) -> bool:  # type: ignore[no-untyped-def]
            del timeout_seconds
            return True

    monkeypatch.setattr("unclaw.core.orchestrator.OllamaProvider", FakeOllamaProvider)

    try:
        session = session_manager.ensure_current_session()
        session_manager.add_message(
            MessageRole.USER,
            "Summarize this test run.",
            session_id=session.id,
        )
        streamed_chunks: list[str] = []

        assistant_reply = run_user_turn(
            session_manager=session_manager,
            command_handler=command_handler,
            user_input="Summarize this test run.",
            tracer=tracer,
            stream_output_func=streamed_chunks.append,
        )

        assert assistant_reply == "Local reply"
        assert streamed_chunks == ["Local reply"]

        messages = session_manager.list_messages(session.id)
        assert messages[-2].role is MessageRole.USER
        assert messages[-2].content == "Summarize this test run."
        assert messages[-1].role is MessageRole.ASSISTANT
        assert messages[-1].content == "Local reply"

        provider_messages = captured["messages"]
        assert isinstance(provider_messages, list)
        assert all(isinstance(message, LLMMessage) for message in provider_messages)
        assert provider_messages[0].content == settings.system_prompt
        assert provider_messages[1].role is LLMRole.SYSTEM
        assert "Enabled built-in tools: 4" in provider_messages[1].content
        assert "/read <path>" in provider_messages[1].content
        assert "/fetch <url>" in provider_messages[1].content
        assert (
            "/search <query>: search the public web, read a few relevant pages, "
            "and answer naturally from grounded web context with compact sources."
            in provider_messages[1].content
        )
        assert "Session memory and summary access." in provider_messages[1].content
        assert "no tools available" not in provider_messages[1].content.lower()
        assert provider_messages[-1].content == "Summarize this test run."
        assert captured["profile_name"] == settings.app.default_model_profile
        assert captured["thinking_enabled"] is False

        event_types = [event.event_type for event in published_events]
        assert event_types == [
            "runtime.started",
            "route.selected",
            "model.called",
            "model.succeeded",
            "assistant.reply.persisted",
        ]

        persisted_events = session_manager.event_repository.list_recent_events(
            session.id,
            limit=10,
        )
        persisted_event_types = [event.event_type for event in persisted_events]
        assert "assistant.reply.persisted" in persisted_event_types
        assert "model.succeeded" in persisted_event_types

        runtime_log = settings.paths.log_file_path.read_text(encoding="utf-8")
        assert '"event_type": "assistant.reply.persisted"' in runtime_log
        assert '"event_type": "model.succeeded"' in runtime_log
    finally:
        session_manager.close()


def test_runtime_capability_summary_reports_available_and_missing_capabilities() -> None:
    registry = ToolRegistry()
    registry.register(
        FETCH_URL_TEXT_DEFINITION,
        lambda call: ToolResult.ok(tool_name=call.tool_name, output_text="ok"),
    )

    summary = build_runtime_capability_summary(
        tool_registry=registry,
        memory_summary_available=False,
    )
    context = build_runtime_capability_context(summary)

    assert summary.enabled_builtin_tool_count == 1
    assert summary.url_fetch_available is True
    assert summary.web_search_available is False
    assert summary.local_file_read_available is False
    assert summary.local_directory_listing_available is False
    assert summary.memory_summary_available is False
    assert "Available built-in tools:" in context
    assert "/fetch <url>: fetch one public URL and extract text." in context
    assert "Web search via /search <query>." in context
    assert "Session memory and summary access." in context
    assert "Do not claim you have no tool access" in context
    assert "Do not say you cannot access it" in context


def test_run_user_turn_includes_prior_tool_output_for_follow_up_questions(
    monkeypatch,
    tmp_path: Path,
) -> None:
    project_root = _create_temp_project(tmp_path)
    settings = load_settings(project_root=project_root)
    session_manager = SessionManager.from_settings(settings)
    tracer = Tracer(
        event_bus=EventBus(),
        event_repository=session_manager.event_repository,
    )
    command_handler = CommandHandler(
        settings=settings,
        session_manager=session_manager,
        memory_manager=SimpleNamespace(),
    )
    captured: dict[str, object] = {}

    class FakeOllamaProvider:
        provider_name = "ollama"

        def __init__(
            self,
            *,
            base_url: str = "http://127.0.0.1:11434",
            default_timeout_seconds: float = 60.0,
        ) -> None:
            del base_url, default_timeout_seconds

        def chat(  # type: ignore[no-untyped-def]
            self,
            profile,
            messages,
            *,
            timeout_seconds=None,
            thinking_enabled=False,
            content_callback=None,
        ):
            del profile, timeout_seconds, thinking_enabled, content_callback
            captured["messages"] = list(messages)
            return LLMResponse(
                provider="ollama",
                model_name="qwen3.5:4b",
                content="Shorter recap.",
                created_at="2026-03-13T12:00:00Z",
                finish_reason="stop",
            )

    monkeypatch.setattr("unclaw.core.orchestrator.OllamaProvider", FakeOllamaProvider)

    try:
        session = session_manager.ensure_current_session()
        session_manager.add_message(
            MessageRole.TOOL,
            (
                "Tool: search_web\n"
                "Outcome: success\n\n"
                "Search query: latest news about Ollama\n"
                "Summary:\n"
                "- I searched 3 public results and read 2 top sources directly.\n"
                "- Source A: Ollama shipped a new update.\n"
            ),
            session_id=session.id,
        )
        session_manager.add_message(
            MessageRole.USER,
            "Summarize that more briefly.",
            session_id=session.id,
        )

        assistant_reply = run_user_turn(
            session_manager=session_manager,
            command_handler=command_handler,
            user_input="Summarize that more briefly.",
            tracer=tracer,
        )

        assert assistant_reply == "Shorter recap."
        provider_messages = captured["messages"]
        assert isinstance(provider_messages, list)
        assert any(
            message.role is LLMRole.TOOL and "Tool: search_web" in message.content
            for message in provider_messages
        )
        assert provider_messages[-1].role is LLMRole.USER
        assert provider_messages[-1].content == "Summarize that more briefly."
        assert "Do not say you cannot access it" in provider_messages[1].content
    finally:
        session_manager.close()


def test_run_search_then_answer_grounds_a_natural_reply_and_preserves_follow_up_context(
    monkeypatch,
    tmp_path: Path,
) -> None:
    project_root = _create_temp_project(tmp_path)
    settings = load_settings(project_root=project_root)
    session_manager = SessionManager.from_settings(settings)
    tracer = Tracer(
        event_bus=EventBus(),
        event_repository=session_manager.event_repository,
    )
    command_handler = CommandHandler(
        settings=settings,
        session_manager=session_manager,
        memory_manager=SimpleNamespace(),
    )
    captured_messages: list[list[LLMMessage]] = []
    reply_texts = iter(
        [
            "Ollama shipped a new update with improved search grounding.",
            "Shorter recap.",
        ]
    )

    class FakeOllamaProvider:
        provider_name = "ollama"

        def __init__(
            self,
            *,
            base_url: str = "http://127.0.0.1:11434",
            default_timeout_seconds: float = 60.0,
        ) -> None:
            del base_url, default_timeout_seconds

        def chat(  # type: ignore[no-untyped-def]
            self,
            profile,
            messages,
            *,
            timeout_seconds=None,
            thinking_enabled=False,
            content_callback=None,
        ):
            del profile, timeout_seconds, thinking_enabled, content_callback
            captured_messages.append(list(messages))
            return LLMResponse(
                provider="ollama",
                model_name="qwen3.5:4b",
                content=next(reply_texts),
                created_at="2026-03-13T12:00:00Z",
                finish_reason="stop",
            )

    monkeypatch.setattr("unclaw.core.orchestrator.OllamaProvider", FakeOllamaProvider)

    search_tool_result = ToolResult.ok(
        tool_name="search_web",
        output_text=(
            "Search query: latest news about Ollama\n"
            "Sources fetched: 2 of 2 attempted\n"
            "Evidence kept: 4\n"
        ),
        payload={
            "query": "latest news about Ollama",
            "summary_points": [
                "Ollama shipped a new update with improved search grounding."
            ],
            "display_sources": [
                {
                    "title": "Ollama Blog",
                    "url": "https://ollama.com/blog/search-update",
                },
                {
                    "title": "Release Notes",
                    "url": "https://example.com/releases/ollama-search",
                },
            ],
        },
    )

    try:
        session = session_manager.ensure_current_session()

        search_reply = run_search_then_answer(
            session_manager=session_manager,
            command_handler=command_handler,
            tracer=tracer,
            tool_executor=SimpleNamespace(
                execute=lambda _tool_call: search_tool_result,
                registry=ToolRegistry(),
            ),
            tool_call=SimpleNamespace(
                tool_name="search_web",
                arguments={"query": "latest news about Ollama"},
            ),
        ).assistant_reply

        assert search_reply == (
            "Ollama shipped a new update with improved search grounding.\n\n"
            "Sources:\n"
            "- Ollama Blog: https://ollama.com/blog/search-update\n"
            "- Release Notes: https://example.com/releases/ollama-search"
        )
        assert "Search query:" not in search_reply
        assert "Sources fetched:" not in search_reply
        assert "Evidence kept:" not in search_reply

        stored_messages = session_manager.list_messages(session.id)
        assert [message.role for message in stored_messages] == [
            MessageRole.USER,
            MessageRole.TOOL,
            MessageRole.ASSISTANT,
        ]
        assert stored_messages[0].content == "latest news about Ollama"
        assert "Search query:" not in stored_messages[1].content
        assert "Evidence kept:" not in stored_messages[1].content
        assert "Findings:" in stored_messages[1].content
        assert "Sources:" in stored_messages[1].content

        search_turn_messages = captured_messages[0]
        tool_messages = [
            message.content
            for message in search_turn_messages
            if message.role is LLMRole.TOOL
        ]
        assert tool_messages == [stored_messages[1].content]
        assert search_turn_messages[-1].role is LLMRole.TOOL
        assert sum(
            1
            for message in search_turn_messages
            if message.role is LLMRole.USER
            and message.content == "latest news about Ollama"
        ) == 1

        session_manager.add_message(
            MessageRole.USER,
            "Summarize that more briefly.",
            session_id=session.id,
        )

        follow_up_reply = run_user_turn(
            session_manager=session_manager,
            command_handler=command_handler,
            user_input="Summarize that more briefly.",
            tracer=tracer,
        )

        assert follow_up_reply == "Shorter recap."
        follow_up_messages = captured_messages[1]
        assert any(
            message.role is LLMRole.TOOL
            and "Ollama Blog: https://ollama.com/blog/search-update" in message.content
            for message in follow_up_messages
        )
    finally:
        session_manager.close()


def test_run_user_turn_uses_configured_ollama_timeout(
    monkeypatch,
    tmp_path: Path,
) -> None:
    project_root = _create_temp_project(tmp_path)
    app_config_path = project_root / "config" / "app.yaml"
    app_payload = yaml.safe_load(app_config_path.read_text(encoding="utf-8"))
    assert isinstance(app_payload, dict)
    providers_payload = app_payload.setdefault("providers", {})
    assert isinstance(providers_payload, dict)
    providers_payload["ollama"] = {"timeout_seconds": 123.0}
    app_config_path.write_text(
        yaml.safe_dump(app_payload, sort_keys=False),
        encoding="utf-8",
    )

    settings = load_settings(project_root=project_root)
    session_manager = SessionManager.from_settings(settings)
    tracer = Tracer(
        event_bus=EventBus(),
        event_repository=session_manager.event_repository,
    )
    command_handler = CommandHandler(
        settings=settings,
        session_manager=session_manager,
    )
    captured: dict[str, object] = {}

    class FakeOllamaProvider:
        provider_name = "ollama"

        def __init__(
            self,
            *,
            base_url: str = "http://127.0.0.1:11434",
            default_timeout_seconds: float = 60.0,
        ) -> None:
            captured["base_url"] = base_url
            captured["default_timeout_seconds"] = default_timeout_seconds

        def chat(  # type: ignore[no-untyped-def]
            self,
            profile,
            messages,
            *,
            timeout_seconds=None,
            thinking_enabled=False,
            content_callback=None,
        ):
            del profile, messages, timeout_seconds, thinking_enabled, content_callback
            return LLMResponse(
                provider="ollama",
                model_name="qwen3.5:4b",
                content="Timed reply",
                created_at="2026-03-13T12:00:00Z",
                finish_reason="stop",
            )

    monkeypatch.setattr("unclaw.core.orchestrator.OllamaProvider", FakeOllamaProvider)

    try:
        session = session_manager.ensure_current_session()
        session_manager.add_message(
            MessageRole.USER,
            "Check timeout wiring.",
            session_id=session.id,
        )

        assistant_reply = run_user_turn(
            session_manager=session_manager,
            command_handler=command_handler,
            user_input="Check timeout wiring.",
            tracer=tracer,
        )

        assert assistant_reply == "Timed reply"
        assert captured["default_timeout_seconds"] == 123.0
        assert captured["base_url"] == "http://127.0.0.1:11434"
    finally:
        session_manager.close()


def _create_temp_project(tmp_path: Path) -> Path:
    source_root = Path(__file__).resolve().parents[1]
    project_root = tmp_path / "project"
    shutil.copytree(source_root / "config", project_root / "config")
    return project_root
