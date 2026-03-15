from __future__ import annotations

import shutil
from datetime import date as real_date
from pathlib import Path
from types import SimpleNamespace

import yaml

from unclaw.core.capabilities import (
    build_runtime_capability_context,
    build_runtime_capability_summary,
)
from unclaw.core.capability_router import CapabilityDecision, CapabilityKind
from unclaw.core.command_handler import CommandHandler
from unclaw.core.research_flow import build_tool_history_content, run_search_then_answer
from unclaw.core.runtime import run_user_turn
from unclaw.core.runtime_modes import RuntimeMode
from unclaw.core.session_manager import SessionManager
from unclaw.llm.base import LLMMessage, LLMResponse, LLMRole
from unclaw.logs.event_bus import EventBus
from unclaw.logs.tracer import TraceEvent, Tracer
from unclaw.schemas.chat import MessageRole
from unclaw.settings import load_settings
from unclaw.tools.contracts import ToolResult
from unclaw.tools.registry import ToolRegistry
from unclaw.tools.web_tools import FETCH_URL_TEXT_DEFINITION


class _StaticCapabilityRouter:
    def __init__(self, decision: CapabilityDecision) -> None:
        self.decision = decision

    def route(self, **kwargs):  # type: ignore[no-untyped-def]
        del kwargs
        return self.decision


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
            capability_router=_StaticCapabilityRouter(
                CapabilityDecision(
                    kind=CapabilityKind.DIRECT_ANSWER,
                    confidence="high",
                    source="test",
                )
            ),
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


def test_run_user_turn_routes_agent_mode_web_research_through_search_flow(
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
            del profile, messages, timeout_seconds, thinking_enabled, content_callback
            return LLMResponse(
                provider="ollama",
                model_name="qwen3.5:4b",
                content="Ollama shipped a new update with improved search grounding.",
                created_at="2026-03-13T12:00:00Z",
                finish_reason="stop",
            )

    monkeypatch.setattr("unclaw.core.orchestrator.OllamaProvider", FakeOllamaProvider)

    search_tool_calls: list[object] = []
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
            ],
        },
    )

    try:
        session = session_manager.ensure_current_session()
        session_manager.add_message(
            MessageRole.USER,
            "latest news about Ollama",
            session_id=session.id,
        )

        assistant_reply = run_user_turn(
            session_manager=session_manager,
            command_handler=command_handler,
            user_input="latest news about Ollama",
            tracer=tracer,
            tool_executor=SimpleNamespace(
                execute=lambda tool_call: (
                    search_tool_calls.append(tool_call) or search_tool_result
                ),
                registry=ToolRegistry(),
            ),
            capability_router=_StaticCapabilityRouter(
                CapabilityDecision(
                    kind=CapabilityKind.WEB_RESEARCH,
                    confidence="high",
                    source="test",
                )
            ),
        )

        assert assistant_reply == (
            "Ollama shipped a new update with improved search grounding.\n\n"
            "Sources:\n"
            "- Ollama Blog: https://ollama.com/blog/search-update"
        )
        assert len(search_tool_calls) == 1
        assert search_tool_calls[0].tool_name == "search_web"
        assert search_tool_calls[0].arguments == {"query": "latest news about Ollama"}

        stored_messages = session_manager.list_messages(session.id)
        assert [message.role for message in stored_messages] == [
            MessageRole.USER,
            MessageRole.TOOL,
            MessageRole.ASSISTANT,
        ]
        assert "Grounding rules:" in stored_messages[1].content
    finally:
        session_manager.close()


def test_run_user_turn_falls_back_to_chatbot_mode_with_warning_and_skips_autonomous_search(
    monkeypatch,
    tmp_path: Path,
) -> None:
    project_root = _create_temp_project(tmp_path)
    models_config_path = project_root / "config" / "models.yaml"
    models_payload = yaml.safe_load(models_config_path.read_text(encoding="utf-8"))
    assert isinstance(models_payload, dict)
    profiles_payload = models_payload["profiles"]
    assert isinstance(profiles_payload, dict)
    profiles_payload["chatonly"] = dict(profiles_payload["main"])
    profiles_payload["chatonly"]["tool_mode"] = "none"
    models_config_path.write_text(
        yaml.safe_dump(models_payload, sort_keys=False),
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
        current_model_profile_name="chatonly",
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
            del profile, messages, timeout_seconds, thinking_enabled, content_callback
            return LLMResponse(
                provider="ollama",
                model_name="qwen3.5:4b",
                content="I can still answer directly in chatbot mode.",
                created_at="2026-03-13T12:00:00Z",
                finish_reason="stop",
            )

    monkeypatch.setattr("unclaw.core.orchestrator.OllamaProvider", FakeOllamaProvider)

    try:
        session = session_manager.ensure_current_session()
        session_manager.add_message(
            MessageRole.USER,
            "latest news about Ollama",
            session_id=session.id,
        )

        assistant_reply = run_user_turn(
            session_manager=session_manager,
            command_handler=command_handler,
            user_input="latest news about Ollama",
            tracer=tracer,
            tool_executor=SimpleNamespace(
                execute=lambda tool_call: (_raise_unexpected_tool_call(tool_call)),
                registry=ToolRegistry(),
            ),
            capability_router=SimpleNamespace(
                route=lambda **kwargs: (_raise_unexpected_router_call(kwargs))
            ),
        )

        assert assistant_reply == (
            "Please note: the selected model profile does not support tools reliably. "
            "Unclaw will switch to Chatbot mode. "
            "Chatbot mode = simple conversation, no web research, no automation.\n\n"
            "I can still answer directly in chatbot mode."
        )
        stored_messages = session_manager.list_messages(session.id)
        assert stored_messages[-1].content == assistant_reply
    finally:
        session_manager.close()


def test_run_user_turn_returns_local_file_follow_up_without_web_execution(
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
    )

    try:
        session = session_manager.ensure_current_session()
        session_manager.add_message(
            MessageRole.USER,
            "Please inspect README.md in this repo.",
            session_id=session.id,
        )

        assistant_reply = run_user_turn(
            session_manager=session_manager,
            command_handler=command_handler,
            user_input="Please inspect README.md in this repo.",
            tracer=tracer,
            tool_executor=SimpleNamespace(
                execute=lambda tool_call: (_raise_unexpected_tool_call(tool_call)),
                registry=ToolRegistry(),
            ),
            capability_router=_StaticCapabilityRouter(
                CapabilityDecision(
                    kind=CapabilityKind.LOCAL_FILE_INTENT,
                    confidence="high",
                    source="test",
                    follow_up_message=(
                        "Which local file should I inspect? "
                        "You can also use /read <path>."
                    ),
                )
            ),
        )

        assert assistant_reply == (
            "Which local file should I inspect? You can also use /read <path>."
        )
        stored_messages = session_manager.list_messages(session.id)
        assert stored_messages[-1].role is MessageRole.ASSISTANT
        assert stored_messages[-1].content == assistant_reply
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
        runtime_mode=RuntimeMode.AGENT,
    )
    context = build_runtime_capability_context(summary)

    assert summary.enabled_builtin_tool_count == 1
    assert summary.runtime_mode is RuntimeMode.AGENT
    assert summary.url_fetch_available is True
    assert summary.web_search_available is False
    assert summary.local_file_read_available is False
    assert summary.local_directory_listing_available is False
    assert summary.memory_summary_available is False
    assert "Available built-in tools:" in context
    assert "Runtime mode: Agent mode" in context
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
            capability_router=_StaticCapabilityRouter(
                CapabilityDecision(
                    kind=CapabilityKind.DIRECT_ANSWER,
                    confidence="high",
                    source="test",
                )
            ),
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
        assert "Grounding rules:" in stored_messages[1].content
        assert "Supported facts:" in stored_messages[1].content
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
            capability_router=_StaticCapabilityRouter(
                CapabilityDecision(
                    kind=CapabilityKind.DIRECT_ANSWER,
                    confidence="high",
                    source="test",
                )
            ),
        )

        assert follow_up_reply == "Shorter recap."
        follow_up_messages = captured_messages[1]
        assert any(
            message.role is LLMRole.TOOL
            and "Ollama Blog: https://ollama.com/blog/search-update" in message.content
            for message in follow_up_messages
        )
        assert any(
            message.role is LLMRole.SYSTEM
            and "Search-backed answer contract:" in message.content
            for message in follow_up_messages
        )
    finally:
        session_manager.close()


def test_run_search_then_answer_removes_stale_relative_dates_from_search_backed_replies(
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

    _freeze_search_grounding_date(monkeypatch)

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
            del profile, messages, timeout_seconds, thinking_enabled, content_callback
            return LLMResponse(
                provider="ollama",
                model_name="qwen3.5:4b",
                content=(
                    "Alex Rivera was born on 1998-05-04 and, as of May 2024, "
                    "is 26 years old."
                ),
                created_at="2026-03-13T12:00:00Z",
                finish_reason="stop",
            )

    monkeypatch.setattr("unclaw.core.orchestrator.OllamaProvider", FakeOllamaProvider)

    search_tool_result = ToolResult.ok(
        tool_name="search_web",
        output_text="Search query: how old is Alex Rivera\n",
        payload={
            "query": "how old is Alex Rivera",
            "summary_points": ["Alex Rivera was born on 1998-05-04."],
            "display_sources": [
                {
                    "title": "Official Bio",
                    "url": "https://alex.example.com/bio",
                },
                {
                    "title": "Magazine Interview",
                    "url": "https://press.example.com/alex-rivera-interview",
                },
            ],
            "synthesized_findings": [
                {
                    "text": "Alex Rivera was born on 1998-05-04.",
                    "score": 7.8,
                    "support_count": 2,
                    "source_titles": ["Official Bio", "Magazine Interview"],
                    "source_urls": [
                        "https://alex.example.com/bio",
                        "https://press.example.com/alex-rivera-interview",
                    ],
                }
            ],
            "results": [
                {
                    "title": "Official Bio",
                    "url": "https://alex.example.com/bio",
                    "takeaway": "Official biography page.",
                    "depth": 1,
                    "fetched": True,
                    "evidence_count": 1,
                    "fetch_error": None,
                    "used_snippet_fallback": False,
                    "usefulness": 9.0,
                },
                {
                    "title": "Magazine Interview",
                    "url": "https://press.example.com/alex-rivera-interview",
                    "takeaway": "Interview confirming the birth date.",
                    "depth": 1,
                    "fetched": True,
                    "evidence_count": 1,
                    "fetch_error": None,
                    "used_snippet_fallback": False,
                    "usefulness": 7.5,
                },
            ],
            "evidence": [
                {
                    "text": "Alex Rivera was born on 1998-05-04.",
                    "url": "https://alex.example.com/bio",
                    "source_title": "Official Bio",
                    "score": 7.8,
                    "depth": 1,
                    "query_relevance": 4.0,
                    "evidence_quality": 4.0,
                    "novelty": 1.0,
                    "supporting_urls": [
                        "https://alex.example.com/bio",
                        "https://press.example.com/alex-rivera-interview",
                    ],
                    "supporting_titles": [
                        "Official Bio",
                        "Magazine Interview",
                    ],
                }
            ],
        },
    )

    try:
        reply = run_search_then_answer(
            session_manager=session_manager,
            command_handler=command_handler,
            tracer=tracer,
            tool_executor=SimpleNamespace(
                execute=lambda _tool_call: search_tool_result,
                registry=ToolRegistry(),
            ),
            tool_call=SimpleNamespace(
                tool_name="search_web",
                arguments={"query": "how old is Alex Rivera"},
            ),
        ).assistant_reply

        assert "as of May 2024" not in reply
        assert "I found a birth date of 1998-05-04." in reply
        assert "On 2026-03-14, that makes them 27 years old." in reply
        assert reply.endswith(
            "- Magazine Interview: https://press.example.com/alex-rivera-interview"
        )
    finally:
        session_manager.close()


def test_run_search_then_answer_does_not_confirm_weak_usernames(
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
            del profile, messages, timeout_seconds, thinking_enabled, content_callback
            return LLMResponse(
                provider="ollama",
                model_name="qwen3.5:4b",
                content=(
                    "Jordan Lee is a product designer and engineer. "
                    "Their Instagram is probably @jordancode."
                ),
                created_at="2026-03-13T12:00:00Z",
                finish_reason="stop",
            )

    monkeypatch.setattr("unclaw.core.orchestrator.OllamaProvider", FakeOllamaProvider)

    search_tool_result = ToolResult.ok(
        tool_name="search_web",
        output_text="Search query: who is Jordan Lee\n",
        payload={
            "query": "who is Jordan Lee",
            "summary_points": [
                "Jordan Lee is a product designer and engineer.",
                "One profile lists the handle @jordancode.",
            ],
            "display_sources": [
                {
                    "title": "Company Bio",
                    "url": "https://company.example.com/jordan-lee",
                },
                {
                    "title": "Guest Q&A",
                    "url": "https://community.example.com/jordan-qa",
                },
            ],
            "synthesized_findings": [
                {
                    "text": "Jordan Lee is a product designer and engineer.",
                    "score": 8.1,
                    "support_count": 2,
                    "source_titles": ["Company Bio", "Guest Q&A"],
                    "source_urls": [
                        "https://company.example.com/jordan-lee",
                        "https://community.example.com/jordan-qa",
                    ],
                },
                {
                    "text": "One profile lists the handle @jordancode.",
                    "score": 4.2,
                    "support_count": 1,
                    "source_titles": ["Guest Q&A"],
                    "source_urls": ["https://community.example.com/jordan-qa"],
                },
            ],
            "results": [
                {
                    "title": "Company Bio",
                    "url": "https://company.example.com/jordan-lee",
                    "takeaway": "Official bio page.",
                    "depth": 1,
                    "fetched": True,
                    "evidence_count": 1,
                    "fetch_error": None,
                    "used_snippet_fallback": False,
                    "usefulness": 9.0,
                },
                {
                    "title": "Guest Q&A",
                    "url": "https://community.example.com/jordan-qa",
                    "takeaway": "Community interview with one social handle mention.",
                    "depth": 1,
                    "fetched": True,
                    "evidence_count": 1,
                    "fetch_error": None,
                    "used_snippet_fallback": False,
                    "usefulness": 5.0,
                },
            ],
        },
    )

    try:
        reply = run_search_then_answer(
            session_manager=session_manager,
            command_handler=command_handler,
            tracer=tracer,
            tool_executor=SimpleNamespace(
                execute=lambda _tool_call: search_tool_result,
                registry=ToolRegistry(),
            ),
            tool_call=SimpleNamespace(
                tool_name="search_web",
                arguments={"query": "who is Jordan Lee"},
            ),
        ).assistant_reply

        assert "Jordan Lee is a product designer and engineer." in reply
        assert "@jordancode" not in reply
        assert "not consistently confirmed" in reply
    finally:
        session_manager.close()


def test_run_search_then_answer_person_summary_prefers_supported_identity_over_fluff(
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
            del profile, messages, timeout_seconds, thinking_enabled, content_callback
            return LLMResponse(
                provider="ollama",
                model_name="qwen3.5:4b",
                content=(
                    "Taylor Stone seems to be an inspiring creator who often shows up "
                    "on podcasts and blogs."
                ),
                created_at="2026-03-13T12:00:00Z",
                finish_reason="stop",
            )

    monkeypatch.setattr("unclaw.core.orchestrator.OllamaProvider", FakeOllamaProvider)

    search_tool_result = ToolResult.ok(
        tool_name="search_web",
        output_text="Search query: tell me everything you know about Taylor Stone\n",
        payload={
            "query": "tell me everything you know about Taylor Stone",
            "summary_points": [
                "Taylor Stone is a robotics researcher and startup founder.",
                "She created the River Hand open-source prosthetics project.",
                "She has appeared on a few podcasts about creativity.",
            ],
            "display_sources": [
                {
                    "title": "Lab Bio",
                    "url": "https://lab.example.com/taylor-stone",
                },
                {
                    "title": "Project Page",
                    "url": "https://riverhand.example.com/about",
                },
            ],
            "synthesized_findings": [
                {
                    "text": "Taylor Stone is a robotics researcher and startup founder.",
                    "score": 8.5,
                    "support_count": 2,
                    "source_titles": ["Lab Bio", "Project Page"],
                    "source_urls": [
                        "https://lab.example.com/taylor-stone",
                        "https://riverhand.example.com/about",
                    ],
                },
                {
                    "text": "She created the River Hand open-source prosthetics project.",
                    "score": 7.4,
                    "support_count": 2,
                    "source_titles": ["Project Page", "Lab Bio"],
                    "source_urls": [
                        "https://riverhand.example.com/about",
                        "https://lab.example.com/taylor-stone",
                    ],
                },
                {
                    "text": "She has appeared on a few podcasts about creativity.",
                    "score": 4.0,
                    "support_count": 1,
                    "source_titles": ["Guest Podcast"],
                    "source_urls": ["https://podcasts.example.com/taylor-stone"],
                },
            ],
            "results": [
                {
                    "title": "Lab Bio",
                    "url": "https://lab.example.com/taylor-stone",
                    "takeaway": "Institutional biography page.",
                    "depth": 1,
                    "fetched": True,
                    "evidence_count": 1,
                    "fetch_error": None,
                    "used_snippet_fallback": False,
                    "usefulness": 8.8,
                },
                {
                    "title": "Project Page",
                    "url": "https://riverhand.example.com/about",
                    "takeaway": "Official project description.",
                    "depth": 1,
                    "fetched": True,
                    "evidence_count": 1,
                    "fetch_error": None,
                    "used_snippet_fallback": False,
                    "usefulness": 8.1,
                },
                {
                    "title": "Guest Podcast",
                    "url": "https://podcasts.example.com/taylor-stone",
                    "takeaway": "Podcast appearance.",
                    "depth": 1,
                    "fetched": True,
                    "evidence_count": 1,
                    "fetch_error": None,
                    "used_snippet_fallback": False,
                    "usefulness": 3.8,
                },
            ],
        },
    )

    try:
        reply = run_search_then_answer(
            session_manager=session_manager,
            command_handler=command_handler,
            tracer=tracer,
            tool_executor=SimpleNamespace(
                execute=lambda _tool_call: search_tool_result,
                registry=ToolRegistry(),
            ),
            tool_call=SimpleNamespace(
                tool_name="search_web",
                arguments={
                    "query": "tell me everything you know about Taylor Stone"
                },
            ),
        ).assistant_reply

        assert "Taylor Stone is a robotics researcher and startup founder." in reply
        assert "She created the River Hand open-source prosthetics project." in reply
        assert "inspiring" not in reply
        assert "podcast" not in reply.lower().split("Sources:")[0]
    finally:
        session_manager.close()


def test_run_search_then_answer_omits_unconfirmed_achievements_and_keeps_compact_sources(
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
            del profile, messages, timeout_seconds, thinking_enabled, content_callback
            return LLMResponse(
                provider="ollama",
                model_name="qwen3.5:4b",
                content=(
                    "Pat Kim leads a major AI lab and won a national innovation prize."
                ),
                created_at="2026-03-13T12:00:00Z",
                finish_reason="stop",
            )

    monkeypatch.setattr("unclaw.core.orchestrator.OllamaProvider", FakeOllamaProvider)

    search_tool_result = ToolResult.ok(
        tool_name="search_web",
        output_text="Search query: what has Pat Kim done\n",
        payload={
            "query": "what has Pat Kim done",
            "summary_points": [
                "Pat Kim leads the Applied Systems Lab.",
                "One blog says Pat Kim won a national innovation prize.",
            ],
            "display_sources": [
                {
                    "title": "Applied Systems Lab",
                    "url": "https://lab.example.com/pat-kim",
                },
                {
                    "title": "Conference Program",
                    "url": "https://conference.example.com/speakers/pat-kim",
                },
            ],
            "synthesized_findings": [
                {
                    "text": "Pat Kim leads the Applied Systems Lab.",
                    "score": 7.3,
                    "support_count": 2,
                    "source_titles": ["Applied Systems Lab", "Conference Program"],
                    "source_urls": [
                        "https://lab.example.com/pat-kim",
                        "https://conference.example.com/speakers/pat-kim",
                    ],
                },
                {
                    "text": "One blog says Pat Kim won a national innovation prize.",
                    "score": 3.7,
                    "support_count": 1,
                    "source_titles": ["Personal Blog"],
                    "source_urls": ["https://blog.example.com/pat-kim-profile"],
                },
            ],
            "results": [
                {
                    "title": "Applied Systems Lab",
                    "url": "https://lab.example.com/pat-kim",
                    "takeaway": "Official lab profile.",
                    "depth": 1,
                    "fetched": True,
                    "evidence_count": 1,
                    "fetch_error": None,
                    "used_snippet_fallback": False,
                    "usefulness": 9.0,
                },
                {
                    "title": "Conference Program",
                    "url": "https://conference.example.com/speakers/pat-kim",
                    "takeaway": "Conference speaker listing.",
                    "depth": 1,
                    "fetched": True,
                    "evidence_count": 1,
                    "fetch_error": None,
                    "used_snippet_fallback": False,
                    "usefulness": 7.2,
                },
                {
                    "title": "Personal Blog",
                    "url": "https://blog.example.com/pat-kim-profile",
                    "takeaway": "One blog post with an award claim.",
                    "depth": 1,
                    "fetched": True,
                    "evidence_count": 1,
                    "fetch_error": None,
                    "used_snippet_fallback": False,
                    "usefulness": 3.2,
                },
            ],
        },
    )

    try:
        reply = run_search_then_answer(
            session_manager=session_manager,
            command_handler=command_handler,
            tracer=tracer,
            tool_executor=SimpleNamespace(
                execute=lambda _tool_call: search_tool_result,
                registry=ToolRegistry(),
            ),
            tool_call=SimpleNamespace(
                tool_name="search_web",
                arguments={"query": "what has Pat Kim done"},
            ),
        ).assistant_reply

        answer_body, sources_block = reply.split("\n\nSources:\n", maxsplit=1)
        assert "Pat Kim leads the Applied Systems Lab." in answer_body
        assert "national innovation prize" not in answer_body
        assert "Sources:\n" not in sources_block
        assert all(line.startswith("- ") and ": https://" in line for line in sources_block.splitlines())
        assert all("takeaway" not in line.casefold() for line in sources_block.splitlines())
    finally:
        session_manager.close()


def test_run_user_turn_keeps_follow_up_turns_grounded_after_search(
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
    captured_messages: list[LLMMessage] = []

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
            captured_messages.extend(messages)
            assert any(
                message.role is LLMRole.SYSTEM
                and "Search-backed answer contract:" in message.content
                for message in messages
            )
            return LLMResponse(
                provider="ollama",
                model_name="qwen3.5:4b",
                content=(
                    "Jordan Lee is a product designer and engineer. "
                    "I couldn't confirm a social handle across the retrieved sources."
                ),
                created_at="2026-03-13T12:00:00Z",
                finish_reason="stop",
            )

    monkeypatch.setattr("unclaw.core.orchestrator.OllamaProvider", FakeOllamaProvider)
    _freeze_search_grounding_date(monkeypatch)

    tool_result = ToolResult.ok(
        tool_name="search_web",
        output_text="Search query: who is Jordan Lee\n",
        payload={
            "query": "who is Jordan Lee",
            "summary_points": [
                "Jordan Lee is a product designer and engineer.",
                "One profile lists the handle @jordancode.",
            ],
            "display_sources": [
                {
                    "title": "Company Bio",
                    "url": "https://company.example.com/jordan-lee",
                },
            ],
            "synthesized_findings": [
                {
                    "text": "Jordan Lee is a product designer and engineer.",
                    "score": 8.1,
                    "support_count": 2,
                    "source_titles": ["Company Bio"],
                    "source_urls": ["https://company.example.com/jordan-lee"],
                },
                {
                    "text": "One profile lists the handle @jordancode.",
                    "score": 4.2,
                    "support_count": 1,
                    "source_titles": ["Community Q&A"],
                    "source_urls": ["https://community.example.com/jordan-qa"],
                },
            ],
        },
    )

    try:
        session = session_manager.ensure_current_session()
        session_manager.add_message(
            MessageRole.TOOL,
            build_tool_history_content(
                tool_result,
                tool_call=SimpleNamespace(
                    tool_name="search_web",
                    arguments={"query": "who is Jordan Lee"},
                ),
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
            capability_router=_StaticCapabilityRouter(
                CapabilityDecision(
                    kind=CapabilityKind.DIRECT_ANSWER,
                    confidence="high",
                    source="test",
                )
            ),
        )

        assert assistant_reply == (
            "Jordan Lee is a product designer and engineer. "
            "I couldn't confirm a social handle across the retrieved sources."
        )
        tool_messages = [
            message.content
            for message in captured_messages
            if message.role is LLMRole.TOOL
        ]
        assert len(tool_messages) == 1
        assert "Grounding rules:" in tool_messages[0]
        assert "Supported facts:" in tool_messages[0]
        assert "Uncertain details:" in tool_messages[0]
        assert "Sources fetched:" not in tool_messages[0]
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
            capability_router=_StaticCapabilityRouter(
                CapabilityDecision(
                    kind=CapabilityKind.DIRECT_ANSWER,
                    confidence="high",
                    source="test",
                )
            ),
        )

        assert assistant_reply == "Timed reply"
        assert captured["default_timeout_seconds"] == 123.0
        assert captured["base_url"] == "http://127.0.0.1:11434"
    finally:
        session_manager.close()


def test_follow_up_after_person_search_keeps_context(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Pronoun follow-up after a person search-backed answer stays as direct_answer."""
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
    command_handler = CommandHandler(
        settings=settings,
        session_manager=session_manager,
        memory_manager=SimpleNamespace(),
    )

    class FakeOllamaProvider:
        provider_name = "ollama"

        def __init__(self, **kwargs) -> None:
            del kwargs

        def chat(self, profile, messages, **kwargs):
            del kwargs
            return LLMResponse(
                provider="ollama",
                model_name=profile.model_name,
                content="Here is a detailed description of her career.",
                created_at="2026-03-14T12:00:00Z",
                finish_reason="stop",
            )

    monkeypatch.setattr("unclaw.core.orchestrator.OllamaProvider", FakeOllamaProvider)

    try:
        session = session_manager.ensure_current_session()
        # Simulate prior search context in history
        session_manager.add_message(
            MessageRole.USER,
            "Who is Marie Curie?",
            session_id=session.id,
        )
        session_manager.add_message(
            MessageRole.TOOL,
            "Tool: search_web\nOutcome: success\n\n"
            "Supported facts:\n"
            "- [strong; 3 sources] Marie Curie was a physicist and chemist.\n"
            "- [supported; 2 sources] She won two Nobel Prizes.\n",
            session_id=session.id,
        )
        session_manager.add_message(
            MessageRole.ASSISTANT,
            "Marie Curie was a physicist and chemist who won two Nobel Prizes.",
            session_id=session.id,
        )
        # Now send the follow-up
        session_manager.add_message(
            MessageRole.USER,
            "Fais une description plus détaillée d'elle",
            session_id=session.id,
        )

        # The follow-up should be routed as direct_answer, NOT ambiguous
        assistant_reply = run_user_turn(
            session_manager=session_manager,
            command_handler=command_handler,
            user_input="Fais une description plus détaillée d'elle",
            tracer=tracer,
        )

        # Verify it went through direct_answer (model was called)
        route_events = [
            event for event in published_events
            if event.event_type == "route.selected"
        ]
        assert len(route_events) == 1
        assert route_events[0].payload["route_kind"] == "direct_answer"
        assert route_events[0].payload["route_source"] == "follow_up_resolution"
        assert assistant_reply  # Got a model reply, not a fallback message
        assert "clarify" not in assistant_reply.lower()
    finally:
        session_manager.close()


def test_agent_mode_never_prints_internal_plan_text(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Agent mode strips planning narration from model replies."""
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

    class FakeOllamaProvider:
        provider_name = "ollama"

        def __init__(self, **kwargs) -> None:
            del kwargs

        def chat(self, profile, messages, **kwargs):
            del kwargs
            return LLMResponse(
                provider="ollama",
                model_name=profile.model_name,
                content=(
                    "Je vais vérifier les dernières informations.\n"
                    "/search Lara Croft next movie\n"
                    "The next Lara Croft project has not been officially announced yet."
                ),
                created_at="2026-03-14T12:00:00Z",
                finish_reason="stop",
            )

    monkeypatch.setattr("unclaw.core.orchestrator.OllamaProvider", FakeOllamaProvider)

    try:
        session = session_manager.ensure_current_session()
        session_manager.add_message(
            MessageRole.USER,
            "Who will play Lara Croft?",
            session_id=session.id,
        )

        assistant_reply = run_user_turn(
            session_manager=session_manager,
            command_handler=command_handler,
            user_input="Who will play Lara Croft?",
            tracer=tracer,
            capability_router=_StaticCapabilityRouter(
                CapabilityDecision(
                    kind=CapabilityKind.DIRECT_ANSWER,
                    confidence="high",
                    source="test",
                )
            ),
        )

        # Planning narration and slash commands must be stripped
        assert "Je vais vérifier" not in assistant_reply
        assert "/search" not in assistant_reply
        # The actual answer content should remain
        assert "Lara Croft" in assistant_reply
    finally:
        session_manager.close()


def test_agent_mode_strips_raw_json_structured_output(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Agent mode replaces raw JSON structured output with neutral message."""
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

    class FakeOllamaProvider:
        provider_name = "ollama"

        def __init__(self, **kwargs) -> None:
            del kwargs

        def chat(self, profile, messages, **kwargs):
            del kwargs
            return LLMResponse(
                provider="ollama",
                model_name=profile.model_name,
                content='{"capability":"web_research","confidence":"high","follow_up":null}',
                created_at="2026-03-14T12:00:00Z",
                finish_reason="stop",
            )

    monkeypatch.setattr("unclaw.core.orchestrator.OllamaProvider", FakeOllamaProvider)

    try:
        session = session_manager.ensure_current_session()
        session_manager.add_message(
            MessageRole.USER,
            "What is the weather?",
            session_id=session.id,
        )

        assistant_reply = run_user_turn(
            session_manager=session_manager,
            command_handler=command_handler,
            user_input="What is the weather?",
            tracer=tracer,
            capability_router=_StaticCapabilityRouter(
                CapabilityDecision(
                    kind=CapabilityKind.DIRECT_ANSWER,
                    confidence="high",
                    source="test",
                )
            ),
        )

        # Raw JSON should not appear in the reply
        assert '"capability"' not in assistant_reply
        assert "web_research" not in assistant_reply
        # Should get a clean fallback message
        assert "rephrasing" in assistant_reply.lower() or "clear answer" in assistant_reply.lower()
    finally:
        session_manager.close()


def test_chatbot_mode_does_not_auto_trigger_web_research(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Chatbot mode always routes to direct_answer, never web_research."""
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
    # Use "fast" profile which has thinking_supported=false and tool_mode=json_plan
    # but actually we need a profile that maps to chatbot mode.
    # Since all profiles with tool_mode != "none" support agent mode,
    # we need to test this differently - use a static router that would return
    # web_research, but the chatbot mode override should prevent it.
    from unclaw.core.router import route_request, RouteKind
    from unclaw.core.capabilities import build_runtime_capability_summary
    from unclaw.core.runtime_modes import RuntimeMode

    # Manually resolve: fast profile has json_plan tool_mode -> agent mode.
    # To test chatbot mode, we need a profile with tool_mode=none.
    # Let's just test the route_request function directly with a fake profile.
    from unclaw.llm.base import ResolvedModelProfile, ModelCapabilities

    chatbot_profile = ResolvedModelProfile(
        name="chatbot_test",
        provider="ollama",
        model_name="test:1b",
        temperature=0.5,
        capabilities=ModelCapabilities(
            thinking_supported=False,
            tool_mode="none",
            supports_tools=False,
            supports_native_tool_calling=False,
            supports_agent_mode=False,
        ),
    )

    monkeypatch.setattr(
        "unclaw.core.router.resolve_model_profile",
        lambda settings, name: chatbot_profile,
    )

    summary = build_runtime_capability_summary(
        tool_registry=ToolRegistry(),
        memory_summary_available=False,
        runtime_mode=RuntimeMode.CHATBOT,
    )

    route = route_request(
        settings=settings,
        model_profile_name="chatbot_test",
        user_message="What are the latest news today?",
        capability_summary=summary,
    )

    assert route.kind is RouteKind.DIRECT_ANSWER
    assert route.runtime_mode is RuntimeMode.CHATBOT
    assert route.route_source == "chatbot_fallback"
    session_manager.close()


def test_logs_distinguish_router_profile_vs_active_profile(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Route selection logs must include router_model_profile_name."""
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
    command_handler = CommandHandler(
        settings=settings,
        session_manager=session_manager,
        memory_manager=SimpleNamespace(),
    )

    class FakeOllamaProvider:
        provider_name = "ollama"

        def __init__(self, **kwargs) -> None:
            del kwargs

        def chat(self, profile, messages, **kwargs):
            del kwargs
            return LLMResponse(
                provider="ollama",
                model_name=profile.model_name,
                content="Model reply.",
                created_at="2026-03-14T12:00:00Z",
                finish_reason="stop",
            )

    monkeypatch.setattr("unclaw.core.orchestrator.OllamaProvider", FakeOllamaProvider)

    try:
        session = session_manager.ensure_current_session()
        session_manager.add_message(
            MessageRole.USER,
            "Hello there",
            session_id=session.id,
        )

        run_user_turn(
            session_manager=session_manager,
            command_handler=command_handler,
            user_input="Hello there",
            tracer=tracer,
            capability_router=_StaticCapabilityRouter(
                CapabilityDecision(
                    kind=CapabilityKind.DIRECT_ANSWER,
                    confidence="high",
                    source="test",
                )
            ),
        )

        route_events = [
            event for event in published_events
            if event.event_type == "route.selected"
        ]
        assert len(route_events) == 1
        payload = route_events[0].payload
        assert "model_profile_name" in payload
        assert "router_model_profile_name" in payload
        assert "runtime_mode" in payload
        assert "route_kind" in payload
        assert "route_confidence" in payload
        assert "route_source" in payload
    finally:
        session_manager.close()


def test_weak_evidence_produces_cautious_wording(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Weak/conflicting evidence must produce cautious wording, not overclaiming."""
    _freeze_search_grounding_date(monkeypatch)
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

    class FakeOllamaProvider:
        provider_name = "ollama"

        def __init__(self, **kwargs) -> None:
            del kwargs

        def chat(self, profile, messages, **kwargs):
            del kwargs
            # The model's reply overclaims something that's only uncertain
            return LLMResponse(
                provider="ollama",
                model_name=profile.model_name,
                content="John Doe is 35 years old and lives in Paris.",
                created_at="2026-03-14T12:00:00Z",
                finish_reason="stop",
            )

    monkeypatch.setattr("unclaw.core.orchestrator.OllamaProvider", FakeOllamaProvider)

    try:
        session = session_manager.ensure_current_session()

        # Build a search result where findings are uncertain
        tool_result = ToolResult.ok(
            tool_name="search_web",
            output_text="Search results for John Doe",
            payload={
                "query": "Who is John Doe?",
                "summary_points": [
                    "John Doe may be a software engineer.",
                ],
                "synthesized_findings": [
                    {
                        "text": "John Doe may be a software engineer.",
                        "score": 3.0,
                        "support_count": 1,
                        "source_titles": ["Some Blog"],
                        "source_urls": ["https://example.com/blog"],
                    },
                ],
                "display_sources": [
                    {"title": "Some Blog", "url": "https://example.com/blog"},
                ],
                "results": [
                    {
                        "title": "Some Blog",
                        "url": "https://example.com/blog",
                        "fetched": False,
                        "evidence_count": 0,
                        "usefulness": 2.0,
                        "used_snippet_fallback": True,
                    },
                ],
            },
        )

        result = run_search_then_answer(
            session_manager=session_manager,
            command_handler=command_handler,
            tracer=tracer,
            tool_executor=SimpleNamespace(
                execute=lambda call: tool_result,
                registry=ToolRegistry(),
            ),
            tool_call=SimpleNamespace(
                tool_name="search_web",
                arguments={"query": "Who is John Doe?"},
            ),
        )

        reply = result.assistant_reply
        # The reply should include cautious language since evidence is weak
        has_caution = any(
            marker in reply.lower()
            for marker in (
                "not confirmed",
                "not consistently confirmed",
                "uncertain",
                "could not confirm",
                "few mentions",
                "weakly supported",
                "unconfirmed",
            )
        )
        # Either the grounding rewriter added cautious language, or the model's
        # reply was left clean (grounding passed it through). At minimum, it
        # should NOT contain the age claim without a birth date.
        assert "35 years old" not in reply or has_caution
    finally:
        session_manager.close()


def test_freshness_question_routes_to_web_research_via_llm_router(
    monkeypatch,
) -> None:
    """Freshness/current-event questions with the LLM router should prefer web_research."""
    from unclaw.core.capability_router import (
        _build_router_messages,
    )
    from unclaw.core.capabilities import build_runtime_capability_summary
    from unclaw.core.runtime_modes import RuntimeMode

    settings = load_settings(project_root=Path(__file__).resolve().parents[1])
    summary = build_runtime_capability_summary(
        tool_registry=ToolRegistry(),
        memory_summary_available=False,
        runtime_mode=RuntimeMode.AGENT,
    )

    # Verify the router prompt includes freshness signals
    messages = _build_router_messages(
        user_message="Who is the current CEO of Apple?",
        capability_summary=summary,
    )
    system_content = messages[0].content
    assert "Freshness signals" in system_content
    assert "current" in system_content.lower()
    assert "latest" in system_content.lower()
    assert "upcoming" in system_content.lower()


def test_slash_commands_still_work_after_routing_changes(
    tmp_path: Path,
) -> None:
    """Slash commands must continue to work unchanged."""
    project_root = _create_temp_project(tmp_path)
    settings = load_settings(project_root=project_root)
    session_manager = SessionManager.from_settings(settings)

    try:
        command_handler = CommandHandler(
            settings=settings,
            session_manager=session_manager,
            memory_manager=SimpleNamespace(),
        )

        # /help should work
        result = command_handler.handle("/help")
        assert result.status.value == "ok"
        assert any("slash commands" in line.lower() for line in result.lines)

        # /model should work
        result = command_handler.handle("/model")
        assert result.status.value == "ok"

        # /search should produce a tool call
        result = command_handler.handle("/search test query")
        assert result.tool_call is not None
        assert result.tool_call.tool_name == "search_web"
        assert result.tool_call.arguments["query"] == "test query"

        # /think should work
        result = command_handler.handle("/think")
        assert result.status.value == "ok"
    finally:
        session_manager.close()


def test_pending_clarification_resumes_web_research(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """After assistant asks clarification, one-word 'internet' → web research."""
    from unclaw.core.executor import create_default_tool_registry

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
    command_handler = CommandHandler(
        settings=settings,
        session_manager=session_manager,
        memory_manager=SimpleNamespace(),
    )

    class FakeOllamaProvider:
        provider_name = "ollama"

        def __init__(self, **kwargs) -> None:
            del kwargs

        def chat(self, profile, messages, **kwargs):
            del kwargs
            return LLMResponse(
                provider="ollama",
                model_name=profile.model_name,
                content="Marine Leleu est une sportive française.",
                created_at="2026-03-14T12:00:00Z",
                finish_reason="stop",
            )

    monkeypatch.setattr("unclaw.core.orchestrator.OllamaProvider", FakeOllamaProvider)

    search_tool_result = ToolResult.ok(
        tool_name="search_web",
        output_text="Search query: internet\n",
        payload={
            "query": "internet",
            "summary_points": ["Marine Leleu est une sportive française."],
            "display_sources": [
                {"title": "Bio", "url": "https://example.com/bio"},
            ],
        },
    )

    default_registry = create_default_tool_registry(settings)

    try:
        session = session_manager.ensure_current_session()
        # Setup: user asked a question, assistant asked clarification
        session_manager.add_message(
            MessageRole.USER,
            "fais des recherches sur Marine Leleu",
            session_id=session.id,
        )
        session_manager.add_message(
            MessageRole.ASSISTANT,
            "Souhaitez-vous que je cherche sur internet ou dans vos fichiers locaux ?",
            session_id=session.id,
        )
        # User answers with a single word
        session_manager.add_message(
            MessageRole.USER,
            "internet",
            session_id=session.id,
        )

        assistant_reply = run_user_turn(
            session_manager=session_manager,
            command_handler=command_handler,
            user_input="internet",
            tracer=tracer,
            tool_executor=SimpleNamespace(
                execute=lambda _: search_tool_result,
                registry=default_registry,
            ),
        )

        # The reply should contain the search result, NOT another question
        route_events = [
            e for e in published_events if e.event_type == "route.selected"
        ]
        assert len(route_events) == 1
        assert route_events[0].payload["route_kind"] == "web_research"
        assert route_events[0].payload["route_source"] == "pending_clarification"
        assert "sportive" in assistant_reply.lower() or "marine" in assistant_reply.lower()
    finally:
        session_manager.close()


def test_subject_continuation_wikipedia_triggers_web_research_with_subject(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """After person search, 'cherche sur Wikipédia' → web research (subject continuation)."""
    from unclaw.core.executor import create_default_tool_registry

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
    command_handler = CommandHandler(
        settings=settings,
        session_manager=session_manager,
        memory_manager=SimpleNamespace(),
    )

    class FakeOllamaProvider:
        provider_name = "ollama"

        def __init__(self, **kwargs) -> None:
            del kwargs

        def chat(self, profile, messages, **kwargs):
            del kwargs
            return LLMResponse(
                provider="ollama",
                model_name=profile.model_name,
                content="Voici plus de détails depuis Wikipédia.",
                created_at="2026-03-14T12:00:00Z",
                finish_reason="stop",
            )

    monkeypatch.setattr("unclaw.core.orchestrator.OllamaProvider", FakeOllamaProvider)

    search_tool_result = ToolResult.ok(
        tool_name="search_web",
        output_text="Search query: cherche sur Wikipédia\n",
        payload={
            "query": "cherche sur Wikipédia",
            "summary_points": ["Details from Wikipedia."],
            "display_sources": [
                {"title": "Wikipedia", "url": "https://fr.wikipedia.org/wiki/Marine_Leleu"},
            ],
        },
    )

    default_registry = create_default_tool_registry(settings)

    try:
        session = session_manager.ensure_current_session()
        # Prior context: person search with results
        session_manager.add_message(
            MessageRole.USER,
            "fais des recherches sur Marine Leleu",
            session_id=session.id,
        )
        session_manager.add_message(
            MessageRole.TOOL,
            (
                "Tool: search_web\nOutcome: success\n\n"
                "Supported facts:\n"
                "- [strong; 2 sources] Marine Leleu est une sportive française.\n"
            ),
            session_id=session.id,
        )
        session_manager.add_message(
            MessageRole.ASSISTANT,
            "Marine Leleu est une sportive française.",
            session_id=session.id,
        )
        session_manager.add_message(
            MessageRole.USER,
            "cherche sur Wikipédia",
            session_id=session.id,
        )

        assistant_reply = run_user_turn(
            session_manager=session_manager,
            command_handler=command_handler,
            user_input="cherche sur Wikipédia",
            tracer=tracer,
            tool_executor=SimpleNamespace(
                execute=lambda _: search_tool_result,
                registry=default_registry,
            ),
        )

        route_events = [
            e for e in published_events if e.event_type == "route.selected"
        ]
        assert len(route_events) == 1
        assert route_events[0].payload["route_kind"] == "web_research"
        assert route_events[0].payload["route_source"] == "subject_continuation"
    finally:
        session_manager.close()


def test_freshness_news_request_auto_triggers_web_research(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """'Résumé des actualités du jour' in Agent mode → reliable web research."""
    from unclaw.core.executor import create_default_tool_registry

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
    command_handler = CommandHandler(
        settings=settings,
        session_manager=session_manager,
        memory_manager=SimpleNamespace(),
    )

    class FakeOllamaProvider:
        provider_name = "ollama"

        def __init__(self, **kwargs) -> None:
            del kwargs

        def chat(self, profile, messages, **kwargs):
            del kwargs
            return LLMResponse(
                provider="ollama",
                model_name=profile.model_name,
                content="Voici les actualités du jour.",
                created_at="2026-03-14T12:00:00Z",
                finish_reason="stop",
            )

    monkeypatch.setattr("unclaw.core.orchestrator.OllamaProvider", FakeOllamaProvider)

    search_tool_result = ToolResult.ok(
        tool_name="search_web",
        output_text="Search query: actualités du jour\n",
        payload={
            "query": "actualités du jour",
            "summary_points": ["Today's key events."],
            "display_sources": [
                {"title": "News", "url": "https://example.com/news"},
            ],
        },
    )

    default_registry = create_default_tool_registry(settings)

    try:
        session = session_manager.ensure_current_session()
        session_manager.add_message(
            MessageRole.USER,
            "fais moi un résumé des actualités importantes du jour",
            session_id=session.id,
        )

        assistant_reply = run_user_turn(
            session_manager=session_manager,
            command_handler=command_handler,
            user_input="fais moi un résumé des actualités importantes du jour",
            tracer=tracer,
            tool_executor=SimpleNamespace(
                execute=lambda _: search_tool_result,
                registry=default_registry,
            ),
        )

        route_events = [
            e for e in published_events if e.event_type == "route.selected"
        ]
        assert len(route_events) == 1
        assert route_events[0].payload["route_kind"] == "web_research"
        assert route_events[0].payload["route_source"] == "freshness_heuristic"
    finally:
        session_manager.close()


def test_retrieval_debris_stripped_from_agent_reply(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Retrieval debris like 'Some profile details...' is cleaned from replies."""
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

    class FakeOllamaProvider:
        provider_name = "ollama"

        def __init__(self, **kwargs) -> None:
            del kwargs

        def chat(self, profile, messages, **kwargs):
            del kwargs
            return LLMResponse(
                provider="ollama",
                model_name=profile.model_name,
                content=(
                    "Marine Leleu est une sportive française.\n"
                    "Some profile details were not confirmed.\n"
                    "Sources fetched: 3 of 5 attempted\n"
                    "Evidence kept: 2\n"
                    "She is known for her athletic achievements."
                ),
                created_at="2026-03-14T12:00:00Z",
                finish_reason="stop",
            )

    monkeypatch.setattr("unclaw.core.orchestrator.OllamaProvider", FakeOllamaProvider)

    try:
        session = session_manager.ensure_current_session()
        session_manager.add_message(
            MessageRole.USER,
            "Tell me about Marine Leleu.",
            session_id=session.id,
        )

        assistant_reply = run_user_turn(
            session_manager=session_manager,
            command_handler=command_handler,
            user_input="Tell me about Marine Leleu.",
            tracer=tracer,
            capability_router=_StaticCapabilityRouter(
                CapabilityDecision(
                    kind=CapabilityKind.DIRECT_ANSWER,
                    confidence="high",
                    source="test",
                )
            ),
        )

        assert "Some profile details" not in assistant_reply
        assert "Sources fetched:" not in assistant_reply
        assert "Evidence kept:" not in assistant_reply
        assert "Marine Leleu" in assistant_reply
        assert "athletic" in assistant_reply.lower()
    finally:
        session_manager.close()


def test_chatbot_mode_preserves_slash_commands_and_no_auto_routing(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Chatbot mode: no auto routing, slash commands still work."""
    project_root = _create_temp_project(tmp_path)
    settings = load_settings(project_root=project_root)
    session_manager = SessionManager.from_settings(settings)

    try:
        command_handler = CommandHandler(
            settings=settings,
            session_manager=session_manager,
            memory_manager=SimpleNamespace(),
        )

        # Slash commands should still work
        result = command_handler.handle("/search test query about news")
        assert result.tool_call is not None
        assert result.tool_call.tool_name == "search_web"
        assert "news" in result.tool_call.arguments["query"]

        result = command_handler.handle("/help")
        assert result.status.value == "ok"

        # Direct route_request in chatbot mode should always return direct_answer
        from unclaw.core.router import route_request, RouteKind
        from unclaw.core.capabilities import build_runtime_capability_summary
        from unclaw.core.runtime_modes import RuntimeMode
        from unclaw.llm.base import ResolvedModelProfile, ModelCapabilities

        chatbot_profile = ResolvedModelProfile(
            name="chatbot_test",
            provider="ollama",
            model_name="test:1b",
            temperature=0.5,
            capabilities=ModelCapabilities(
                thinking_supported=False,
                tool_mode="none",
                supports_tools=False,
                supports_native_tool_calling=False,
                supports_agent_mode=False,
            ),
        )
        monkeypatch.setattr(
            "unclaw.core.router.resolve_model_profile",
            lambda s, n: chatbot_profile,
        )

        summary = build_runtime_capability_summary(
            tool_registry=ToolRegistry(),
            memory_summary_available=False,
            runtime_mode=RuntimeMode.CHATBOT,
        )

        # Freshness query should NOT auto-trigger web in chatbot mode
        route = route_request(
            settings=settings,
            model_profile_name="chatbot_test",
            user_message="fais moi un résumé des actualités du jour",
            capability_summary=summary,
        )
        assert route.kind is RouteKind.DIRECT_ANSWER
        assert route.runtime_mode is RuntimeMode.CHATBOT
    finally:
        session_manager.close()


def test_french_uncertainty_note_in_french_query(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Uncertainty notes should be in French when query is French."""
    _freeze_search_grounding_date(monkeypatch)
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

    class FakeOllamaProvider:
        provider_name = "ollama"

        def __init__(self, **kwargs) -> None:
            del kwargs

        def chat(self, profile, messages, **kwargs):
            del kwargs
            return LLMResponse(
                provider="ollama",
                model_name=profile.model_name,
                content="Marine Leleu seems to be an inspiring person who often shares on podcasts.",
                created_at="2026-03-14T12:00:00Z",
                finish_reason="stop",
            )

    monkeypatch.setattr("unclaw.core.orchestrator.OllamaProvider", FakeOllamaProvider)

    search_tool_result = ToolResult.ok(
        tool_name="search_web",
        output_text="Search query: qui est Marine Leleu\n",
        payload={
            "query": "qui est Marine Leleu",
            "synthesized_findings": [
                {
                    "text": "Marine Leleu est une sportive française.",
                    "score": 8.0,
                    "support_count": 2,
                    "source_titles": ["Bio", "Sport"],
                    "source_urls": [
                        "https://example.com/bio",
                        "https://example.com/sport",
                    ],
                },
                {
                    "text": "She appears on various podcasts.",
                    "score": 3.5,
                    "support_count": 1,
                    "source_titles": ["Podcast"],
                    "source_urls": ["https://example.com/podcast"],
                },
            ],
            "display_sources": [
                {"title": "Bio", "url": "https://example.com/bio"},
                {"title": "Sport", "url": "https://example.com/sport"},
            ],
            "results": [
                {
                    "title": "Bio",
                    "url": "https://example.com/bio",
                    "fetched": True,
                    "evidence_count": 1,
                    "usefulness": 8.0,
                    "used_snippet_fallback": False,
                },
                {
                    "title": "Sport",
                    "url": "https://example.com/sport",
                    "fetched": True,
                    "evidence_count": 1,
                    "usefulness": 7.0,
                    "used_snippet_fallback": False,
                },
            ],
        },
    )

    try:
        reply = run_search_then_answer(
            session_manager=session_manager,
            command_handler=command_handler,
            tracer=tracer,
            tool_executor=SimpleNamespace(
                execute=lambda _: search_tool_result,
                registry=ToolRegistry(),
            ),
            tool_call=SimpleNamespace(
                tool_name="search_web",
                arguments={"query": "qui est Marine Leleu"},
            ),
        ).assistant_reply

        # The answer should contain supported facts
        assert "Marine Leleu" in reply
        assert "sportive" in reply.lower()
        # Uncertainty note should be in French since query was French
        if "détails" in reply or "confirmé" in reply or "omis" in reply:
            # French uncertainty note detected — good
            assert "Some profile details" not in reply
            assert "Some lower-confidence" not in reply
        # No English boilerplate should appear
        assert "inspiring" not in reply
        assert "podcast" not in reply.lower().split("Sources:")[0]
    finally:
        session_manager.close()


def _create_temp_project(tmp_path: Path) -> Path:
    source_root = Path(__file__).resolve().parents[1]
    project_root = tmp_path / "project"
    shutil.copytree(source_root / "config", project_root / "config")
    return project_root


def _freeze_search_grounding_date(monkeypatch) -> None:
    class FixedDate(real_date):
        @classmethod
        def today(cls) -> FixedDate:
            return cls(2026, 3, 14)

    monkeypatch.setattr("unclaw.core.context_builder.date", FixedDate)
    monkeypatch.setattr("unclaw.core.research_flow.date", FixedDate)
    monkeypatch.setattr("unclaw.core.search_grounding.date", FixedDate)


def _raise_unexpected_tool_call(tool_call):
    raise AssertionError(f"Unexpected tool call: {tool_call}")


def _raise_unexpected_router_call(payload):
    raise AssertionError(f"Unexpected capability router call: {payload}")
