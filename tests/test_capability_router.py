from __future__ import annotations

from pathlib import Path

from unclaw.core.capabilities import build_runtime_capability_summary
from unclaw.core.capability_router import CapabilityKind, LLMCapabilityRouter
from unclaw.core.executor import create_default_tool_registry
from unclaw.core.runtime_modes import RuntimeMode
from unclaw.llm.base import LLMResponse
from unclaw.llm.model_profiles import resolve_model_profile
from unclaw.settings import load_settings


def test_llm_capability_router_selects_web_research_from_structured_json(
    monkeypatch,
) -> None:
    settings = _load_repo_settings()

    class FakeOllamaProvider:
        provider_name = "ollama"

        def __init__(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
            del kwargs

        def chat(self, **kwargs):  # type: ignore[no-untyped-def]
            del kwargs
            return LLMResponse(
                provider="ollama",
                model_name="qwen3.5:4b",
                content=(
                    '{"capability":"web_research","confidence":"high",'
                    '"follow_up":null}'
                ),
                created_at="2026-03-13T12:00:00Z",
                finish_reason="stop",
            )

    monkeypatch.setattr(
        "unclaw.core.capability_router.OllamaProvider",
        FakeOllamaProvider,
    )

    summary = build_runtime_capability_summary(
        tool_registry=create_default_tool_registry(settings),
        memory_summary_available=False,
        runtime_mode=RuntimeMode.AGENT,
    )
    decision = LLMCapabilityRouter().route(
        settings=settings,
        profile=resolve_model_profile(settings, "main"),
        user_message="What's the latest Ollama release?",
        capability_summary=summary,
    )

    assert decision.kind is CapabilityKind.WEB_RESEARCH
    assert decision.confidence == "high"
    assert decision.source == "llm_router"


def test_llm_capability_router_protects_local_file_intents_from_web_misrouting(
    monkeypatch,
) -> None:
    settings = _load_repo_settings()

    class FakeOllamaProvider:
        provider_name = "ollama"

        def __init__(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
            del kwargs

        def chat(self, **kwargs):  # type: ignore[no-untyped-def]
            del kwargs
            return LLMResponse(
                provider="ollama",
                model_name="qwen3.5:4b",
                content=(
                    '{"capability":"web_research","confidence":"high",'
                    '"follow_up":null}'
                ),
                created_at="2026-03-13T12:00:00Z",
                finish_reason="stop",
            )

    monkeypatch.setattr(
        "unclaw.core.capability_router.OllamaProvider",
        FakeOllamaProvider,
    )

    summary = build_runtime_capability_summary(
        tool_registry=create_default_tool_registry(settings),
        memory_summary_available=False,
        runtime_mode=RuntimeMode.AGENT,
    )
    decision = LLMCapabilityRouter().route(
        settings=settings,
        profile=resolve_model_profile(settings, "main"),
        user_message="Please inspect README.md in this repo.",
        capability_summary=summary,
    )

    assert decision.kind is CapabilityKind.LOCAL_FILE_INTENT
    assert decision.source == "local_artifact_guardrail"
    assert "local file" in decision.follow_up_message.lower()


def test_llm_capability_router_falls_back_to_ambiguous_when_router_output_is_invalid(
    monkeypatch,
) -> None:
    settings = _load_repo_settings()

    class FakeOllamaProvider:
        provider_name = "ollama"

        def __init__(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
            del kwargs

        def chat(self, **kwargs):  # type: ignore[no-untyped-def]
            del kwargs
            return LLMResponse(
                provider="ollama",
                model_name="qwen3.5:4b",
                content="not json",
                created_at="2026-03-13T12:00:00Z",
                finish_reason="stop",
            )

    monkeypatch.setattr(
        "unclaw.core.capability_router.OllamaProvider",
        FakeOllamaProvider,
    )

    summary = build_runtime_capability_summary(
        tool_registry=create_default_tool_registry(settings),
        memory_summary_available=False,
        runtime_mode=RuntimeMode.AGENT,
    )
    decision = LLMCapabilityRouter().route(
        settings=settings,
        profile=resolve_model_profile(settings, "main"),
        user_message="Can you help with this?",
        capability_summary=summary,
    )

    assert decision.kind is CapabilityKind.AMBIGUOUS
    assert decision.source == "router_fallback"
    assert "direct answer" in decision.follow_up_message


def test_follow_up_pronoun_with_recent_context_routes_direct_answer() -> None:
    """Short follow-up with pronouns after assistant reply -> direct_answer."""
    from unclaw.core.capability_router import (
        _is_obvious_follow_up,
    )
    from unclaw.schemas.chat import ChatMessage, MessageRole

    _ts = "2026-03-14T12:00:00Z"
    history = [
        ChatMessage(
            id="m1", session_id="s1", role=MessageRole.USER,
            content="Who is Marie Curie?", created_at=_ts,
        ),
        ChatMessage(
            id="m2", session_id="s1", role=MessageRole.TOOL,
            content=(
                "Tool: search_web\nOutcome: success\n\n"
                "Supported facts:\n"
                "- [strong; 3 sources] Marie Curie was a physicist.\n"
            ),
            created_at=_ts,
        ),
        ChatMessage(
            id="m3", session_id="s1", role=MessageRole.ASSISTANT,
            content="Marie Curie was a physicist and chemist who won two Nobel Prizes.",
            created_at=_ts,
        ),
    ]

    # These should all be detected as obvious follow-ups
    assert _is_obvious_follow_up("Fais une description plus détaillée d'elle", history)
    assert _is_obvious_follow_up("et son âge ?", history)
    assert _is_obvious_follow_up("où est-elle née ?", history)
    assert _is_obvious_follow_up("et sa carrière récente ?", history)
    assert _is_obvious_follow_up("tell me more about her", history)
    assert _is_obvious_follow_up("describe him in more detail", history)
    assert _is_obvious_follow_up("and what about his career?", history)

    # Long messages should NOT be detected as follow-ups
    assert not _is_obvious_follow_up(
        "I want to know about the complete history of nuclear physics "
        "and how it changed the world in the 20th century and beyond",
        history,
    )

    # Without history, nothing is a follow-up
    assert not _is_obvious_follow_up("tell me more about her", [])


def test_follow_up_without_substantive_context_is_not_resolved() -> None:
    """Short follow-up without prior assistant reply -> not resolved."""
    from unclaw.core.capability_router import _is_obvious_follow_up
    from unclaw.schemas.chat import ChatMessage, MessageRole

    # Only user messages, no assistant reply
    history = [
        ChatMessage(
            id="m1", session_id="s1", role=MessageRole.USER,
            content="Hello", created_at="2026-03-14T12:00:00Z",
        ),
    ]
    assert not _is_obvious_follow_up("tell me more about her", history)


def test_router_prompt_includes_freshness_signals() -> None:
    """The router prompt must include freshness signal guidance."""
    from unclaw.core.capability_router import _build_router_messages
    from unclaw.core.capabilities import build_runtime_capability_summary
    from unclaw.core.runtime_modes import RuntimeMode
    from unclaw.tools.registry import ToolRegistry

    summary = build_runtime_capability_summary(
        tool_registry=ToolRegistry(),
        memory_summary_available=False,
        runtime_mode=RuntimeMode.AGENT,
    )
    messages = _build_router_messages(
        user_message="Who is the current CEO of Apple?",
        capability_summary=summary,
    )
    system_content = messages[0].content
    assert "Freshness signals" in system_content
    assert "current" in system_content
    assert "latest" in system_content
    assert "upcoming" in system_content
    assert "web_research" in system_content


def _load_repo_settings():
    return load_settings(project_root=Path(__file__).resolve().parents[1])

