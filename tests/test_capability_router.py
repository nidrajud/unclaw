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


def test_pending_clarification_internet_answer_routes_to_web_research() -> None:
    """One-word 'internet' after a clarification question → web_research."""
    from unclaw.core.capability_router import _is_pending_clarification_answer
    from unclaw.schemas.chat import ChatMessage, MessageRole

    _ts = "2026-03-14T12:00:00Z"
    history = [
        ChatMessage(
            id="m1", session_id="s1", role=MessageRole.USER,
            content="fais des recherches sur Marine Leleu",
            created_at=_ts,
        ),
        ChatMessage(
            id="m2", session_id="s1", role=MessageRole.ASSISTANT,
            content=(
                "Souhaitez-vous que je cherche sur internet ou "
                "dans vos fichiers locaux ?"
            ),
            created_at=_ts,
        ),
    ]

    decision = _is_pending_clarification_answer("internet", history)
    assert decision is not None
    assert decision.kind is CapabilityKind.WEB_RESEARCH
    assert decision.source == "pending_clarification"

    decision = _is_pending_clarification_answer("web", history)
    assert decision is not None
    assert decision.kind is CapabilityKind.WEB_RESEARCH

    decision = _is_pending_clarification_answer("oui", history)
    assert decision is not None
    assert decision.kind is CapabilityKind.WEB_RESEARCH


def test_pending_clarification_local_answer_routes_to_local_file() -> None:
    """One-word 'local' after a clarification question → local_file_intent."""
    from unclaw.core.capability_router import _is_pending_clarification_answer
    from unclaw.schemas.chat import ChatMessage, MessageRole

    _ts = "2026-03-14T12:00:00Z"
    history = [
        ChatMessage(
            id="m1", session_id="s1", role=MessageRole.ASSISTANT,
            content="Would you like me to search the internet or look at local files?",
            created_at=_ts,
        ),
    ]

    decision = _is_pending_clarification_answer("local", history)
    assert decision is not None
    assert decision.kind is CapabilityKind.LOCAL_FILE_INTENT
    assert decision.source == "pending_clarification"


def test_pending_clarification_wikipedia_routes_to_web_research() -> None:
    """'Wikipedia' after a clarification → web_research."""
    from unclaw.core.capability_router import _is_pending_clarification_answer
    from unclaw.schemas.chat import ChatMessage, MessageRole

    _ts = "2026-03-14T12:00:00Z"
    history = [
        ChatMessage(
            id="m1", session_id="s1", role=MessageRole.ASSISTANT,
            content="Voulez-vous chercher sur internet ou dans vos fichiers ?",
            created_at=_ts,
        ),
    ]

    decision = _is_pending_clarification_answer("wikipedia", history)
    assert decision is not None
    assert decision.kind is CapabilityKind.WEB_RESEARCH


def test_pending_clarification_not_triggered_without_question() -> None:
    """Short message after a non-question assistant reply is NOT clarification."""
    from unclaw.core.capability_router import _is_pending_clarification_answer
    from unclaw.schemas.chat import ChatMessage, MessageRole

    _ts = "2026-03-14T12:00:00Z"
    history = [
        ChatMessage(
            id="m1", session_id="s1", role=MessageRole.ASSISTANT,
            content="Marine Leleu est une sportive française.",
            created_at=_ts,
        ),
    ]

    decision = _is_pending_clarification_answer("internet", history)
    assert decision is None


def test_pending_clarification_not_triggered_for_long_messages() -> None:
    """Long messages after a clarification question are NOT resolved."""
    from unclaw.core.capability_router import _is_pending_clarification_answer
    from unclaw.schemas.chat import ChatMessage, MessageRole

    _ts = "2026-03-14T12:00:00Z"
    history = [
        ChatMessage(
            id="m1", session_id="s1", role=MessageRole.ASSISTANT,
            content="Souhaitez-vous que je cherche sur internet ?",
            created_at=_ts,
        ),
    ]

    decision = _is_pending_clarification_answer(
        "oui cherche sur internet et donne moi tous les détails possibles",
        history,
    )
    assert decision is None


def test_subject_continuation_wikipedia_follow_up() -> None:
    """'cherche sur Wikipédia' after a search → subject continuation."""
    from unclaw.core.capability_router import _is_subject_continuation
    from unclaw.schemas.chat import ChatMessage, MessageRole

    _ts = "2026-03-14T12:00:00Z"
    history = [
        ChatMessage(
            id="m1", session_id="s1", role=MessageRole.USER,
            content="fais des recherches sur Marine Leleu",
            created_at=_ts,
        ),
        ChatMessage(
            id="m2", session_id="s1", role=MessageRole.TOOL,
            content=(
                "Tool: search_web\nOutcome: success\n\n"
                "Supported facts:\n"
                "- [strong; 2 sources] Marine Leleu est une sportive française.\n"
            ),
            created_at=_ts,
        ),
        ChatMessage(
            id="m3", session_id="s1", role=MessageRole.ASSISTANT,
            content="Marine Leleu est une sportive française connue.",
            created_at=_ts,
        ),
    ]

    assert _is_subject_continuation("cherche sur Wikipédia", history)
    assert _is_subject_continuation("search on Wikipedia", history)
    assert _is_subject_continuation("regarde sur Wikipedia", history)
    assert not _is_subject_continuation("cherche sur Wikipédia", [])
    # Long message should not match
    assert not _is_subject_continuation(
        "je voudrais que tu fasses une recherche très approfondie sur Wikipédia "
        "avec tous les détails possibles et les références croisées",
        history,
    )


def test_freshness_heuristic_detects_news_requests() -> None:
    """Obvious freshness requests should be detected pre-LLM."""
    from unclaw.core.capability_router import _has_obvious_freshness_signal

    # French freshness requests
    assert _has_obvious_freshness_signal(
        "fais moi un résumé des actualités importantes du jour"
    )
    assert _has_obvious_freshness_signal("quoi de neuf aujourd'hui ?")
    assert _has_obvious_freshness_signal("les dernières nouvelles")

    # English freshness requests
    assert _has_obvious_freshness_signal("today's news headlines")
    assert _has_obvious_freshness_signal("what's happening today?")
    assert _has_obvious_freshness_signal("latest news updates")
    assert _has_obvious_freshness_signal("current weather forecast")
    assert _has_obvious_freshness_signal("breaking news")

    # Non-freshness requests should NOT match
    assert not _has_obvious_freshness_signal("who is Marie Curie?")
    assert not _has_obvious_freshness_signal("explain quantum computing")
    assert not _has_obvious_freshness_signal("hello how are you")


def test_freshness_heuristic_routes_to_web_research_in_router(
    monkeypatch,
) -> None:
    """Freshness heuristic in the router should bypass LLM and route web_research."""
    settings = _load_repo_settings()

    # The LLM should NOT be called — patch it to fail if called
    class FailingOllamaProvider:
        provider_name = "ollama"

        def __init__(self, **kwargs) -> None:
            del kwargs

        def chat(self, **kwargs):
            raise AssertionError("LLM should not be called for freshness heuristic")

    monkeypatch.setattr(
        "unclaw.core.capability_router.OllamaProvider",
        FailingOllamaProvider,
    )

    summary = build_runtime_capability_summary(
        tool_registry=create_default_tool_registry(settings),
        memory_summary_available=False,
        runtime_mode=RuntimeMode.AGENT,
    )
    decision = LLMCapabilityRouter().route(
        settings=settings,
        profile=resolve_model_profile(settings, "main"),
        user_message="fais moi un résumé des actualités importantes du jour",
        capability_summary=summary,
    )

    assert decision.kind is CapabilityKind.WEB_RESEARCH
    assert decision.confidence == "high"
    assert decision.source == "freshness_heuristic"


def _load_repo_settings():
    return load_settings(project_root=Path(__file__).resolve().parents[1])

