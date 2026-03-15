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


def _load_repo_settings():
    return load_settings(project_root=Path(__file__).resolve().parents[1])

