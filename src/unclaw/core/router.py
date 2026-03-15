"""Top-level routing between explicit runtime modes and capabilities."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from unclaw.core.capability_router import (
    CapabilityKind,
    CapabilityRouter,
    LLMCapabilityRouter,
)
from unclaw.core.capabilities import RuntimeCapabilitySummary
from unclaw.core.runtime_modes import RuntimeMode, RuntimeModeDecision, resolve_runtime_mode
from unclaw.llm.model_profiles import resolve_model_profile
from unclaw.settings import Settings


class RouteKind(StrEnum):
    """Supported runtime routes for Block B."""

    DIRECT_ANSWER = CapabilityKind.DIRECT_ANSWER.value
    WEB_RESEARCH = CapabilityKind.WEB_RESEARCH.value
    LOCAL_FILE_INTENT = CapabilityKind.LOCAL_FILE_INTENT.value
    AMBIGUOUS = CapabilityKind.AMBIGUOUS.value
    AUTOMATION_INTENT = CapabilityKind.AUTOMATION_INTENT.value


@dataclass(frozen=True, slots=True)
class RouteDecision:
    """Result returned by the current capability-aware router."""

    kind: RouteKind
    runtime_mode: RuntimeMode
    model_profile_name: str
    warning_message: str | None = None
    follow_up_message: str | None = None
    route_source: str = "direct"
    route_confidence: str = "high"


def route_request(
    *,
    settings: Settings,
    model_profile_name: str,
    user_message: str,
    capability_summary: RuntimeCapabilitySummary,
    capability_router: CapabilityRouter | None = None,
) -> RouteDecision:
    """Resolve runtime mode, then select the bounded capability path for one turn."""
    profile = resolve_model_profile(settings, model_profile_name)
    runtime_mode = resolve_runtime_mode(profile)
    if runtime_mode.mode is RuntimeMode.CHATBOT:
        return RouteDecision(
            kind=RouteKind.DIRECT_ANSWER,
            runtime_mode=runtime_mode.mode,
            model_profile_name=model_profile_name,
            warning_message=runtime_mode.warning_message,
            route_source="chatbot_fallback",
            route_confidence="high",
        )

    active_capability_router = capability_router or LLMCapabilityRouter()
    capability_decision = active_capability_router.route(
        settings=settings,
        profile=profile,
        user_message=user_message,
        capability_summary=capability_summary,
    )

    return RouteDecision(
        kind=RouteKind(capability_decision.kind.value),
        runtime_mode=runtime_mode.mode,
        model_profile_name=model_profile_name,
        follow_up_message=capability_decision.follow_up_message,
        route_source=capability_decision.source,
        route_confidence=capability_decision.confidence,
    )


__all__ = [
    "RouteDecision",
    "RouteKind",
    "route_request",
]

