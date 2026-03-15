"""Runtime mode resolution for the current model profile."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from unclaw.constants import CHATBOT_MODE_FALLBACK_WARNING
from unclaw.llm.base import ResolvedModelProfile


class RuntimeMode(StrEnum):
    """Supported top-level runtime behaviors."""

    AGENT = "agent"
    CHATBOT = "chatbot"


@dataclass(frozen=True, slots=True)
class RuntimeModeDecision:
    """Resolved runtime mode for one selected model profile."""

    mode: RuntimeMode
    warning_message: str | None = None


def resolve_runtime_mode(profile: ResolvedModelProfile) -> RuntimeModeDecision:
    """Select the explicit runtime mode allowed by one model profile."""
    if profile.capabilities.supports_agent_mode:
        return RuntimeModeDecision(mode=RuntimeMode.AGENT)

    return RuntimeModeDecision(
        mode=RuntimeMode.CHATBOT,
        warning_message=CHATBOT_MODE_FALLBACK_WARNING,
    )


def format_runtime_mode(mode: RuntimeMode) -> str:
    """Return a short user-facing runtime mode label."""
    if mode is RuntimeMode.AGENT:
        return "Agent mode"
    return "Chatbot mode"

