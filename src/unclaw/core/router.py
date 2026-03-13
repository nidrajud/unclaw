"""Minimal rule-based routing for the current runtime phase."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from unclaw.errors import ConfigurationError
from unclaw.settings import Settings


class RouteKind(StrEnum):
    """Supported runtime routes for the current phase."""

    COMMAND = "command"
    CHAT = "chat"
    CHAT_WITH_THINKING = "chat_with_thinking"


@dataclass(frozen=True, slots=True)
class RouteDecision:
    """Result returned by the current lightweight router."""

    kind: RouteKind
    model_profile_name: str


def route_request(
    *,
    settings: Settings,
    model_profile_name: str,
    thinking_enabled: bool,
    is_command: bool = False,
) -> RouteDecision:
    """Select the current route with simple deterministic rules."""
    profile = settings.models.get(model_profile_name)
    if profile is None:
        raise ConfigurationError(
            f"Model profile '{model_profile_name}' is not defined in settings."
        )

    if is_command:
        route_kind = RouteKind.COMMAND
    elif thinking_enabled and profile.thinking_supported:
        route_kind = RouteKind.CHAT_WITH_THINKING
    else:
        route_kind = RouteKind.CHAT

    return RouteDecision(
        kind=route_kind,
        model_profile_name=model_profile_name,
    )

