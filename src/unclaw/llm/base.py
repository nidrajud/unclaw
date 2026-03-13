"""Core abstractions for local LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from unclaw.errors import UnclawError


class LLMError(UnclawError):
    """Base exception for LLM-related failures."""


class LLMProviderError(LLMError):
    """Raised when a provider call fails."""


class LLMConnectionError(LLMProviderError):
    """Raised when a local LLM provider cannot be reached."""


class LLMResponseError(LLMProviderError):
    """Raised when a provider response is invalid or incomplete."""


class LLMRole(StrEnum):
    """Supported message roles for provider chat calls."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass(frozen=True, slots=True)
class LLMMessage:
    """One chat message sent to a provider."""

    role: LLMRole | str
    content: str

    def as_payload(self) -> dict[str, str]:
        """Return a provider-friendly message payload."""
        return {"role": str(self.role), "content": self.content}


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """Normalized response returned by a provider."""

    provider: str
    model_name: str
    content: str
    created_at: str
    finish_reason: str | None = None
    raw_payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ModelCapabilities:
    """Simple capability flags derived from one configured model profile."""

    thinking_supported: bool
    tool_mode: str
    supports_tools: bool
    supports_native_tool_calling: bool
    supports_streaming: bool = False
    supports_vision: bool = False


@dataclass(frozen=True, slots=True)
class ResolvedModelProfile:
    """Runtime-ready model profile resolved from settings."""

    name: str
    provider: str
    model_name: str
    temperature: float
    capabilities: ModelCapabilities


class BaseLLMProvider(ABC):
    """Common interface for local LLM providers."""

    provider_name: str = ""

    @abstractmethod
    def chat(
        self,
        profile: ResolvedModelProfile,
        messages: Sequence[LLMMessage],
        *,
        timeout_seconds: float | None = None,
    ) -> LLMResponse:
        """Run a synchronous chat request."""

    @abstractmethod
    def is_available(self, *, timeout_seconds: float | None = None) -> bool:
        """Return whether the provider looks reachable."""

    def validate_profile(self, profile: ResolvedModelProfile) -> None:
        """Ensure the resolved profile belongs to this provider."""
        if profile.provider != self.provider_name:
            raise LLMProviderError(
                f"Model profile '{profile.name}' uses provider "
                f"'{profile.provider}', not '{self.provider_name}'."
            )


def utc_now_iso() -> str:
    """Return an ISO 8601 UTC timestamp."""
    return datetime.now(tz=UTC).isoformat().replace("+00:00", "Z")
