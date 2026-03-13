"""Minimal orchestration for one chat turn."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from unclaw.core.context_builder import build_context_messages
from unclaw.core.session_manager import SessionManager
from unclaw.errors import UnclawError
from unclaw.llm.base import LLMContentCallback, LLMError, LLMResponse
from unclaw.llm.model_profiles import resolve_model_profile
from unclaw.llm.ollama_provider import OllamaProvider
from unclaw.logs.tracer import Tracer
from unclaw.settings import Settings


class OrchestratorError(UnclawError):
    """Raised when the minimal orchestrator cannot run a turn."""


class ModelCallFailedError(OrchestratorError):
    """Raised when the provider call fails after the route has been selected."""

    def __init__(
        self,
        *,
        provider: str,
        model_profile_name: str,
        model_name: str,
        duration_ms: int,
        error: str,
    ) -> None:
        super().__init__(error)
        self.provider = provider
        self.model_profile_name = model_profile_name
        self.model_name = model_name
        self.duration_ms = duration_ms
        self.error = error


@dataclass(frozen=True, slots=True)
class OrchestratorTurnResult:
    """Completed model turn metadata returned to the runtime."""

    response: LLMResponse
    model_duration_ms: int


@dataclass(slots=True)
class Orchestrator:
    """Coordinate the minimal model call for one user turn."""

    settings: Settings
    session_manager: SessionManager
    tracer: Tracer

    def run_turn(
        self,
        *,
        session_id: str,
        user_message: str,
        model_profile_name: str,
        max_history_size: int | None = 20,
        thinking_enabled: bool = False,
        content_callback: LLMContentCallback | None = None,
    ) -> OrchestratorTurnResult:
        """Resolve the model, build context, and call the provider."""
        profile = resolve_model_profile(self.settings, model_profile_name)
        provider = self._create_provider(profile.provider)
        context_messages = build_context_messages(
            session_manager=self.session_manager,
            session_id=session_id,
            user_message=user_message,
            max_history_size=max_history_size,
        )

        self.tracer.trace_model_called(
            session_id=session_id,
            provider=profile.provider,
            model_profile_name=profile.name,
            model_name=profile.model_name,
            message_count=len(context_messages),
        )

        model_started_at = perf_counter()
        try:
            response = provider.chat(
                profile=profile,
                messages=context_messages,
                thinking_enabled=thinking_enabled,
                content_callback=content_callback,
            )
        except LLMError as exc:
            raise ModelCallFailedError(
                provider=profile.provider,
                model_profile_name=profile.name,
                model_name=profile.model_name,
                duration_ms=_elapsed_ms(model_started_at),
                error=str(exc),
            ) from exc

        return OrchestratorTurnResult(
            response=response,
            model_duration_ms=_elapsed_ms(model_started_at),
        )

    def _create_provider(self, provider_name: str) -> OllamaProvider:
        if provider_name == OllamaProvider.provider_name:
            return OllamaProvider()

        raise OrchestratorError(
            f"Provider '{provider_name}' is not supported by the minimal runtime."
        )


def _elapsed_ms(started_at: float) -> int:
    return max(0, round((perf_counter() - started_at) * 1000))
