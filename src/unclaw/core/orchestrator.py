"""Minimal orchestration for one chat turn."""

from __future__ import annotations

from dataclasses import dataclass

from unclaw.core.context_builder import build_context_messages
from unclaw.core.session_manager import SessionManager
from unclaw.errors import UnclawError
from unclaw.llm.base import LLMResponse
from unclaw.llm.model_profiles import resolve_model_profile
from unclaw.llm.ollama_provider import OllamaProvider
from unclaw.logs.tracer import Tracer
from unclaw.settings import Settings


class OrchestratorError(UnclawError):
    """Raised when the minimal orchestrator cannot run a turn."""


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
    ) -> LLMResponse:
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

        response = provider.chat(profile=profile, messages=context_messages)

        self.tracer.trace_model_succeeded(
            session_id=session_id,
            provider=response.provider,
            model_name=response.model_name,
            finish_reason=response.finish_reason,
            output_length=len(response.content),
        )
        return response

    def _create_provider(self, provider_name: str) -> OllamaProvider:
        if provider_name == OllamaProvider.provider_name:
            return OllamaProvider()

        raise OrchestratorError(
            f"Provider '{provider_name}' is not supported by the minimal runtime."
        )

