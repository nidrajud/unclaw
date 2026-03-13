"""Minimal runtime entrypoint for one non-command chat turn."""

from __future__ import annotations

from time import perf_counter

from unclaw.core.command_handler import CommandHandler
from unclaw.core.orchestrator import (
    ModelCallFailedError,
    Orchestrator,
    OrchestratorError,
)
from unclaw.core.router import route_request
from unclaw.core.session_manager import SessionManager, SessionManagerError
from unclaw.errors import ConfigurationError
from unclaw.llm.base import LLMContentCallback, LLMError
from unclaw.logs.event_bus import EventBus
from unclaw.logs.tracer import Tracer
from unclaw.schemas.chat import MessageRole

_RUNTIME_ERROR_REPLY = (
    "I could not complete that request locally right now. "
    "Check that Ollama is running and the selected model is available."
)
_EMPTY_RESPONSE_REPLY = "The local model returned an empty response."


def run_user_turn(
    *,
    session_manager: SessionManager,
    command_handler: CommandHandler,
    user_input: str,
    tracer: Tracer | None = None,
    event_bus: EventBus | None = None,
    stream_output_func: LLMContentCallback | None = None,
) -> str:
    """Run the minimal runtime path and persist the assistant reply."""
    session = session_manager.ensure_current_session()
    active_tracer = tracer or Tracer(
        event_bus=event_bus or EventBus(),
        event_repository=session_manager.event_repository,
    )
    active_tracer.runtime_log_path = (
        session_manager.settings.paths.log_file_path
        if session_manager.settings.app.logging.file_enabled
        else None
    )

    selected_model_profile_name = command_handler.current_model_profile.name
    selected_model = command_handler.current_model_profile
    thinking_enabled = command_handler.thinking_enabled is True
    turn_started_at = perf_counter()

    active_tracer.trace_runtime_started(
        session_id=session.id,
        model_profile_name=selected_model_profile_name,
        provider=selected_model.provider,
        model_name=selected_model.model_name,
        thinking_enabled=thinking_enabled,
        input_length=len(user_input),
    )

    try:
        route = route_request(
            settings=session_manager.settings,
            model_profile_name=selected_model_profile_name,
            thinking_enabled=thinking_enabled,
        )
        active_tracer.trace_route_selected(
            session_id=session.id,
            route_kind=route.kind.value,
            model_profile_name=route.model_profile_name,
        )

        orchestrator = Orchestrator(
            settings=session_manager.settings,
            session_manager=session_manager,
            tracer=active_tracer,
        )
        response = orchestrator.run_turn(
            session_id=session.id,
            user_message=user_input,
            model_profile_name=route.model_profile_name,
            thinking_enabled=thinking_enabled,
            content_callback=stream_output_func,
        )
        active_tracer.trace_model_succeeded(
            session_id=session.id,
            provider=response.response.provider,
            model_name=response.response.model_name,
            finish_reason=response.response.finish_reason,
            output_length=len(response.response.content),
            model_duration_ms=response.model_duration_ms,
            reasoning=response.response.reasoning,
        )
        assistant_reply = response.response.content.strip() or _EMPTY_RESPONSE_REPLY
    except ModelCallFailedError as exc:
        active_tracer.trace_model_failed(
            session_id=session.id,
            provider=exc.provider,
            model_profile_name=exc.model_profile_name,
            model_name=exc.model_name,
            model_duration_ms=exc.duration_ms,
            error=str(exc),
        )
        assistant_reply = _RUNTIME_ERROR_REPLY
    except (
        ConfigurationError,
        LLMError,
        OrchestratorError,
        SessionManagerError,
    ) as exc:
        active_tracer.trace_model_failed(
            session_id=session.id,
            provider=selected_model.provider,
            model_profile_name=selected_model_profile_name,
            model_name=selected_model.model_name,
            error=str(exc),
        )
        assistant_reply = _RUNTIME_ERROR_REPLY

    session_manager.add_message(
        MessageRole.ASSISTANT,
        assistant_reply,
        session_id=session.id,
    )
    active_tracer.trace_assistant_reply_persisted(
        session_id=session.id,
        output_length=len(assistant_reply),
        turn_duration_ms=_elapsed_ms(turn_started_at),
    )
    return assistant_reply


def _elapsed_ms(started_at: float) -> int:
    return max(0, round((perf_counter() - started_at) * 1000))
