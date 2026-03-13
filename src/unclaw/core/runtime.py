"""Minimal runtime entrypoint for one non-command chat turn."""

from __future__ import annotations

from unclaw.core.command_handler import CommandHandler
from unclaw.core.orchestrator import Orchestrator, OrchestratorError
from unclaw.core.router import route_request
from unclaw.core.session_manager import SessionManager, SessionManagerError
from unclaw.errors import ConfigurationError
from unclaw.llm.base import LLMError
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
    thinking_enabled = command_handler.thinking_enabled is True

    active_tracer.trace_runtime_started(
        session_id=session.id,
        model_profile_name=selected_model_profile_name,
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
        )
        assistant_reply = response.content.strip() or _EMPTY_RESPONSE_REPLY
    except (
        ConfigurationError,
        LLMError,
        OrchestratorError,
        SessionManagerError,
    ) as exc:
        active_tracer.trace_model_failed(
            session_id=session.id,
            provider=_resolve_provider_name(
                session_manager=session_manager,
                model_profile_name=selected_model_profile_name,
            ),
            model_profile_name=selected_model_profile_name,
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
    )
    return assistant_reply


def _resolve_provider_name(
    *,
    session_manager: SessionManager,
    model_profile_name: str,
) -> str | None:
    profile = session_manager.settings.models.get(model_profile_name)
    if profile is None:
        return None
    return profile.provider
