"""Direct-answer chat flow shared by routed runtime paths."""

from __future__ import annotations

from collections.abc import Callable
from time import perf_counter
from typing import TYPE_CHECKING

from unclaw.constants import EMPTY_RESPONSE_REPLY, RUNTIME_ERROR_REPLY
from unclaw.core.capabilities import build_runtime_capability_summary
from unclaw.core.executor import create_default_tool_registry
from unclaw.core.orchestrator import (
    ModelCallFailedError,
    Orchestrator,
    OrchestratorError,
)
from unclaw.core.session_manager import SessionManagerError
from unclaw.core.runtime_modes import resolve_runtime_mode
from unclaw.errors import ConfigurationError
from unclaw.llm.base import LLMContentCallback, LLMError
from unclaw.llm.model_profiles import resolve_model_profile
from unclaw.logs.tracer import Tracer
from unclaw.schemas.chat import MessageRole

if TYPE_CHECKING:
    from unclaw.core.capabilities import RuntimeCapabilitySummary
    from unclaw.core.command_handler import CommandHandler
    from unclaw.core.session_manager import SessionManager
    from unclaw.tools.registry import ToolRegistry


def run_direct_chat_turn(
    *,
    session_manager: SessionManager,
    command_handler: CommandHandler,
    user_input: str,
    tracer: Tracer,
    stream_output_func: LLMContentCallback | None = None,
    tool_registry: ToolRegistry | None = None,
    capability_summary: RuntimeCapabilitySummary | None = None,
    assistant_reply_transform: Callable[[str], str] | None = None,
) -> str:
    """Call the selected local model once and persist the assistant reply."""
    session = session_manager.ensure_current_session()
    selected_model_profile_name = command_handler.current_model_profile.name
    selected_model = command_handler.current_model_profile
    thinking_enabled = command_handler.thinking_enabled is True
    active_capability_summary = capability_summary
    if active_capability_summary is None:
        active_tool_registry = tool_registry or create_default_tool_registry(
            session_manager.settings
        )
        runtime_mode = resolve_runtime_mode(
            resolve_model_profile(
                session_manager.settings,
                selected_model_profile_name,
            )
        )
        active_capability_summary = build_runtime_capability_summary(
            tool_registry=active_tool_registry,
            memory_summary_available=command_handler.memory_manager is not None,
            runtime_mode=runtime_mode.mode,
        )

    turn_started_at = perf_counter()
    try:
        orchestrator = Orchestrator(
            settings=session_manager.settings,
            session_manager=session_manager,
            tracer=tracer,
        )
        response = orchestrator.run_turn(
            session_id=session.id,
            user_message=user_input,
            model_profile_name=selected_model_profile_name,
            capability_summary=active_capability_summary,
            thinking_enabled=thinking_enabled,
            content_callback=stream_output_func,
        )
        tracer.trace_model_succeeded(
            session_id=session.id,
            provider=response.response.provider,
            model_name=response.response.model_name,
            finish_reason=response.response.finish_reason,
            output_length=len(response.response.content),
            model_duration_ms=response.model_duration_ms,
            reasoning=response.response.reasoning,
        )
        assistant_reply = response.response.content.strip() or EMPTY_RESPONSE_REPLY
    except ModelCallFailedError as exc:
        tracer.trace_model_failed(
            session_id=session.id,
            provider=exc.provider,
            model_profile_name=exc.model_profile_name,
            model_name=exc.model_name,
            model_duration_ms=exc.duration_ms,
            error=str(exc),
        )
        assistant_reply = RUNTIME_ERROR_REPLY
    except (
        ConfigurationError,
        LLMError,
        OrchestratorError,
        SessionManagerError,
    ) as exc:
        tracer.trace_model_failed(
            session_id=session.id,
            provider=selected_model.provider,
            model_profile_name=selected_model_profile_name,
            model_name=selected_model.model_name,
            error=str(exc),
        )
        assistant_reply = RUNTIME_ERROR_REPLY

    if assistant_reply_transform is not None:
        assistant_reply = assistant_reply_transform(assistant_reply)

    session_manager.add_message(
        MessageRole.ASSISTANT,
        assistant_reply,
        session_id=session.id,
    )
    tracer.trace_assistant_reply_persisted(
        session_id=session.id,
        output_length=len(assistant_reply),
        turn_duration_ms=_elapsed_ms(turn_started_at),
    )
    return assistant_reply


def _elapsed_ms(started_at: float) -> int:
    return max(0, round((perf_counter() - started_at) * 1000))
