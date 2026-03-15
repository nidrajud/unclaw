"""Runtime entrypoint for one non-command user turn."""

from __future__ import annotations

from collections.abc import Callable
from time import perf_counter
from typing import TYPE_CHECKING

from unclaw.core.capabilities import build_runtime_capability_summary
from unclaw.core.capability_router import CapabilityRouter
from unclaw.core.chat_flow import run_direct_chat_turn
from unclaw.core.command_handler import CommandHandler
from unclaw.core.executor import ToolExecutor, create_default_tool_registry
from unclaw.core.research_flow import run_search_then_answer
from unclaw.core.router import RouteKind, route_request
from unclaw.core.session_manager import SessionManager
from unclaw.logs.event_bus import EventBus
from unclaw.logs.tracer import Tracer
from unclaw.schemas.chat import MessageRole
from unclaw.tools.contracts import ToolCall
from unclaw.tools.web_tools import SEARCH_WEB_DEFINITION

if TYPE_CHECKING:
    from unclaw.llm.base import LLMContentCallback
    from unclaw.tools.registry import ToolRegistry


def run_user_turn(
    *,
    session_manager: SessionManager,
    command_handler: CommandHandler,
    user_input: str,
    tracer: Tracer | None = None,
    event_bus: EventBus | None = None,
    stream_output_func: LLMContentCallback | None = None,
    tool_registry: ToolRegistry | None = None,
    tool_executor: ToolExecutor | None = None,
    capability_router: CapabilityRouter | None = None,
    assistant_reply_transform: Callable[[str], str] | None = None,
) -> str:
    """Run the Block B routed runtime path and persist the assistant reply."""
    session = session_manager.ensure_current_session()
    active_tracer = tracer or Tracer(
        event_bus=event_bus or EventBus(),
        event_repository=session_manager.event_repository,
        include_reasoning_text=(
            session_manager.settings.app.logging.include_reasoning_text
        ),
    )
    active_tracer.runtime_log_path = (
        session_manager.settings.paths.log_file_path
        if session_manager.settings.app.logging.file_enabled
        else None
    )

    selected_model_profile_name = command_handler.current_model_profile.name
    selected_model = command_handler.current_model_profile
    runtime_mode_decision = command_handler.current_runtime_mode_decision()
    thinking_enabled = command_handler.thinking_enabled is True
    active_tool_registry = (
        tool_registry
        or getattr(tool_executor, "registry", None)
        or create_default_tool_registry(session_manager.settings)
    )
    active_tool_executor = tool_executor or ToolExecutor(registry=active_tool_registry)
    capability_summary = build_runtime_capability_summary(
        tool_registry=active_tool_registry,
        memory_summary_available=command_handler.memory_manager is not None,
        runtime_mode=runtime_mode_decision.mode,
    )
    turn_started_at = perf_counter()

    active_tracer.trace_runtime_started(
        session_id=session.id,
        model_profile_name=selected_model_profile_name,
        provider=selected_model.provider,
        model_name=selected_model.model_name,
        thinking_enabled=thinking_enabled,
        input_length=len(user_input),
        runtime_mode=runtime_mode_decision.mode.value,
    )

    route = route_request(
        settings=session_manager.settings,
        model_profile_name=selected_model_profile_name,
        user_message=user_input,
        capability_summary=capability_summary,
        capability_router=capability_router,
    )
    active_tracer.trace_route_selected(
        session_id=session.id,
        route_kind=route.kind.value,
        model_profile_name=route.model_profile_name,
        runtime_mode=route.runtime_mode.value,
        route_source=route.route_source,
        route_confidence=route.route_confidence,
    )

    if route.kind is RouteKind.DIRECT_ANSWER:
        warning_message = command_handler.consume_runtime_mode_warning()
        return run_direct_chat_turn(
            session_manager=session_manager,
            command_handler=command_handler,
            user_input=user_input,
            tracer=active_tracer,
            stream_output_func=stream_output_func,
            tool_registry=active_tool_registry,
            capability_summary=capability_summary,
            assistant_reply_transform=_compose_reply_transforms(
                _warning_prefix_transform(warning_message),
                assistant_reply_transform,
            ),
        )

    if route.kind is RouteKind.WEB_RESEARCH:
        return run_search_then_answer(
            session_manager=session_manager,
            command_handler=command_handler,
            tracer=active_tracer,
            tool_executor=active_tool_executor,
            tool_call=_build_web_research_tool_call(user_input),
            persist_user_message=False,
            assistant_reply_transform=assistant_reply_transform,
        ).assistant_reply

    assistant_reply = _build_non_autonomous_reply(
        route_kind=route.kind,
        follow_up_message=route.follow_up_message,
    )
    if assistant_reply_transform is not None:
        assistant_reply = assistant_reply_transform(assistant_reply)

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


def _build_non_autonomous_reply(
    *,
    route_kind: RouteKind,
    follow_up_message: str | None,
) -> str:
    if isinstance(follow_up_message, str) and follow_up_message.strip():
        return follow_up_message.strip()

    if route_kind is RouteKind.LOCAL_FILE_INTENT:
        return (
            "Tell me which local file or folder you want to inspect. "
            "You can also use /read <path> or /ls [path]."
        )
    if route_kind is RouteKind.AUTOMATION_INTENT:
        return (
            "That looks like a system or automation request. "
            "Unclaw does not execute that autonomously in this runtime path yet."
        )
    return (
        "I am not sure whether you want a direct answer, web research, "
        "or help with a local file. Please clarify."
    )


def _build_web_research_tool_call(user_input: str) -> ToolCall:
    return ToolCall(
        tool_name=SEARCH_WEB_DEFINITION.name,
        arguments={"query": user_input.strip()},
    )


def _warning_prefix_transform(
    warning_message: str | None,
) -> Callable[[str], str] | None:
    if warning_message is None:
        return None

    def apply(reply_text: str) -> str:
        return f"{warning_message}\n\n{reply_text}"

    return apply


def _compose_reply_transforms(
    *transforms: Callable[[str], str] | None,
) -> Callable[[str], str] | None:
    active_transforms = tuple(transform for transform in transforms if transform is not None)
    if not active_transforms:
        return None

    def apply(reply_text: str) -> str:
        updated_reply = reply_text
        for transform in active_transforms:
            updated_reply = transform(updated_reply)
        return updated_reply

    return apply


def _elapsed_ms(started_at: float) -> int:
    return max(0, round((perf_counter() - started_at) * 1000))
