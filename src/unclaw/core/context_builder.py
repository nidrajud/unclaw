"""Context assembly for the minimal runtime path."""

from __future__ import annotations

from collections.abc import Sequence

from unclaw.core.capabilities import (
    RuntimeCapabilitySummary,
    build_runtime_capability_context,
)
from unclaw.core.session_manager import SessionManager
from unclaw.llm.base import LLMMessage, LLMRole
from unclaw.schemas.chat import ChatMessage, MessageRole


def build_context_messages(
    *,
    session_manager: SessionManager,
    session_id: str,
    user_message: str,
    max_history_size: int | None = 20,
    capability_summary: RuntimeCapabilitySummary | None = None,
) -> list[LLMMessage]:
    """Build the minimal message list sent to the selected model."""
    normalized_user_message = user_message.strip()
    if not normalized_user_message:
        raise ValueError("user_message must be a non-empty string.")

    history = session_manager.list_messages(session_id)
    recent_history = _limit_history(history, max_history_size)

    context_messages = [
        LLMMessage(role=LLMRole.SYSTEM, content=session_manager.settings.system_prompt)
    ]
    if capability_summary is not None:
        context_messages.append(
            LLMMessage(
                role=LLMRole.SYSTEM,
                content=build_runtime_capability_context(capability_summary),
            )
        )
    context_messages.extend(_to_llm_message(message) for message in recent_history)

    if _should_append_current_user_message(recent_history, normalized_user_message):
        context_messages.append(
            LLMMessage(role=LLMRole.USER, content=normalized_user_message)
        )

    return context_messages


def _limit_history(
    history: Sequence[ChatMessage],
    max_history_size: int | None,
) -> Sequence[ChatMessage]:
    if max_history_size is None:
        return history
    if max_history_size < 1:
        return ()
    return history[-max_history_size:]


def _to_llm_message(message: ChatMessage) -> LLMMessage:
    return LLMMessage(role=LLMRole(message.role.value), content=message.content)


def _should_append_current_user_message(
    history: Sequence[ChatMessage],
    user_message: str,
) -> bool:
    if not history:
        return True

    latest_message = history[-1]
    if latest_message.role is not MessageRole.USER:
        return True

    return latest_message.content.strip() != user_message
