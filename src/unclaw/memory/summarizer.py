"""Deterministic session summarization for persisted chat messages."""

from __future__ import annotations

from collections.abc import Iterable

from unclaw.schemas.chat import ChatMessage, MessageRole

_MAX_INTENT_COUNT = 3
_MAX_INTENT_LENGTH = 80
_MAX_REPLY_LENGTH = 120


def summarize_session_messages(messages: Iterable[ChatMessage]) -> str:
    """Build a compact rule-based summary from persisted session messages."""

    ordered_messages = list(messages)
    if not ordered_messages:
        return "No messages yet."

    user_intents = _collect_recent_user_intents(ordered_messages)
    latest_assistant_reply = _find_latest_reply(
        ordered_messages,
        role=MessageRole.ASSISTANT,
        limit=_MAX_REPLY_LENGTH,
    )

    user_count = sum(1 for message in ordered_messages if message.role == MessageRole.USER)
    assistant_count = sum(
        1 for message in ordered_messages if message.role == MessageRole.ASSISTANT
    )

    parts: list[str] = []
    if user_intents:
        label = "User intent" if len(user_intents) == 1 else "Recent user intents"
        parts.append(f"{label}: {'; '.join(user_intents)}.")

    if latest_assistant_reply is not None:
        parts.append(f"Latest assistant reply: {latest_assistant_reply}.")

    if not parts:
        parts.append("Session has messages but no user or assistant content yet.")

    parts.append(
        "Session size: "
        f"{len(ordered_messages)} messages ({user_count} user, {assistant_count} assistant)."
    )
    return " ".join(parts)


def _collect_recent_user_intents(messages: list[ChatMessage]) -> list[str]:
    snippets: list[str] = []
    seen_snippets: set[str] = set()

    for message in reversed(messages):
        if message.role != MessageRole.USER:
            continue

        snippet = _summary_fragment(message.content, limit=_MAX_INTENT_LENGTH)
        if snippet is None:
            continue

        normalized_snippet = snippet.casefold()
        if normalized_snippet in seen_snippets:
            continue

        snippets.append(snippet)
        seen_snippets.add(normalized_snippet)
        if len(snippets) >= _MAX_INTENT_COUNT:
            break

    snippets.reverse()
    return snippets


def _find_latest_reply(
    messages: list[ChatMessage],
    *,
    role: MessageRole,
    limit: int,
) -> str | None:
    for message in reversed(messages):
        if message.role != role:
            continue

        snippet = _summary_fragment(message.content, limit=limit)
        if snippet is not None:
            return snippet

    return None


def _message_snippet(content: str, *, limit: int) -> str | None:
    normalized = " ".join(content.split()).strip()
    if not normalized:
        return None
    return _clip_text(normalized, limit=limit)


def _summary_fragment(content: str, *, limit: int) -> str | None:
    snippet = _message_snippet(content, limit=limit)
    if snippet is None:
        return None

    cleaned_snippet = snippet.rstrip(" .!?;:")
    return cleaned_snippet or None


def _clip_text(value: str, *, limit: int) -> str:
    if len(value) <= limit:
        return value

    clipped = value[: limit - 3].rstrip(" ,;:.")
    return f"{clipped}..."
