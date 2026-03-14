"""Schemas for chat-related runtime data."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class MessageRole(StrEnum):
    """Supported roles for stored chat messages."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """A persisted chat message belonging to one session."""

    id: str
    session_id: str
    role: MessageRole
    content: str
    created_at: str
