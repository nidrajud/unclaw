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


@dataclass(frozen=True, slots=True)
class UserRequest:
    """One user turn entering the runtime."""

    session_id: str
    content: str
    created_at: str | None = None


@dataclass(frozen=True, slots=True)
class AssistantResponse:
    """One assistant turn returned by the runtime."""

    session_id: str
    content: str
    created_at: str | None = None
