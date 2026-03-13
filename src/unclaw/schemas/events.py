"""Schemas for runtime event logging."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class EventLevel(StrEnum):
    """Supported severity levels for runtime events."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class RuntimeEvent:
    """One runtime event persisted for observability."""

    id: str
    session_id: str | None
    event_type: str
    level: EventLevel
    message: str
    payload_json: str | None
    created_at: str
