"""Schemas for session-related runtime data."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SessionRecord:
    """Full session data stored in SQLite."""

    id: str
    title: str
    created_at: str
    updated_at: str
    is_active: bool


@dataclass(frozen=True, slots=True)
class SessionSummary:
    """Compact session view used for listings."""

    id: str
    title: str
    created_at: str
    updated_at: str
    is_active: bool

    @classmethod
    def from_record(cls, record: SessionRecord) -> "SessionSummary":
        return cls(
            id=record.id,
            title=record.title,
            created_at=record.created_at,
            updated_at=record.updated_at,
            is_active=record.is_active,
        )
