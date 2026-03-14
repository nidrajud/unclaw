"""Small SQLite repositories for sessions, messages, and events."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from uuid import uuid4

from unclaw.llm.base import utc_now_iso
from unclaw.schemas.chat import ChatMessage, MessageRole
from unclaw.schemas.events import EventLevel, RuntimeEvent
from unclaw.schemas.session import SessionRecord, SessionSummary


@dataclass(slots=True)
class SessionRepository:
    """Persistence helpers for chat sessions."""

    connection: sqlite3.Connection

    def create_session(
        self,
        title: str = "New session",
        *,
        session_id: str | None = None,
        created_at: str | None = None,
        is_active: bool = True,
    ) -> SessionRecord:
        timestamp = created_at or utc_now_iso()
        record = SessionRecord(
            id=session_id or _new_id("sess"),
            title=_normalize_title(title),
            created_at=timestamp,
            updated_at=timestamp,
            is_active=is_active,
        )

        self.connection.execute(
            """
            INSERT INTO sessions (id, title, created_at, updated_at, is_active)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                record.id,
                record.title,
                record.created_at,
                record.updated_at,
                _bool_to_db(record.is_active),
            ),
        )
        self.connection.commit()
        return record

    def list_sessions(self) -> list[SessionSummary]:
        rows = self.connection.execute(
            """
            SELECT id, title, created_at, updated_at, is_active
            FROM sessions
            ORDER BY updated_at DESC, rowid DESC
            """
        ).fetchall()
        return [_session_summary_from_row(row) for row in rows]

    def get_session(self, session_id: str) -> SessionRecord | None:
        row = self.connection.execute(
            """
            SELECT id, title, created_at, updated_at, is_active
            FROM sessions
            WHERE id = ?
            """,
            (session_id,),
        ).fetchone()
        if row is None:
            return None
        return _session_record_from_row(row)

    def get_summary_text(self, session_id: str) -> str | None:
        row = self.connection.execute(
            """
            SELECT summary_text
            FROM sessions
            WHERE id = ?
            """,
            (session_id,),
        ).fetchone()
        if row is None:
            return None
        return row["summary_text"]

    def update_session(
        self,
        session_id: str,
        *,
        updated_at: str | None = None,
        title: str | None = None,
        is_active: bool | None = None,
    ) -> SessionRecord | None:
        cursor = self.connection.execute(
            """
            UPDATE sessions
            SET title = COALESCE(?, title),
                updated_at = ?,
                is_active = COALESCE(?, is_active)
            WHERE id = ?
            """,
            (
                _normalize_title(title) if title is not None else None,
                updated_at or utc_now_iso(),
                _bool_to_db(is_active) if is_active is not None else None,
                session_id,
            ),
        )
        if cursor.rowcount == 0:
            self.connection.rollback()
            return None

        self.connection.commit()
        return self.get_session(session_id)

    def set_active_session(self, session_id: str) -> SessionRecord | None:
        if self.get_session(session_id) is None:
            return None

        self.connection.execute(
            """
            UPDATE sessions
            SET is_active = CASE WHEN id = ? THEN 1 ELSE 0 END
            WHERE id = ? OR is_active = 1
            """,
            (session_id, session_id),
        )
        self.connection.commit()
        return self.get_session(session_id)

    def update_summary_text(
        self,
        session_id: str,
        summary_text: str | None,
    ) -> str | None:
        cursor = self.connection.execute(
            """
            UPDATE sessions
            SET summary_text = ?
            WHERE id = ?
            """,
            (_normalize_optional_text(summary_text), session_id),
        )
        if cursor.rowcount == 0:
            self.connection.rollback()
            return None

        self.connection.commit()
        return self.get_summary_text(session_id)


@dataclass(slots=True)
class MessageRepository:
    """Persistence helpers for chat messages."""

    connection: sqlite3.Connection

    def add_message(
        self,
        session_id: str,
        role: MessageRole | str,
        content: str,
        *,
        message_id: str | None = None,
        created_at: str | None = None,
    ) -> ChatMessage:
        timestamp = created_at or utc_now_iso()
        message = ChatMessage(
            id=message_id or _new_id("msg"),
            session_id=session_id,
            role=MessageRole(role),
            content=_require_text(content, field_name="content"),
            created_at=timestamp,
        )

        self.connection.execute(
            """
            INSERT INTO messages (id, session_id, role, content, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                message.id,
                message.session_id,
                message.role.value,
                message.content,
                message.created_at,
            ),
        )
        self.connection.execute(
            """
            UPDATE sessions
            SET updated_at = ?
            WHERE id = ?
            """,
            (timestamp, session_id),
        )
        self.connection.commit()
        return message

    def list_messages(self, session_id: str) -> list[ChatMessage]:
        rows = self.connection.execute(
            """
            SELECT id, session_id, role, content, created_at
            FROM messages
            WHERE session_id = ?
            ORDER BY created_at ASC, rowid ASC
            """,
            (session_id,),
        ).fetchall()
        return [_chat_message_from_row(row) for row in rows]


@dataclass(slots=True)
class EventRepository:
    """Persistence helpers for runtime events."""

    connection: sqlite3.Connection

    def add_event(
        self,
        *,
        session_id: str | None,
        event_type: str,
        level: EventLevel | str,
        message: str,
        payload_json: str | None = None,
        event_id: str | None = None,
        created_at: str | None = None,
    ) -> RuntimeEvent:
        timestamp = created_at or utc_now_iso()
        event = RuntimeEvent(
            id=event_id or _new_id("evt"),
            session_id=session_id,
            event_type=_require_text(event_type, field_name="event_type"),
            level=EventLevel(level),
            message=_require_text(message, field_name="message"),
            payload_json=payload_json,
            created_at=timestamp,
        )

        self.connection.execute(
            """
            INSERT INTO events (
                id,
                session_id,
                event_type,
                level,
                message,
                payload_json,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.id,
                event.session_id,
                event.event_type,
                event.level.value,
                event.message,
                event.payload_json,
                event.created_at,
            ),
        )
        self.connection.commit()
        return event

    def list_recent_events(
        self,
        session_id: str,
        *,
        limit: int = 50,
    ) -> list[RuntimeEvent]:
        if limit < 1:
            return []

        rows = self.connection.execute(
            """
            SELECT id, session_id, event_type, level, message, payload_json, created_at
            FROM events
            WHERE session_id = ?
            ORDER BY created_at DESC, rowid DESC
            LIMIT ?
            """,
            (session_id, limit),
        ).fetchall()
        return [_runtime_event_from_row(row) for row in rows]


def _session_record_from_row(row: sqlite3.Row) -> SessionRecord:
    return SessionRecord(
        id=row["id"],
        title=row["title"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        is_active=_db_to_bool(row["is_active"]),
    )


def _session_summary_from_row(row: sqlite3.Row) -> SessionSummary:
    return SessionSummary(
        id=row["id"],
        title=row["title"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        is_active=_db_to_bool(row["is_active"]),
    )


def _chat_message_from_row(row: sqlite3.Row) -> ChatMessage:
    return ChatMessage(
        id=row["id"],
        session_id=row["session_id"],
        role=MessageRole(row["role"]),
        content=row["content"],
        created_at=row["created_at"],
    )


def _runtime_event_from_row(row: sqlite3.Row) -> RuntimeEvent:
    return RuntimeEvent(
        id=row["id"],
        session_id=row["session_id"],
        event_type=row["event_type"],
        level=EventLevel(row["level"]),
        message=row["message"],
        payload_json=row["payload_json"],
        created_at=row["created_at"],
    )


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


def _normalize_title(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("title must be a non-empty string.")
    return normalized


def _require_text(value: str, *, field_name: str) -> str:
    if not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    return value


def _normalize_optional_text(value: str | None) -> str | None:
    if value is None:
        return None

    normalized = value.strip()
    if not normalized:
        return None
    return normalized


def _bool_to_db(value: bool) -> int:
    return 1 if value else 0


def _db_to_bool(value: int) -> bool:
    return bool(value)
