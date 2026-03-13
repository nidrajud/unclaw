"""Session lifecycle helpers backed by the SQLite repositories."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Self

from unclaw.db.repositories import EventRepository, MessageRepository, SessionRepository
from unclaw.db.sqlite import initialize_schema, open_connection
from unclaw.errors import UnclawError
from unclaw.schemas.chat import ChatMessage, MessageRole
from unclaw.schemas.session import SessionRecord, SessionSummary
from unclaw.settings import Settings


class SessionManagerError(UnclawError):
    """Raised when session lifecycle operations fail."""


@dataclass(slots=True)
class SessionManager:
    """Coordinate session state for one running CLI process."""

    settings: Settings
    connection: sqlite3.Connection
    session_repository: SessionRepository
    message_repository: MessageRepository
    event_repository: EventRepository
    current_session_id: str | None = None

    @classmethod
    def from_settings(cls, settings: Settings) -> Self:
        """Create a session manager bound to the configured SQLite database."""
        connection = open_connection(settings.paths.database_path)
        manager = cls(
            settings=settings,
            connection=connection,
            session_repository=SessionRepository(connection),
            message_repository=MessageRepository(connection),
            event_repository=EventRepository(connection),
        )
        manager.initialize()
        return manager

    def initialize(self) -> None:
        """Ensure the SQLite schema exists and restore the current session."""
        initialize_schema(self.connection)
        self._restore_current_session()

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self.connection.close()

    def create_session(
        self,
        title: str | None = None,
        *,
        make_current: bool = True,
    ) -> SessionRecord:
        """Create a new session and optionally switch to it."""
        record = self.session_repository.create_session(
            title=title or self._default_session_title(),
            is_active=False,
        )
        if not make_current:
            return record
        return self.switch_session(record.id)

    def list_sessions(self, *, limit: int | None = None) -> list[SessionSummary]:
        """Return recent sessions ordered by last update time."""
        sessions = self.session_repository.list_sessions()
        if limit is None:
            return sessions
        if limit < 1:
            return []
        return sessions[:limit]

    def load_session(self, session_id: str) -> SessionRecord | None:
        """Load one session by id."""
        return self.session_repository.get_session(session_id)

    def get_current_session(self) -> SessionRecord | None:
        """Return the current in-memory session, if available."""
        if self.current_session_id is None:
            return None
        return self.load_session(self.current_session_id)

    def ensure_current_session(self) -> SessionRecord:
        """Return a current session, creating or restoring one when needed."""
        current_session = self.get_current_session()
        if current_session is not None:
            return current_session

        existing_sessions = self.list_sessions(limit=1)
        if existing_sessions:
            return self.switch_session(existing_sessions[0].id)

        return self.create_session()

    def switch_session(self, session_id: str) -> SessionRecord:
        """Switch the active session for the current CLI process."""
        session = self.load_session(session_id)
        if session is None:
            raise SessionManagerError(f"Session '{session_id}' was not found.")

        self._sync_active_flags(session_id)
        self.current_session_id = session_id

        switched_session = self.load_session(session_id)
        if switched_session is None:
            raise SessionManagerError(f"Session '{session_id}' could not be reloaded.")
        return switched_session

    def rename_session(self, session_id: str, title: str) -> SessionRecord:
        """Rename an existing session."""
        updated_session = self.session_repository.update_session(
            session_id,
            title=title,
        )
        if updated_session is None:
            raise SessionManagerError(f"Session '{session_id}' was not found.")
        return updated_session

    def add_message(
        self,
        role: MessageRole | str,
        content: str,
        *,
        session_id: str | None = None,
    ) -> ChatMessage:
        """Persist one chat message for a session."""
        resolved_session_id = session_id or self.current_session_id
        if resolved_session_id is None:
            raise SessionManagerError("No active session is available.")

        if self.load_session(resolved_session_id) is None:
            raise SessionManagerError(
                f"Session '{resolved_session_id}' is not available."
            )

        return self.message_repository.add_message(
            session_id=resolved_session_id,
            role=role,
            content=content,
        )

    def list_messages(self, session_id: str | None = None) -> list[ChatMessage]:
        """Return the stored messages for one session."""
        resolved_session_id = session_id or self.current_session_id
        if resolved_session_id is None:
            raise SessionManagerError("No active session is available.")
        return self.message_repository.list_messages(resolved_session_id)

    def _restore_current_session(self) -> None:
        sessions = self.session_repository.list_sessions()
        if not sessions:
            self.current_session_id = None
            return

        active_session = next((session for session in sessions if session.is_active), None)
        selected_session = active_session or sessions[0]
        self.current_session_id = selected_session.id

    def _sync_active_flags(self, active_session_id: str) -> None:
        for session in self.session_repository.list_sessions():
            should_be_active = session.id == active_session_id
            if session.is_active == should_be_active:
                continue

            self.session_repository.update_session(
                session.id,
                updated_at=session.updated_at,
                is_active=should_be_active,
            )

    def _default_session_title(self) -> str:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        return f"Session {timestamp}"

