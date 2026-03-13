"""Session memory helpers built on top of the session manager."""

from __future__ import annotations

from dataclasses import dataclass

from unclaw.core.session_manager import SessionManager, SessionManagerError
from unclaw.schemas.chat import ChatMessage, MessageRole
from unclaw.schemas.session import SessionRecord

from unclaw.memory.summarizer import summarize_session_messages


@dataclass(frozen=True, slots=True)
class SessionMemoryState:
    """Compact view of the current session memory state."""

    session_id: str
    title: str
    updated_at: str
    message_count: int
    user_message_count: int
    assistant_message_count: int
    summary_text: str
    recent_snippets: tuple[str, ...]


@dataclass(slots=True)
class MemoryManager:
    """Manage small session summaries and recent snippets."""

    session_manager: SessionManager
    recent_snippet_limit: int = 3

    def build_or_refresh_session_summary(self, session_id: str | None = None) -> str:
        """Rebuild the stored summary text for one session."""

        session = self._resolve_session(session_id)
        messages = self.session_manager.list_messages(session.id)
        return self._store_summary(session.id, messages)

    def get_session_summary(self, session_id: str | None = None) -> str:
        """Return the current stored summary, generating it when missing."""

        session = self._resolve_session(session_id)
        summary_text = self.session_manager.session_repository.get_summary_text(session.id)
        if summary_text is not None:
            return summary_text

        messages = self.session_manager.list_messages(session.id)
        return self._store_summary(session.id, messages)

    def get_session_state(
        self,
        session_id: str | None = None,
        *,
        recent_limit: int | None = None,
    ) -> SessionMemoryState:
        """Return a compact view of summary text, counts, and recent snippets."""

        session = self._resolve_session(session_id)
        messages = self.session_manager.list_messages(session.id)

        summary_text = self.session_manager.session_repository.get_summary_text(session.id)
        if summary_text is None:
            summary_text = self._store_summary(session.id, messages)

        return SessionMemoryState(
            session_id=session.id,
            title=session.title,
            updated_at=session.updated_at,
            message_count=len(messages),
            user_message_count=sum(
                1 for message in messages if message.role == MessageRole.USER
            ),
            assistant_message_count=sum(
                1 for message in messages if message.role == MessageRole.ASSISTANT
            ),
            summary_text=summary_text,
            recent_snippets=self.list_recent_snippets(
                session.id,
                limit=(
                    self.recent_snippet_limit
                    if recent_limit is None
                    else recent_limit
                ),
                messages=messages,
            ),
        )

    def list_recent_snippets(
        self,
        session_id: str | None = None,
        *,
        limit: int | None = None,
        messages: list[ChatMessage] | None = None,
    ) -> tuple[str, ...]:
        """Return short snippets from the latest messages in a session."""

        session = self._resolve_session(session_id)
        resolved_limit = self.recent_snippet_limit if limit is None else max(0, limit)
        if resolved_limit == 0:
            return ()

        ordered_messages = messages
        if ordered_messages is None:
            ordered_messages = self.session_manager.list_messages(session.id)

        snippets: list[str] = []
        for message in reversed(ordered_messages):
            snippet = self._format_message_snippet(message)
            if snippet is None:
                continue

            snippets.append(snippet)
            if len(snippets) >= resolved_limit:
                break

        snippets.reverse()
        return tuple(snippets)

    def _resolve_session(self, session_id: str | None) -> SessionRecord:
        if session_id is None:
            return self.session_manager.ensure_current_session()

        session = self.session_manager.load_session(session_id)
        if session is None:
            raise SessionManagerError(f"Session '{session_id}' was not found.")
        return session

    def _store_summary(self, session_id: str, messages: list[ChatMessage]) -> str:
        summary_text = summarize_session_messages(messages)
        stored_summary = self.session_manager.session_repository.update_summary_text(
            session_id,
            summary_text,
        )
        if stored_summary is None:
            raise SessionManagerError(f"Session '{session_id}' was not found.")
        return stored_summary

    def _format_message_snippet(self, message: ChatMessage) -> str | None:
        normalized_content = " ".join(message.content.split()).strip()
        if not normalized_content:
            return None

        if len(normalized_content) > 90:
            normalized_content = f"{normalized_content[:87].rstrip(' ,;:.')}..."
        return f"- {message.role.value}: {normalized_content}"
