"""Minimal runtime tracing built on the in-process event bus."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from unclaw.db.repositories import EventRepository
from unclaw.llm.base import utc_now_iso
from unclaw.logs.event_bus import EventBus
from unclaw.schemas.events import EventLevel


@dataclass(frozen=True, slots=True)
class TraceEvent:
    """One runtime event published through the in-process event bus."""

    session_id: str | None
    event_type: str
    level: EventLevel
    message: str
    payload: dict[str, Any]
    created_at: str


@dataclass(slots=True)
class Tracer:
    """Publish and optionally persist runtime trace events."""

    event_bus: EventBus
    event_repository: EventRepository | None = None
    persist_events: bool = True
    runtime_log_path: Path | None = None

    def trace_channel_started(
        self,
        *,
        channel_name: str,
        session_id: str | None = None,
        model_profile_name: str | None = None,
        thinking_enabled: bool | None = None,
        extra_payload: dict[str, Any] | None = None,
    ) -> None:
        payload: dict[str, Any] = {"channel_name": channel_name}
        if model_profile_name is not None:
            payload["model_profile_name"] = model_profile_name
        if thinking_enabled is not None:
            payload["thinking_enabled"] = thinking_enabled
        if extra_payload:
            payload.update(extra_payload)

        self._emit(
            session_id=session_id,
            event_type="channel.started",
            level=EventLevel.INFO,
            message="Channel started.",
            payload=payload,
        )

    def trace_session_started(
        self,
        *,
        session_id: str,
        title: str,
        source: str,
    ) -> None:
        self._emit(
            session_id=session_id,
            event_type="session.started",
            level=EventLevel.INFO,
            message="Session started.",
            payload={
                "title": title,
                "source": source,
            },
        )

    def trace_session_selected(
        self,
        *,
        session_id: str,
        title: str,
        reason: str,
    ) -> None:
        self._emit(
            session_id=session_id,
            event_type="session.selected",
            level=EventLevel.INFO,
            message="Session selected.",
            payload={
                "title": title,
                "reason": reason,
            },
        )

    def trace_model_profile_selected(
        self,
        *,
        session_id: str | None,
        model_profile_name: str,
        provider: str,
        model_name: str,
        reason: str,
    ) -> None:
        self._emit(
            session_id=session_id,
            event_type="model.profile.selected",
            level=EventLevel.INFO,
            message="Model profile selected.",
            payload={
                "model_profile_name": model_profile_name,
                "provider": provider,
                "model_name": model_name,
                "reason": reason,
            },
        )

    def trace_thinking_changed(
        self,
        *,
        session_id: str | None,
        model_profile_name: str,
        thinking_enabled: bool,
        reason: str,
    ) -> None:
        self._emit(
            session_id=session_id,
            event_type="thinking.changed",
            level=EventLevel.INFO,
            message="Thinking mode changed.",
            payload={
                "model_profile_name": model_profile_name,
                "thinking_enabled": thinking_enabled,
                "reason": reason,
            },
        )

    def trace_runtime_started(
        self,
        *,
        session_id: str,
        model_profile_name: str,
        provider: str | None = None,
        model_name: str | None = None,
        thinking_enabled: bool,
        input_length: int,
    ) -> None:
        payload: dict[str, Any] = {
            "model_profile_name": model_profile_name,
            "thinking_enabled": thinking_enabled,
            "input_length": input_length,
        }
        if provider is not None:
            payload["provider"] = provider
        if model_name is not None:
            payload["model_name"] = model_name

        self._emit(
            session_id=session_id,
            event_type="runtime.started",
            level=EventLevel.INFO,
            message="Runtime turn started.",
            payload=payload,
        )

    def trace_route_selected(
        self,
        *,
        session_id: str,
        route_kind: str,
        model_profile_name: str,
    ) -> None:
        self._emit(
            session_id=session_id,
            event_type="route.selected",
            level=EventLevel.INFO,
            message="Runtime route selected.",
            payload={
                "route_kind": route_kind,
                "model_profile_name": model_profile_name,
            },
        )

    def trace_model_called(
        self,
        *,
        session_id: str,
        provider: str,
        model_profile_name: str,
        model_name: str,
        message_count: int,
    ) -> None:
        self._emit(
            session_id=session_id,
            event_type="model.called",
            level=EventLevel.INFO,
            message="Model call started.",
            payload={
                "provider": provider,
                "model_profile_name": model_profile_name,
                "model_name": model_name,
                "message_count": message_count,
            },
        )

    def trace_model_succeeded(
        self,
        *,
        session_id: str,
        provider: str,
        model_name: str,
        finish_reason: str | None,
        output_length: int,
        model_duration_ms: int | None = None,
        reasoning: str | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "provider": provider,
            "model_name": model_name,
            "finish_reason": finish_reason,
            "output_length": output_length,
        }
        if model_duration_ms is not None:
            payload["model_duration_ms"] = model_duration_ms
        if reasoning is not None and reasoning.strip():
            payload["reasoning_text"] = reasoning
            payload["reasoning_length"] = len(reasoning)

        self._emit(
            session_id=session_id,
            event_type="model.succeeded",
            level=EventLevel.INFO,
            message="Model call finished successfully.",
            payload=payload,
        )

    def trace_assistant_reply_persisted(
        self,
        *,
        session_id: str,
        output_length: int,
        turn_duration_ms: int | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "output_length": output_length,
        }
        if turn_duration_ms is not None:
            payload["turn_duration_ms"] = turn_duration_ms

        self._emit(
            session_id=session_id,
            event_type="assistant.reply.persisted",
            level=EventLevel.INFO,
            message="Assistant reply persisted.",
            payload=payload,
        )

    def trace_tool_started(
        self,
        *,
        session_id: str | None,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> None:
        self._emit(
            session_id=session_id,
            event_type="tool.started",
            level=EventLevel.INFO,
            message="Tool command started.",
            payload={
                "tool_name": tool_name,
                "arguments": arguments,
            },
        )

    def trace_tool_finished(
        self,
        *,
        session_id: str | None,
        tool_name: str,
        success: bool,
        output_length: int,
        error: str | None = None,
        tool_duration_ms: int | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "tool_name": tool_name,
            "success": success,
            "output_length": output_length,
            "error": error,
        }
        if tool_duration_ms is not None:
            payload["tool_duration_ms"] = tool_duration_ms

        self._emit(
            session_id=session_id,
            event_type="tool.finished",
            level=EventLevel.INFO if success else EventLevel.ERROR,
            message=(
                "Tool command finished successfully."
                if success
                else "Tool command failed."
            ),
            payload=payload,
        )

    def trace_telegram_message_received(
        self,
        *,
        session_id: str,
        chat_id: int,
        text_length: int,
        is_command: bool,
    ) -> None:
        self._emit(
            session_id=session_id,
            event_type="telegram.message.received",
            level=EventLevel.INFO,
            message="Telegram message received.",
            payload={
                "chat_id": chat_id,
                "text_length": text_length,
                "is_command": is_command,
            },
        )

    def trace_model_failed(
        self,
        *,
        session_id: str,
        provider: str | None,
        model_profile_name: str,
        model_name: str | None = None,
        model_duration_ms: int | None = None,
        error: str,
    ) -> None:
        payload: dict[str, Any] = {
            "provider": provider,
            "model_profile_name": model_profile_name,
            "error": error,
        }
        if model_name is not None:
            payload["model_name"] = model_name
        if model_duration_ms is not None:
            payload["model_duration_ms"] = model_duration_ms

        self._emit(
            session_id=session_id,
            event_type="model.failed",
            level=EventLevel.ERROR,
            message="Model call failed.",
            payload=payload,
        )

    def _emit(
        self,
        *,
        session_id: str | None,
        event_type: str,
        level: EventLevel,
        message: str,
        payload: dict[str, Any],
    ) -> None:
        created_at = utc_now_iso()
        event = TraceEvent(
            session_id=session_id,
            event_type=event_type,
            level=level,
            message=message,
            payload=payload,
            created_at=created_at,
        )
        self.event_bus.publish(event)
        self._append_runtime_log(event)

        if not self.persist_events or self.event_repository is None:
            return

        self.event_repository.add_event(
            session_id=session_id,
            event_type=event_type,
            level=level,
            message=message,
            payload_json=_encode_payload(payload),
            created_at=created_at,
        )

    def _append_runtime_log(self, event: TraceEvent) -> None:
        if self.runtime_log_path is None:
            return

        try:
            self.runtime_log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.runtime_log_path.open("a", encoding="utf-8") as handle:
                handle.write(_encode_runtime_log_event(event))
                handle.write("\n")
        except OSError:
            return


def _encode_payload(payload: dict[str, Any]) -> str | None:
    if not payload:
        return None
    return json.dumps(payload, sort_keys=True)


def _encode_runtime_log_event(event: TraceEvent) -> str:
    payload: dict[str, Any] = {
        "created_at": event.created_at,
        "level": event.level.value,
        "event_type": event.event_type,
        "message": event.message,
        "session_id": event.session_id,
    }
    if event.payload:
        payload["payload"] = event.payload
    return json.dumps(payload, sort_keys=True)
