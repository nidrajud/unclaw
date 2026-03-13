"""Minimal runtime tracing built on the in-process event bus."""

from __future__ import annotations

import json
from dataclasses import dataclass
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

    def trace_runtime_started(
        self,
        *,
        session_id: str,
        model_profile_name: str,
        thinking_enabled: bool,
        input_length: int,
    ) -> None:
        self._emit(
            session_id=session_id,
            event_type="runtime.started",
            level=EventLevel.INFO,
            message="Runtime turn started.",
            payload={
                "model_profile_name": model_profile_name,
                "thinking_enabled": thinking_enabled,
                "input_length": input_length,
            },
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
    ) -> None:
        self._emit(
            session_id=session_id,
            event_type="model.succeeded",
            level=EventLevel.INFO,
            message="Model call finished successfully.",
            payload={
                "provider": provider,
                "model_name": model_name,
                "finish_reason": finish_reason,
                "output_length": output_length,
            },
        )

    def trace_model_failed(
        self,
        *,
        session_id: str,
        provider: str | None,
        model_profile_name: str,
        error: str,
    ) -> None:
        self._emit(
            session_id=session_id,
            event_type="model.failed",
            level=EventLevel.ERROR,
            message="Model call failed.",
            payload={
                "provider": provider,
                "model_profile_name": model_profile_name,
                "error": error,
            },
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


def _encode_payload(payload: dict[str, Any]) -> str | None:
    if not payload:
        return None
    return json.dumps(payload, sort_keys=True)

