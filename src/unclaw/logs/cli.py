"""User-facing local log inspection commands."""

from __future__ import annotations

import json
import sys
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from unclaw.errors import UnclawError
from unclaw.settings import Settings, load_settings

OutputFunc = Callable[[str], None]
SleepFunc = Callable[[float], None]

_SIMPLE_DISPLAY_LINES = 30
_SIMPLE_SCAN_LINES = 200
_FULL_TAIL_LINES = 80
_FOLLOW_POLL_INTERVAL_SECONDS = 0.5
_SIMPLE_EVENT_TYPES = frozenset(
    {
        "channel.started",
        "session.started",
        "session.selected",
        "model.profile.selected",
        "model.succeeded",
        "thinking.changed",
        "runtime.started",
        "route.selected",
        "assistant.reply.persisted",
        "tool.started",
        "tool.finished",
        "model.failed",
        "telegram.message.received",
    }
)


@dataclass(frozen=True, slots=True)
class LogView:
    """Resolved local log view for one `unclaw logs` invocation."""

    mode: str
    log_path: Path
    log_exists: bool
    file_logging_enabled: bool


@dataclass(frozen=True, slots=True)
class RuntimeLogEvent:
    """Structured event decoded from one runtime log line."""

    created_at: str
    level: str
    event_type: str
    message: str
    session_id: str | None
    payload: dict[str, Any]


def main(project_root: Path | None = None, *, mode: str = "simple") -> int:
    """Print the configured local runtime logs."""

    try:
        return run_logs(project_root=project_root, mode=mode)
    except UnclawError as exc:
        print(f"Failed to read Unclaw logs: {exc}", file=sys.stderr)
        return 1


def run_logs(
    *,
    project_root: Path | None = None,
    mode: str = "simple",
    output_func: OutputFunc = print,
    follow: bool = True,
    sleep_func: SleepFunc = time.sleep,
) -> int:
    """Show a local runtime log stream."""

    settings = load_settings(project_root=project_root)
    log_view = build_log_view(settings, mode=mode)

    for line in format_log_header(log_view, follow=follow):
        output_func(line)

    if not log_view.log_exists:
        output_func("No runtime log file exists yet.")
        output_func("Generate logs with `unclaw start` or `unclaw telegram`, then try again.")
        return 0

    initial_raw_lines = _read_recent_lines(
        log_view.log_path,
        limit=_SIMPLE_SCAN_LINES if log_view.mode == "simple" else _FULL_TAIL_LINES,
    )
    initial_lines = _render_initial_lines(log_view, initial_raw_lines)

    if initial_lines:
        output_func("")
        output_func("Recent activity:")
        for line in initial_lines:
            output_func(line)
    elif initial_raw_lines:
        output_func("")
        output_func(
            "No concise runtime events matched the simple view yet. "
            "Use `unclaw logs full` for the raw JSON stream."
        )
    else:
        output_func("")
        output_func("The runtime log file exists but does not contain any lines yet.")

    if not follow:
        return 0

    try:
        _follow_log_stream(
            log_view=log_view,
            output_func=output_func,
            sleep_func=sleep_func,
        )
    except KeyboardInterrupt:
        output_func("")
        output_func("Stopped log streaming.")

    return 0


def build_log_view(settings: Settings, *, mode: str) -> LogView:
    """Build one local log view from the current project settings."""

    normalized_mode = mode.strip().lower()
    if normalized_mode not in {"simple", "full"}:
        raise ValueError(f"Unsupported log mode: {mode}")

    log_path = settings.paths.log_file_path
    return LogView(
        mode=normalized_mode,
        log_path=log_path,
        log_exists=log_path.is_file(),
        file_logging_enabled=settings.app.logging.file_enabled,
    )


def format_log_header(log_view: LogView, *, follow: bool) -> tuple[str, ...]:
    """Render the live log header."""

    lines = [
        "Unclaw logs",
        f"Mode: {log_view.mode}",
        f"Runtime log: {log_view.log_path}",
        (
            "View: important runtime events"
            if log_view.mode == "simple"
            else "View: raw JSON runtime log stream"
        ),
    ]
    if follow:
        lines.append("Press Ctrl-C to stop.")
    if not log_view.file_logging_enabled:
        lines.append(
            "Note: `logging.file_enabled` is off, so this file may not update until it is enabled again."
        )
    return tuple(lines)


def parse_runtime_log_event(line: str) -> RuntimeLogEvent | None:
    """Parse one JSON runtime log line, if possible."""

    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        return None

    if not isinstance(payload, dict):
        return None

    created_at = payload.get("created_at")
    level = payload.get("level")
    event_type = payload.get("event_type")
    message = payload.get("message")
    session_id = payload.get("session_id")
    event_payload = payload.get("payload", {})

    if not all(isinstance(value, str) and value for value in (created_at, level, event_type, message)):
        return None
    if session_id is not None and not isinstance(session_id, str):
        return None
    if not isinstance(event_payload, dict):
        return None

    return RuntimeLogEvent(
        created_at=created_at,
        level=level,
        event_type=event_type,
        message=message,
        session_id=session_id,
        payload=event_payload,
    )


def render_simple_log_line(raw_line: str) -> str | None:
    """Render one raw runtime log line into the simple live view."""

    line = raw_line.rstrip("\n")
    if not line:
        return None

    event = parse_runtime_log_event(line)
    if event is None:
        return line if _looks_like_warning_or_error(line) else None

    if event.event_type not in _SIMPLE_EVENT_TYPES and event.level not in {"warning", "error"}:
        return None

    timestamp = _compact_timestamp(event.created_at)
    payload = event.payload

    match event.event_type:
        case "channel.started":
            details = [f"channel={payload.get('channel_name', '?')}"]
            model_profile_name = payload.get("model_profile_name")
            if isinstance(model_profile_name, str):
                details.append(f"model={model_profile_name}")
            thinking_enabled = payload.get("thinking_enabled")
            if isinstance(thinking_enabled, bool):
                details.append(f"thinking={'on' if thinking_enabled else 'off'}")
            username = payload.get("username")
            if isinstance(username, str) and username:
                details.append(f"bot=@{username}")
            return f"[{timestamp}] channel started | {' | '.join(details)}"
        case "session.started":
            return (
                f"[{timestamp}] session started | id={event.session_id} | "
                f"title={payload.get('title', '?')}"
            )
        case "session.selected":
            return (
                f"[{timestamp}] session active | id={event.session_id} | "
                f"title={payload.get('title', '?')}"
            )
        case "model.profile.selected":
            details = [
                f"{payload.get('model_profile_name', '?')} -> {payload.get('model_name', '?')}"
            ]
            provider = payload.get("provider")
            if isinstance(provider, str) and provider:
                details.append(f"provider={provider}")
            return f"[{timestamp}] model selected | {' | '.join(details)}"
        case "thinking.changed":
            thinking_enabled = payload.get("thinking_enabled")
            thinking_label = "on" if thinking_enabled is True else "off"
            reason = payload.get("reason")
            details = [
                f"model={payload.get('model_profile_name', '?')}",
                f"thinking={thinking_label}",
            ]
            if isinstance(reason, str) and reason:
                details.append(reason)
            return f"[{timestamp}] thinking changed | {' | '.join(details)}"
        case "runtime.started":
            thinking_enabled = payload.get("thinking_enabled")
            thinking_label = "on" if thinking_enabled is True else "off"
            details = [
                f"session={event.session_id}",
                f"profile={payload.get('model_profile_name', '?')}",
            ]
            provider_model = _format_provider_model(payload)
            if provider_model is not None:
                details.append(f"model={provider_model}")
            details.append(f"thinking={thinking_label}")
            return f"[{timestamp}] assistant turn started | {' | '.join(details)}"
        case "route.selected":
            return (
                f"[{timestamp}] route selected | "
                f"{payload.get('route_kind', '?')} | profile={payload.get('model_profile_name', '?')}"
            )
        case "model.succeeded":
            details = [
                f"model={_format_provider_model(payload) or '?'}",
                f"chars={payload.get('output_length', '?')}",
            ]
            duration_label = _format_duration_value(payload.get("model_duration_ms"))
            if duration_label is not None:
                details.append(f"duration={duration_label}")
            finish_reason = payload.get("finish_reason")
            if isinstance(finish_reason, str) and finish_reason:
                details.append(f"finish={finish_reason}")
            reasoning_label = _format_reasoning_label(payload)
            if reasoning_label is not None:
                details.append(reasoning_label)
            return f"[{timestamp}] model reply | {' | '.join(details)}"
        case "assistant.reply.persisted":
            details = [
                f"session={event.session_id}",
                f"chars={payload.get('output_length', '?')}",
            ]
            duration_label = _format_duration_value(payload.get("turn_duration_ms"))
            if duration_label is not None:
                details.append(f"turn={duration_label}")
            return f"[{timestamp}] assistant reply saved | {' | '.join(details)}"
        case "tool.started":
            details = [str(payload.get("tool_name", "?"))]
            argument_summary = _format_tool_argument_summary(payload.get("arguments"))
            if argument_summary is not None:
                details.append(argument_summary)
            return f"[{timestamp}] tool started | {' | '.join(details)}"
        case "tool.finished":
            success = payload.get("success")
            status = "ok" if success is True else "failed"
            error = payload.get("error")
            details = [str(payload.get("tool_name", "?"))]
            duration_label = _format_duration_value(payload.get("tool_duration_ms"))
            if duration_label is not None:
                details.append(f"duration={duration_label}")
            details.append(f"chars={payload.get('output_length', '?')}")
            if isinstance(error, str) and error:
                details.append(f"error={error}")
            return f"[{timestamp}] tool {status} | {' | '.join(details)}"
        case "model.failed":
            details = [f"profile={payload.get('model_profile_name', '?')}"]
            provider_model = _format_provider_model(payload)
            if provider_model is not None:
                details.append(f"model={provider_model}")
            duration_label = _format_duration_value(payload.get("model_duration_ms"))
            if duration_label is not None:
                details.append(f"duration={duration_label}")
            details.append(str(payload.get("error", event.message)))
            return f"[{timestamp}] model failed | {' | '.join(details)}"
        case "telegram.message.received":
            is_command = payload.get("is_command")
            return (
                f"[{timestamp}] telegram message | session={event.session_id} | "
                f"chat={payload.get('chat_id', '?')} | command={'yes' if is_command else 'no'}"
            )
        case _:
            return f"[{timestamp}] {event.level} {event.event_type} | {event.message}"


def _render_initial_lines(log_view: LogView, raw_lines: tuple[str, ...]) -> tuple[str, ...]:
    if log_view.mode == "full":
        return raw_lines[-_FULL_TAIL_LINES:]

    rendered_lines = tuple(
        rendered_line
        for raw_line in raw_lines
        if (rendered_line := render_simple_log_line(raw_line)) is not None
    )
    return rendered_lines[-_SIMPLE_DISPLAY_LINES:]


def _follow_log_stream(
    *,
    log_view: LogView,
    output_func: OutputFunc,
    sleep_func: SleepFunc,
) -> None:
    with log_view.log_path.open("r", encoding="utf-8") as handle:
        handle.seek(0, 2)
        current_position = handle.tell()

        while True:
            line = handle.readline()
            if line:
                current_position = handle.tell()
                rendered_line = (
                    line.rstrip("\n")
                    if log_view.mode == "full"
                    else render_simple_log_line(line)
                )
                if rendered_line is not None:
                    output_func(rendered_line)
                continue

            try:
                current_size = log_view.log_path.stat().st_size
            except OSError:
                current_size = current_position

            if current_size < current_position:
                handle.seek(0)
                current_position = handle.tell()

            sleep_func(_FOLLOW_POLL_INTERVAL_SECONDS)


def _read_recent_lines(path: Path, *, limit: int) -> tuple[str, ...]:
    with path.open("r", encoding="utf-8") as handle:
        recent_lines = deque((line.rstrip("\n") for line in handle), maxlen=limit)
    return tuple(recent_lines)


def _compact_timestamp(value: str) -> str:
    if "T" not in value:
        return value
    date_part, time_part = value.split("T", maxsplit=1)
    return f"{date_part} {time_part.rstrip('Z')}"


def _format_duration_value(value: object) -> str | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return f"{value} ms"


def _format_provider_model(payload: dict[str, Any]) -> str | None:
    provider = payload.get("provider")
    model_name = payload.get("model_name")
    if isinstance(provider, str) and provider and isinstance(model_name, str) and model_name:
        return f"{provider}/{model_name}"
    if isinstance(model_name, str) and model_name:
        return model_name
    if isinstance(provider, str) and provider:
        return provider
    return None


def _format_reasoning_label(payload: dict[str, Any]) -> str | None:
    reasoning_length = payload.get("reasoning_length")
    if isinstance(reasoning_length, int) and not isinstance(reasoning_length, bool):
        return f"reasoning={reasoning_length} chars"

    reasoning_text = payload.get("reasoning_text")
    if isinstance(reasoning_text, str) and reasoning_text:
        return f"reasoning={len(reasoning_text)} chars"

    return None


def _format_tool_argument_summary(arguments: object) -> str | None:
    if not isinstance(arguments, dict):
        return None

    for key in ("path", "url"):
        value = arguments.get(key)
        if isinstance(value, str) and value:
            return f"{key}={value}"

    return None


def _looks_like_warning_or_error(line: str) -> bool:
    upper_line = line.upper()
    return "WARNING" in upper_line or "ERROR" in upper_line
