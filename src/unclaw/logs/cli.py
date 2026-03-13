"""User-facing local log inspection commands."""

from __future__ import annotations

import sys
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from unclaw.errors import UnclawError
from unclaw.settings import Settings, load_settings

OutputFunc = Callable[[str], None]

_SIMPLE_TAIL_LINES = 20
_FULL_TAIL_LINES = 80


@dataclass(frozen=True, slots=True)
class LogView:
    """Resolved local log view for one `unclaw logs` invocation."""

    mode: str
    log_path: Path
    log_exists: bool
    lines: tuple[str, ...]
    tail_limit: int
    logging_mode: str
    file_logging_enabled: bool


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
) -> int:
    """Show a local runtime log summary and recent lines."""

    settings = load_settings(project_root=project_root)
    log_view = build_log_view(settings, mode=mode)
    for line in format_log_view(log_view):
        output_func(line)
    return 0


def build_log_view(settings: Settings, *, mode: str) -> LogView:
    """Build one local log view from the current project settings."""

    normalized_mode = mode.strip().lower()
    if normalized_mode not in {"simple", "full"}:
        raise ValueError(f"Unsupported log mode: {mode}")

    tail_limit = _SIMPLE_TAIL_LINES if normalized_mode == "simple" else _FULL_TAIL_LINES
    log_path = settings.paths.log_file_path
    log_exists = log_path.is_file()

    return LogView(
        mode=normalized_mode,
        log_path=log_path,
        log_exists=log_exists,
        lines=_read_recent_lines(log_path, limit=tail_limit) if log_exists else (),
        tail_limit=tail_limit,
        logging_mode=settings.app.logging.mode,
        file_logging_enabled=settings.app.logging.file_enabled,
    )


def format_log_view(log_view: LogView) -> tuple[str, ...]:
    """Render one log view into printable lines."""

    lines = [
        f"Unclaw logs ({log_view.mode})",
        f"Runtime log file: {log_view.log_path}",
    ]

    if not log_view.file_logging_enabled:
        lines.append(
            "File logging is disabled in config/app.yaml, so this file may not update "
            "until `logging.file_enabled` is turned back on."
        )
    elif log_view.mode == "simple":
        lines.append(
            f"Simple view shows the latest {log_view.tail_limit} lines from the "
            "standard local runtime log."
        )
    else:
        lines.append(
            "Unclaw currently uses one runtime log file for both simple and full views."
        )
        lines.append(
            f"Full view shows a longer tail. Detailed event volume depends on "
            f"`logging.mode`, which is currently `{log_view.logging_mode}`."
        )

    if not log_view.log_exists:
        lines.append("No runtime log file exists yet.")
        lines.append("Start `unclaw start` or `unclaw telegram`, then run this again.")
        return tuple(lines)

    if not log_view.lines:
        lines.append("The runtime log file exists but does not contain any lines yet.")
        return tuple(lines)

    lines.append("")
    lines.append("Recent lines:")
    lines.extend(log_view.lines)
    return tuple(lines)


def _read_recent_lines(path: Path, *, limit: int) -> tuple[str, ...]:
    with path.open("r", encoding="utf-8") as handle:
        recent_lines = deque((line.rstrip("\n") for line in handle), maxlen=limit)
    return tuple(recent_lines)
