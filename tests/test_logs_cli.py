from __future__ import annotations

import json
import shutil
from pathlib import Path

from unclaw.logs.cli import render_simple_log_line, run_logs


def test_run_logs_defaults_to_simple_live_view_when_follow_is_disabled(
    tmp_path: Path,
) -> None:
    project_root = _create_temp_project(tmp_path)
    log_path = project_root / "data" / "logs" / "runtime.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "\n".join(
            (
                _runtime_event_line(
                    event_type="model.profile.selected",
                    message="Model profile selected.",
                    payload={
                        "model_profile_name": "fast",
                        "model_name": "llama3.2:3b",
                    },
                ),
                _runtime_event_line(
                    event_type="model.called",
                    message="Model call started.",
                    payload={"model_profile_name": "fast"},
                ),
                _runtime_event_line(
                    event_type="assistant.reply.persisted",
                    message="Assistant reply persisted.",
                    payload={"output_length": 84},
                    session_id="sess-1",
                ),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    outputs: list[str] = []
    result = run_logs(
        project_root=project_root,
        output_func=outputs.append,
        follow=False,
    )

    assert result == 0
    assert outputs[0] == "Unclaw logs"
    assert "Mode: simple" in outputs
    assert f"Runtime log: {log_path.resolve()}" in outputs
    assert "Recent activity:" in outputs
    assert any("model selected | fast -> llama3.2:3b" in line for line in outputs)
    assert any("assistant reply saved | session=sess-1 | chars=84" in line for line in outputs)
    assert not any("model.called" in line for line in outputs)


def test_run_logs_full_shows_raw_lines_when_follow_is_disabled(tmp_path: Path) -> None:
    project_root = _create_temp_project(tmp_path)
    log_path = project_root / "data" / "logs" / "runtime.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "plain raw line\n"
        + _runtime_event_line(
            event_type="runtime.started",
            message="Runtime turn started.",
            payload={"model_profile_name": "main", "thinking_enabled": False},
            session_id="sess-2",
        )
        + "\n",
        encoding="utf-8",
    )

    outputs: list[str] = []
    result = run_logs(
        project_root=project_root,
        mode="full",
        output_func=outputs.append,
        follow=False,
    )

    assert result == 0
    assert outputs[0] == "Unclaw logs"
    assert "Mode: full" in outputs
    assert "View: raw runtime log stream" in outputs
    assert "plain raw line" in outputs
    assert any('"event_type": "runtime.started"' in line for line in outputs)


def test_run_logs_explains_how_to_generate_logs_when_file_is_missing(
    tmp_path: Path,
) -> None:
    project_root = _create_temp_project(tmp_path)

    outputs: list[str] = []
    result = run_logs(
        project_root=project_root,
        output_func=outputs.append,
        follow=False,
    )

    assert result == 0
    assert outputs[0] == "Unclaw logs"
    assert "No runtime log file exists yet." in outputs
    assert (
        "Generate logs with `unclaw start` or `unclaw telegram`, then try again."
        in outputs
    )


def test_render_simple_log_line_filters_low_signal_events() -> None:
    assert (
        render_simple_log_line(
            _runtime_event_line(
                event_type="model.called",
                message="Model call started.",
                payload={"model_profile_name": "main"},
            )
        )
        is None
    )


def test_render_simple_log_line_keeps_raw_error_lines() -> None:
    assert render_simple_log_line("ERROR request failed") == "ERROR request failed"


def _create_temp_project(tmp_path: Path) -> Path:
    source_root = Path(__file__).resolve().parents[1]
    project_root = tmp_path / "project"
    shutil.copytree(source_root / "config", project_root / "config")
    return project_root


def _runtime_event_line(
    *,
    event_type: str,
    message: str,
    payload: dict[str, object],
    session_id: str | None = None,
) -> str:
    return json.dumps(
        {
            "created_at": "2026-03-13T12:00:00Z",
            "event_type": event_type,
            "level": "info",
            "message": message,
            "payload": payload,
            "session_id": session_id,
        },
        sort_keys=True,
    )
