from __future__ import annotations

import shutil
from pathlib import Path

import yaml

from unclaw.logs.cli import run_logs


def test_run_logs_defaults_to_simple_tail(tmp_path: Path) -> None:
    project_root = _create_temp_project(tmp_path)
    log_path = project_root / "data" / "logs" / "runtime.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(_build_log_lines(30), encoding="utf-8")

    outputs: list[str] = []
    result = run_logs(project_root=project_root, output_func=outputs.append)

    assert result == 0
    assert outputs[0] == "Unclaw logs (simple)"
    assert outputs[1] == f"Runtime log file: {log_path.resolve()}"
    assert any("latest 20 lines" in line for line in outputs)
    assert "line-10" not in outputs
    assert "line-11" in outputs
    assert "line-30" in outputs


def test_run_logs_full_explains_shared_file_mode(tmp_path: Path) -> None:
    project_root = _create_temp_project(tmp_path)
    _write_logging_mode(project_root, mode="simple")
    log_path = project_root / "data" / "logs" / "runtime.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(_build_log_lines(100), encoding="utf-8")

    outputs: list[str] = []
    result = run_logs(
        project_root=project_root,
        mode="full",
        output_func=outputs.append,
    )

    assert result == 0
    assert outputs[0] == "Unclaw logs (full)"
    assert outputs[1] == f"Runtime log file: {log_path.resolve()}"
    assert any("one runtime log file" in line for line in outputs)
    assert any("currently `simple`" in line for line in outputs)
    assert "line-20" not in outputs
    assert "line-21" in outputs
    assert "line-100" in outputs


def _create_temp_project(tmp_path: Path) -> Path:
    source_root = Path(__file__).resolve().parents[1]
    project_root = tmp_path / "project"
    shutil.copytree(source_root / "config", project_root / "config")
    return project_root


def _build_log_lines(count: int) -> str:
    return "".join(f"line-{index}\n" for index in range(1, count + 1))


def _write_logging_mode(project_root: Path, *, mode: str) -> None:
    app_config_path = project_root / "config" / "app.yaml"
    payload = yaml.safe_load(app_config_path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    logging_section = payload["logging"]
    assert isinstance(logging_section, dict)
    logging_section["mode"] = mode
    app_config_path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
