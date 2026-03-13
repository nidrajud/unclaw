from __future__ import annotations

import shutil
from pathlib import Path

import yaml

from unclaw.onboarding import run_onboarding
from unclaw.settings import load_settings
from unclaw.startup import OllamaStatus


def test_onboarding_writes_beginner_friendly_config(monkeypatch, tmp_path: Path) -> None:
    project_root = _create_temp_project(tmp_path)
    settings = load_settings(project_root=project_root)

    monkeypatch.setattr(
        "unclaw.onboarding.inspect_ollama",
        lambda timeout_seconds=1.5: OllamaStatus(
            cli_path=None,
            is_installed=False,
            is_running=False,
            model_names=(),
            error_message="not installed",
        ),
    )

    responses = iter(["", "", "", "terminal,telegram", "", ""])
    outputs: list[str] = []

    result = run_onboarding(
        settings,
        input_func=lambda _prompt: next(responses),
        output_func=outputs.append,
    )

    assert result == 0

    app_payload = _read_yaml(project_root / "config" / "app.yaml")
    models_payload = _read_yaml(project_root / "config" / "models.yaml")
    telegram_payload = _read_yaml(project_root / "config" / "telegram.yaml")

    assert app_payload["logging"]["mode"] == "simple"
    assert app_payload["channels"] == {
        "terminal_enabled": True,
        "telegram_enabled": True,
    }
    assert models_payload["profiles"]["codex"]["model_name"] == "qwen2.5-coder:7b"
    assert telegram_payload["bot_token_env_var"] == "TELEGRAM_BOT_TOKEN"
    assert any("Ollama is not installed yet." in line for line in outputs)
    assert any("unclaw telegram" in line for line in outputs)


def _create_temp_project(tmp_path: Path) -> Path:
    source_root = Path(__file__).resolve().parents[1]
    project_root = tmp_path / "project"
    shutil.copytree(source_root / "config", project_root / "config")
    return project_root


def _read_yaml(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    assert isinstance(payload, dict)
    return payload
