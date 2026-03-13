from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

from unclaw.channels.telegram_bot import load_telegram_config
from unclaw.errors import ConfigurationError
from unclaw.onboarding import InteractivePromptUI, MenuOption, run_onboarding
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

    responses = iter(["", "", "", "y", "", ""])
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
    assert models_payload["profiles"]["fast"]["model_name"] == "qwen3:1.7b"
    assert models_payload["profiles"]["fast"]["thinking_supported"] is False
    assert models_payload["profiles"]["main"]["model_name"] == "qwen3:4b"
    assert models_payload["profiles"]["deep"]["model_name"] == "qwen3:8b"
    assert models_payload["profiles"]["codex"]["model_name"] == "qwen2.5-coder:7b"
    assert telegram_payload["bot_token_env_var"] == "TELEGRAM_BOT_TOKEN"
    assert any("Ollama is not installed yet." in line for line in outputs)
    assert any("unclaw telegram" in line for line in outputs)


def test_advanced_onboarding_can_choose_installed_and_custom_models(
    monkeypatch,
    tmp_path: Path,
) -> None:
    project_root = _create_temp_project(tmp_path)
    settings = load_settings(project_root=project_root)

    monkeypatch.setattr(
        "unclaw.onboarding.inspect_ollama",
        lambda timeout_seconds=1.5: OllamaStatus(
            cli_path="/usr/bin/ollama",
            is_installed=True,
            is_running=True,
            model_names=("qwen3:1.7b", "phi4-mini:3.8b", "qwen3.5:4b"),
            error_message=None,
        ),
    )

    responses = iter(
        [
            "2",
            "",
            "",
            "",
            "",
            "3",
            "2",
            "",
            "2",
            "4",
            "devstral:latest",
            "",
            "",
        ]
    )

    result = run_onboarding(
        settings,
        input_func=lambda _prompt: next(responses),
        output_func=lambda _message: None,
    )

    assert result == 0

    models_payload = _read_yaml(project_root / "config" / "models.yaml")
    assert models_payload["profiles"]["fast"]["model_name"] == "phi4-mini:3.8b"
    assert models_payload["profiles"]["fast"]["thinking_supported"] is False
    assert models_payload["profiles"]["main"]["model_name"] == "qwen3.5:4b"
    assert models_payload["profiles"]["deep"]["model_name"] == "qwen3:8b"
    assert models_payload["profiles"]["codex"]["model_name"] == "devstral:latest"


def test_interactive_select_uses_value_for_questionary_default(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeChoice:
        def __init__(self, *, title: str, value: str) -> None:
            self.title = title
            self.value = value

    class FakePrompt:
        def ask(self) -> str:
            return "advanced"

    class FakeQuestionary:
        Choice = FakeChoice

        @staticmethod
        def select(prompt: str, **kwargs: object) -> FakePrompt:
            captured["prompt"] = prompt
            captured.update(kwargs)
            return FakePrompt()

    monkeypatch.setattr("unclaw.onboarding.questionary", FakeQuestionary)

    result = InteractivePromptUI(output_func=lambda _message: None).select(
        "How do you want to configure Unclaw?",
        options=(
            MenuOption(
                value="recommended",
                label="Recommended setup",
                description="Guided defaults.",
            ),
            MenuOption(
                value="advanced",
                label="Advanced setup",
                description="Manual choices.",
            ),
        ),
        default="advanced",
    )

    assert result == "advanced"
    assert captured["default"] == "advanced"


def test_load_telegram_config_rejects_pasted_bot_token(tmp_path: Path) -> None:
    project_root = _create_temp_project(tmp_path)
    settings = load_settings(project_root=project_root)
    telegram_config_path = project_root / "config" / "telegram.yaml"
    telegram_config_path.write_text(
        yaml.safe_dump(
            {
                "bot_token_env_var": "123456789:AAExampleTelegramBotTokenValue",
                "polling_timeout_seconds": 30,
                "allowed_chat_ids": [],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="environment variable name"):
        load_telegram_config(settings)


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
