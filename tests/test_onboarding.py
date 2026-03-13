from __future__ import annotations

import stat
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import unclaw.onboarding as onboarding
from unclaw.channels.telegram_bot import load_telegram_config
from unclaw.errors import ConfigurationError
from unclaw.local_secrets import LocalSecrets, write_local_secrets
from unclaw.onboarding import (
    InteractivePromptUI,
    MenuOption,
    ModelProfileDraft,
    recommended_model_profiles,
    run_onboarding,
)
from unclaw.settings import load_settings
from unclaw.startup import OllamaStatus

EXAMPLE_TELEGRAM_TOKEN = "123456789:AAExampleTelegramBotTokenValue"


def test_recommended_onboarding_writes_terminal_and_telegram_preset(
    monkeypatch,
    tmp_path: Path,
) -> None:
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

    responses = iter(["", "", "2", EXAMPLE_TELEGRAM_TOKEN, ""])
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
    secrets_payload = _read_yaml(project_root / "config" / "secrets.yaml")

    assert app_payload["logging"]["mode"] == "simple"
    assert app_payload["logging"]["include_reasoning_text"] is False
    assert app_payload["channels"] == {
        "terminal_enabled": True,
        "telegram_enabled": True,
    }
    assert models_payload["profiles"]["fast"]["model_name"] == "llama3.2:3b"
    assert models_payload["profiles"]["fast"]["thinking_supported"] is False
    assert models_payload["profiles"]["main"]["model_name"] == "qwen3.5:4b"
    assert models_payload["profiles"]["deep"]["model_name"] == "qwen3.5:9b"
    assert models_payload["profiles"]["codex"]["model_name"] == "qwen2.5-coder:7b"
    assert telegram_payload["bot_token_env_var"] == "TELEGRAM_BOT_TOKEN"
    assert secrets_payload["telegram"]["bot_token"] == EXAMPLE_TELEGRAM_TOKEN
    assert any("BotFather" in line for line in outputs)
    assert any("config/secrets.yaml" in line for line in outputs)
    assert any("visible while you type" in line for line in outputs)
    assert any("Secure by default" in line for line in outputs)
    assert any("deny-by-default" in line for line in outputs)
    assert any("Ollama is not installed yet." in line for line in outputs)
    assert any("stored locally" in line for line in outputs)
    assert any("unclaw telegram" in line for line in outputs)
    assert any("allow-latest" in line for line in outputs)
    assert stat.S_IMODE((project_root / "config" / "secrets.yaml").stat().st_mode) == 0o600


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
            model_names=("llama3.2:3b", "phi4-mini:3.8b", "qwen3.5:4b"),
            error_message=None,
        ),
    )

    responses = iter(
        [
            "2",
            "",
            "1",
            "3",
            "2",
            "",
            "",
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

    app_payload = _read_yaml(project_root / "config" / "app.yaml")
    models_payload = _read_yaml(project_root / "config" / "models.yaml")
    assert app_payload["channels"] == {
        "terminal_enabled": True,
        "telegram_enabled": False,
    }
    assert models_payload["profiles"]["fast"]["model_name"] == "phi4-mini:3.8b"
    assert models_payload["profiles"]["fast"]["thinking_supported"] is False
    assert models_payload["profiles"]["main"]["model_name"] == "qwen3.5:4b"
    assert models_payload["profiles"]["deep"]["model_name"] == "qwen3.5:9b"
    assert models_payload["profiles"]["codex"]["model_name"] == "devstral:latest"


def test_interactive_select_uses_value_for_initial_choice(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeChoice:
        def __init__(
            self,
            *,
            title: str,
            value: str,
            description: str | None = None,
        ) -> None:
            self.title = title
            self.value = value
            self.description = description

    def fake_ask_select_question(prompt: str, **kwargs: object) -> str:
        captured["prompt"] = prompt
        captured.update(kwargs)
        return "advanced"

    monkeypatch.setattr(onboarding, "questionary", SimpleNamespace(Choice=FakeChoice))
    monkeypatch.setattr(onboarding, "_ask_select_question", fake_ask_select_question)

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
    assert captured["initial_choice"] == "advanced"
    choices = captured["choices"]
    assert isinstance(choices, list)
    assert choices[0].description == "Guided defaults."


def test_onboarding_can_keep_existing_local_telegram_token(
    monkeypatch,
    tmp_path: Path,
) -> None:
    project_root = _create_temp_project(tmp_path)
    settings = load_settings(project_root=project_root)
    (project_root / "config" / "secrets.yaml").write_text(
        yaml.safe_dump(
            {"telegram": {"bot_token": EXAMPLE_TELEGRAM_TOKEN}},
            sort_keys=False,
        ),
        encoding="utf-8",
    )

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

    responses = iter(["", "", "2", "", ""])
    prompts: list[str] = []

    result = run_onboarding(
        settings,
        input_func=lambda prompt: prompts.append(prompt) or next(responses),
        output_func=lambda _message: None,
    )

    secrets_payload = _read_yaml(project_root / "config" / "secrets.yaml")

    assert result == 0
    assert secrets_payload["telegram"]["bot_token"] == EXAMPLE_TELEGRAM_TOKEN
    assert any(
        "Keep the Telegram bot token already stored" in prompt for prompt in prompts
    )


def test_channel_preset_writes_telegram_only_to_app_config(
    monkeypatch,
    tmp_path: Path,
) -> None:
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

    responses = iter(["", "", "3", EXAMPLE_TELEGRAM_TOKEN, ""])

    result = run_onboarding(
        settings,
        input_func=lambda _prompt: next(responses),
        output_func=lambda _message: None,
    )

    assert result == 0

    app_payload = _read_yaml(project_root / "config" / "app.yaml")
    secrets_payload = _read_yaml(project_root / "config" / "secrets.yaml")
    assert app_payload["channels"] == {
        "terminal_enabled": False,
        "telegram_enabled": True,
    }
    assert secrets_payload["telegram"]["bot_token"] == EXAMPLE_TELEGRAM_TOKEN


def test_onboarding_creates_backups_before_overwriting_existing_files(
    monkeypatch,
    tmp_path: Path,
) -> None:
    project_root = _create_temp_project(tmp_path)
    settings = load_settings(project_root=project_root)
    secrets_path = project_root / "config" / "secrets.yaml"
    secrets_path.write_text(
        yaml.safe_dump(
            {"telegram": {"bot_token": "123456789:AAOldTelegramBotTokenValue"}},
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    original_contents = {
        "app": (project_root / "config" / "app.yaml").read_text(encoding="utf-8"),
        "models": (project_root / "config" / "models.yaml").read_text(
            encoding="utf-8"
        ),
        "telegram": (project_root / "config" / "telegram.yaml").read_text(
            encoding="utf-8"
        ),
        "secrets": secrets_path.read_text(encoding="utf-8"),
    }

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

    responses = iter(["", "", "2", "", ""])
    result = run_onboarding(
        settings,
        input_func=lambda _prompt: next(responses),
        output_func=lambda _message: None,
    )

    assert result == 0
    assert (project_root / "config" / "app.yaml.bak").read_text(
        encoding="utf-8"
    ) == original_contents["app"]
    assert (project_root / "config" / "models.yaml.bak").read_text(
        encoding="utf-8"
    ) == original_contents["models"]
    assert (project_root / "config" / "telegram.yaml.bak").read_text(
        encoding="utf-8"
    ) == original_contents["telegram"]
    assert (project_root / "config" / "secrets.yaml.bak").read_text(
        encoding="utf-8"
    ) == original_contents["secrets"]


def test_recommended_model_profiles_match_target_defaults() -> None:
    assert recommended_model_profiles() == {
        "fast": ModelProfileDraft(
            provider="ollama",
            model_name="llama3.2:3b",
            temperature=0.2,
            thinking_supported=False,
            tool_mode="json_plan",
        ),
        "main": ModelProfileDraft(
            provider="ollama",
            model_name="qwen3.5:4b",
            temperature=0.3,
            thinking_supported=True,
            tool_mode="json_plan",
        ),
        "deep": ModelProfileDraft(
            provider="ollama",
            model_name="qwen3.5:9b",
            temperature=0.2,
            thinking_supported=True,
            tool_mode="json_plan",
        ),
        "codex": ModelProfileDraft(
            provider="ollama",
            model_name="qwen2.5-coder:7b",
            temperature=0.1,
            thinking_supported=True,
            tool_mode="json_plan",
        ),
    }


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


def test_write_local_secrets_rewrites_permissions_to_owner_only(
    tmp_path: Path,
) -> None:
    project_root = _create_temp_project(tmp_path)
    settings = load_settings(project_root=project_root)
    secrets_path = project_root / "config" / "secrets.yaml"
    secrets_path.write_text("telegram:\n  bot_token: stale\n", encoding="utf-8")
    secrets_path.chmod(0o644)

    write_local_secrets(
        settings,
        LocalSecrets(telegram_bot_token=EXAMPLE_TELEGRAM_TOKEN),
    )

    assert stat.S_IMODE(secrets_path.stat().st_mode) == 0o600


def _create_temp_project(tmp_path: Path) -> Path:
    source_root = Path(__file__).resolve().parents[1]
    project_root = tmp_path / "project"
    shutil.copytree(source_root / "config", project_root / "config")
    telegram_config_path = project_root / "config" / "telegram.yaml"
    telegram_payload = yaml.safe_load(telegram_config_path.read_text(encoding="utf-8"))
    assert isinstance(telegram_payload, dict)
    telegram_payload["allowed_chat_ids"] = []
    telegram_config_path.write_text(
        yaml.safe_dump(telegram_payload, sort_keys=False),
        encoding="utf-8",
    )
    secrets_path = project_root / "config" / "secrets.yaml"
    if secrets_path.exists():
        secrets_path.unlink()
    return project_root


def _read_yaml(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    assert isinstance(payload, dict)
    return payload
