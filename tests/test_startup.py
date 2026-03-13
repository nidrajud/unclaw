from __future__ import annotations

import shutil
from pathlib import Path

import yaml

from unclaw.settings import load_settings
from unclaw.startup import CheckStatus, OllamaStatus, build_banner, build_startup_report


def test_startup_report_warns_for_missing_optional_models(monkeypatch) -> None:
    settings = load_settings(project_root=_repo_root())

    monkeypatch.setattr(
        "unclaw.startup.inspect_ollama",
        lambda timeout_seconds=1.5: OllamaStatus(
            cli_path="/usr/bin/ollama",
            is_installed=True,
            is_running=True,
            model_names=(settings.default_model.model_name,),
        ),
    )

    report = build_startup_report(
        settings,
        channel_name="terminal",
        channel_enabled=True,
        required_profile_names=(settings.app.default_model_profile,),
        optional_profile_names=tuple(
            profile_name
            for profile_name in settings.models
            if profile_name != settings.app.default_model_profile
        ),
    )

    assert report.has_errors is False
    assert any(
        check.status is CheckStatus.WARN and check.label == "Extra models"
        for check in report.checks
    )


def test_startup_report_errors_when_required_model_is_missing(monkeypatch) -> None:
    settings = load_settings(project_root=_repo_root())

    monkeypatch.setattr(
        "unclaw.startup.inspect_ollama",
        lambda timeout_seconds=1.5: OllamaStatus(
            cli_path="/usr/bin/ollama",
            is_installed=True,
            is_running=True,
            model_names=(),
        ),
    )

    report = build_startup_report(
        settings,
        channel_name="terminal",
        channel_enabled=True,
        required_profile_names=(settings.app.default_model_profile,),
    )

    assert report.has_errors is True
    assert any(
        check.status is CheckStatus.ERROR and check.label == "Models"
        for check in report.checks
    )


def test_startup_report_accepts_local_telegram_token(monkeypatch, tmp_path: Path) -> None:
    project_root = _create_temp_project(tmp_path)
    settings = load_settings(project_root=project_root)
    (project_root / "config" / "secrets.yaml").write_text(
        yaml.safe_dump(
            {
                "telegram": {
                    "bot_token": "123456789:AAExampleTelegramBotTokenValue",
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "unclaw.startup.inspect_ollama",
        lambda timeout_seconds=1.5: OllamaStatus(
            cli_path="/usr/bin/ollama",
            is_installed=True,
            is_running=False,
            model_names=(),
        ),
    )

    report = build_startup_report(
        settings,
        channel_name="telegram",
        channel_enabled=True,
        required_profile_names=(),
        telegram_token_env_var="TELEGRAM_BOT_TOKEN",
    )

    token_check = next(check for check in report.checks if check.label == "Telegram token")
    assert token_check.status is CheckStatus.OK
    assert "local file" in token_check.detail


def test_startup_report_suggests_local_telegram_allow_commands(monkeypatch) -> None:
    settings = load_settings(project_root=_repo_root())

    monkeypatch.setattr(
        "unclaw.startup.inspect_ollama",
        lambda timeout_seconds=1.5: OllamaStatus(
            cli_path="/usr/bin/ollama",
            is_installed=True,
            is_running=False,
            model_names=(),
        ),
    )

    report = build_startup_report(
        settings,
        channel_name="telegram",
        channel_enabled=True,
        required_profile_names=(),
        telegram_allowed_chat_ids=frozenset(),
    )

    access_check = next(check for check in report.checks if check.label == "Telegram access")
    assert access_check.status is CheckStatus.WARN
    assert access_check.guidance is not None
    assert "allow-latest" in access_check.guidance


def test_build_banner_centers_brand_tagline() -> None:
    banner = build_banner(
        title="Onboarding",
        subtitle="Guided local setup.",
        rows=(("mode", "setup"),),
        use_color=False,
    )

    tagline_line = next(
        line for line in banner.splitlines() if "Local-first AI, no cloud claws" in line
    )
    content = tagline_line[2:-2]
    left_padding = len(content) - len(content.lstrip(" "))
    right_padding = len(content) - len(content.rstrip(" "))

    assert abs(left_padding - right_padding) <= 1


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _create_temp_project(tmp_path: Path) -> Path:
    project_root = tmp_path / "project"
    shutil.copytree(_repo_root() / "config", project_root / "config")
    return project_root
