from __future__ import annotations

from pathlib import Path

from unclaw.settings import load_settings
from unclaw.startup import CheckStatus, OllamaStatus, build_startup_report


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


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]
