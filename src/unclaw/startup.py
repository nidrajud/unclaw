"""Startup diagnostics and lightweight terminal presentation."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from time import sleep

from unclaw.llm.base import LLMProviderError
from unclaw.llm.ollama_provider import OllamaProvider
from unclaw.settings import Settings


class CheckStatus(StrEnum):
    """Render-friendly startup check states."""

    OK = "ok"
    WARN = "warn"
    ERROR = "error"
    INFO = "info"


@dataclass(frozen=True, slots=True)
class StartupCheck:
    """One user-facing startup check line."""

    status: CheckStatus
    label: str
    detail: str
    guidance: str | None = None


@dataclass(frozen=True, slots=True)
class StartupReport:
    """Aggregate startup validation results."""

    channel_name: str
    checks: tuple[StartupCheck, ...]

    @property
    def has_errors(self) -> bool:
        return any(check.status is CheckStatus.ERROR for check in self.checks)


@dataclass(frozen=True, slots=True)
class OllamaStatus:
    """Current local Ollama environment state."""

    cli_path: str | None
    is_installed: bool
    is_running: bool
    model_names: tuple[str, ...]
    error_message: str | None = None


def inspect_ollama(*, timeout_seconds: float = 1.5) -> OllamaStatus:
    """Inspect whether the local Ollama runtime looks ready."""

    cli_path = shutil.which("ollama")
    if cli_path is None:
        return OllamaStatus(
            cli_path=None,
            is_installed=False,
            is_running=False,
            model_names=(),
            error_message="The Ollama CLI is not installed.",
        )

    provider = OllamaProvider()
    if not provider.is_available(timeout_seconds=timeout_seconds):
        return OllamaStatus(
            cli_path=cli_path,
            is_installed=True,
            is_running=False,
            model_names=(),
            error_message="Ollama is installed but the local server is not reachable.",
        )

    try:
        model_names = provider.list_models(timeout_seconds=timeout_seconds)
    except LLMProviderError as exc:
        return OllamaStatus(
            cli_path=cli_path,
            is_installed=True,
            is_running=True,
            model_names=(),
            error_message=str(exc),
        )

    return OllamaStatus(
        cli_path=cli_path,
        is_installed=True,
        is_running=True,
        model_names=model_names,
    )


def wait_for_ollama(
    *,
    timeout_seconds: float = 8.0,
    poll_interval_seconds: float = 0.5,
) -> OllamaStatus:
    """Wait briefly for Ollama to become reachable."""

    elapsed_seconds = 0.0
    while elapsed_seconds <= timeout_seconds:
        status = inspect_ollama(timeout_seconds=1.0)
        if status.is_running:
            return status
        sleep(poll_interval_seconds)
        elapsed_seconds += poll_interval_seconds
    return inspect_ollama(timeout_seconds=1.0)


def start_ollama_server(log_path: Path) -> subprocess.Popen[bytes]:
    """Start `ollama serve` in the background and write output to a log file."""

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("ab")
    return subprocess.Popen(
        ["ollama", "serve"],
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def build_startup_report(
    settings: Settings,
    *,
    channel_name: str,
    channel_enabled: bool,
    required_profile_names: tuple[str, ...],
    optional_profile_names: tuple[str, ...] = (),
    telegram_token_env_var: str | None = None,
) -> StartupReport:
    """Build a user-facing startup report for one channel."""

    checks: list[StartupCheck] = []
    checks.append(_build_channel_check(channel_name=channel_name, channel_enabled=channel_enabled))

    ollama_status = inspect_ollama()
    checks.extend(_build_ollama_checks(ollama_status))

    if ollama_status.is_running:
        required_missing = find_missing_model_profiles(
            settings,
            installed_model_names=ollama_status.model_names,
            profile_names=required_profile_names,
        )
        optional_missing = find_missing_model_profiles(
            settings,
            installed_model_names=ollama_status.model_names,
            profile_names=optional_profile_names,
        )

        checks.extend(
            _build_required_model_checks(
                settings,
                required_profile_names=required_profile_names,
                missing_profiles=required_missing,
            )
        )
        checks.extend(_build_optional_model_checks(missing_profiles=optional_missing))
    else:
        checks.append(
            StartupCheck(
                status=CheckStatus.INFO,
                label="Models",
                detail="Model availability will be checked once Ollama is running.",
                guidance="Run `unclaw onboard` for guided setup.",
            )
        )

    if telegram_token_env_var is not None:
        checks.append(_build_telegram_token_check(telegram_token_env_var))

    return StartupReport(channel_name=channel_name, checks=tuple(checks))


def find_missing_model_profiles(
    settings: Settings,
    *,
    installed_model_names: tuple[str, ...],
    profile_names: tuple[str, ...],
) -> tuple[tuple[str, str], ...]:
    """Return configured profile/model pairs that are not installed locally."""

    installed_lookup = set(installed_model_names)
    missing_profiles: list[tuple[str, str]] = []

    for profile_name in dict.fromkeys(profile_names):
        profile = settings.models.get(profile_name)
        if profile is None or profile.provider != OllamaProvider.provider_name:
            continue
        if profile.model_name in installed_lookup:
            continue
        missing_profiles.append((profile.name, profile.model_name))

    return tuple(missing_profiles)


def build_banner(
    *,
    title: str,
    subtitle: str,
    rows: tuple[tuple[str, str], ...],
) -> str:
    """Build a compact text-only startup banner."""

    content_lines = [title, subtitle, *(_format_row(label, value) for label, value in rows)]
    width = max(64, *(len(line) for line in content_lines))
    border = "=" * width
    divider = "-" * width

    rendered_lines = [border, title, subtitle, divider]
    rendered_lines.extend(_format_row(label, value) for label, value in rows)
    rendered_lines.append(border)
    return "\n".join(rendered_lines)


def format_startup_report(report: StartupReport) -> str:
    """Render a startup report for the terminal."""

    lines = ["Startup checks:"]
    for check in report.checks:
        lines.append(f"  [{check.status.value}] {check.label}: {check.detail}")
        if check.guidance:
            lines.append(f"        {check.guidance}")
    return "\n".join(lines)


def ollama_install_guidance() -> str:
    """Return short platform-aware install guidance for Ollama."""

    system_name = platform.system().lower()
    if system_name == "darwin":
        return "Install it with `brew install ollama`."
    if system_name == "linux":
        return "Install it with `curl -fsSL https://ollama.com/install.sh | sh`."
    if system_name == "windows":
        return "Install it from https://ollama.com/download/windows."
    return "Install Ollama, then rerun `unclaw onboard`."


def _build_channel_check(*, channel_name: str, channel_enabled: bool) -> StartupCheck:
    label = channel_name.title()
    if channel_enabled:
        return StartupCheck(
            status=CheckStatus.OK,
            label=label,
            detail=f"{label} channel is enabled.",
        )

    return StartupCheck(
        status=CheckStatus.ERROR,
        label=label,
        detail=f"{label} channel is disabled in config/app.yaml.",
        guidance="Run `unclaw onboard` to enable it.",
    )


def _build_ollama_checks(ollama_status: OllamaStatus) -> tuple[StartupCheck, ...]:
    if not ollama_status.is_installed:
        return (
            StartupCheck(
                status=CheckStatus.ERROR,
                label="Ollama",
                detail="The Ollama CLI was not found on this machine.",
                guidance=ollama_install_guidance(),
            ),
        )

    if not ollama_status.is_running:
        return (
            StartupCheck(
                status=CheckStatus.ERROR,
                label="Ollama",
                detail="Ollama is installed but the local server is not running.",
                guidance="Start it with `ollama serve`, or run `unclaw onboard` for help.",
            ),
        )

    return (
        StartupCheck(
            status=CheckStatus.OK,
            label="Ollama",
            detail=(
                f"Local server is reachable with {len(ollama_status.model_names)} "
                "installed model(s)."
            ),
        ),
    )


def _build_required_model_checks(
    settings: Settings,
    *,
    required_profile_names: tuple[str, ...],
    missing_profiles: tuple[tuple[str, str], ...],
) -> tuple[StartupCheck, ...]:
    if not required_profile_names:
        return ()

    if missing_profiles:
        missing_text = ", ".join(
            f"{profile_name}={model_name}"
            for profile_name, model_name in missing_profiles
        )
        return (
            StartupCheck(
                status=CheckStatus.ERROR,
                label="Models",
                detail=f"Required local model profiles are missing: {missing_text}.",
                guidance="Pull them with `ollama pull <model>`, or run `unclaw onboard`.",
            ),
        )

    configured_models = ", ".join(
        f"{profile_name}={settings.models[profile_name].model_name}"
        for profile_name in dict.fromkeys(required_profile_names)
        if profile_name in settings.models
    )
    return (
        StartupCheck(
            status=CheckStatus.OK,
            label="Models",
            detail=f"Required model profiles are available: {configured_models}.",
        ),
    )


def _build_optional_model_checks(
    *,
    missing_profiles: tuple[tuple[str, str], ...],
) -> tuple[StartupCheck, ...]:
    if not missing_profiles:
        return ()

    missing_text = ", ".join(
        f"{profile_name}={model_name}"
        for profile_name, model_name in missing_profiles
    )
    return (
        StartupCheck(
            status=CheckStatus.WARN,
            label="Extra models",
            detail=f"Some optional configured profiles are missing: {missing_text}.",
            guidance="You can still start Unclaw, then pull them later or rerun onboarding.",
        ),
    )


def _build_telegram_token_check(bot_token_env_var: str) -> StartupCheck:
    token = os.environ.get(bot_token_env_var)
    if token is not None and token.strip():
        return StartupCheck(
            status=CheckStatus.OK,
            label="Telegram token",
            detail=f"{bot_token_env_var} is set.",
        )

    return StartupCheck(
        status=CheckStatus.ERROR,
        label="Telegram token",
        detail=f"{bot_token_env_var} is not set.",
        guidance=f"Export {bot_token_env_var}=<your bot token> before `unclaw telegram`.",
    )


def _format_row(label: str, value: str) -> str:
    return f"{label:<12} {value}"
