"""Startup diagnostics and lightweight terminal presentation."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import textwrap
import unicodedata
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from time import sleep

from unclaw.errors import ConfigurationError
from unclaw.llm.base import LLMProviderError
from unclaw.local_secrets import local_secrets_path, resolve_telegram_bot_token
from unclaw.llm.ollama_provider import OllamaProvider
from unclaw.settings import Settings

_UNICODE_WORDMARK = (
    "██╗   ██╗███╗   ██╗ ██████╗██╗      █████╗ ██╗    ██╗",
    "██║   ██║████╗  ██║██╔════╝██║     ██╔══██╗██║    ██║",
    "██║   ██║██╔██╗ ██║██║     ██║     ███████║██║ █╗ ██║",
    "██║   ██║██║╚██╗██║██║     ██║     ██╔══██║██║███╗██║",
    "╚██████╔╝██║ ╚████║╚██████╗███████╗██║  ██║╚███╔███╔╝",
    " ╚═════╝ ╚═╝  ╚═══╝ ╚═════╝╚══════╝╚═╝  ╚═╝ ╚══╝╚══╝ ",
)
_ASCII_WORDMARK = (
    " _   _            _                 ",
    "| | | |_ __   ___| | __ ___      __",
    "| | | | '_ \\ / __| |/ _` \\ \\ /\\ / /",
    "| |_| | | | | (__| | (_| |\\ V  V / ",
    " \\___/|_| |_|\\___|_|\\__,_| \\_/\\_/  ",
)
_BRAND_TAGLINE = "🦐 Local-first AI, no cloud claws 🦐"
_BRAND_TAGLINE_ASCII = "Local-first AI, no cloud claws"


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

    @property
    def has_warnings(self) -> bool:
        return any(check.status is CheckStatus.WARN for check in self.checks)

    @property
    def summary_status(self) -> CheckStatus:
        if self.has_errors:
            return CheckStatus.ERROR
        if self.has_warnings:
            return CheckStatus.WARN
        if any(check.status is CheckStatus.OK for check in self.checks):
            return CheckStatus.OK
        return CheckStatus.INFO


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
    checks.append(
        _build_channel_check(channel_name=channel_name, channel_enabled=channel_enabled)
    )

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
        checks.append(
            _build_telegram_token_check(
                settings,
                bot_token_env_var=telegram_token_env_var,
            )
        )

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
    use_color: bool | None = None,
) -> str:
    """Build a polished ASCII banner for terminal startup flows."""

    use_color = _should_use_color() if use_color is None else use_color
    wordmark, tagline = _render_branding()
    row_lines = tuple(_format_row(label, value) for label, value in rows)
    width = max(
        72,
        *(_visible_length(line) for line in wordmark),
        _visible_length(tagline),
        _visible_length(title),
        _visible_length(subtitle),
        *(_visible_length(line) for line in row_lines),
    )

    lines = [_frame_border("=", width)]
    lines.extend(
        _frame_line(_style_art(line, use_color), width, align="center")
        for line in wordmark
    )
    lines.append(
        _frame_line(
            _style_text(title, "shrimp", use_color, bold=True),
            width,
            align="center",
        )
    )
    lines.append(
        _frame_line(_style_text(tagline, "dim", use_color), width, align="center")
    )
    lines.append(
        _frame_line(_style_text(subtitle, "dim", use_color), width, align="center")
    )
    lines.append(_frame_border("-", width))
    lines.extend(_frame_line(line, width) for line in row_lines)
    lines.append(_frame_border("=", width))
    return "\n".join(lines)


def format_startup_report(
    report: StartupReport,
    *,
    use_color: bool | None = None,
) -> str:
    """Render a startup report for the terminal."""

    use_color = _should_use_color() if use_color is None else use_color
    terminal_width = max(80, shutil.get_terminal_size((96, 20)).columns)
    heading = (
        "Preflight: ready"
        if report.summary_status is CheckStatus.OK
        else "Preflight: ready with warnings"
        if report.summary_status is CheckStatus.WARN
        else "Preflight: action required"
        if report.summary_status is CheckStatus.ERROR
        else "Preflight"
    )

    lines = [_style_text(heading, "cyan", use_color, bold=True)]
    for check in report.checks:
        prefix = f"  {_status_badge(check.status, use_color)} {check.label:<15} "
        wrap_width = max(24, terminal_width - _visible_length(prefix))
        detail_lines = textwrap.wrap(check.detail, width=wrap_width) or [check.detail]
        lines.append(f"{prefix}{detail_lines[0]}")
        continuation_prefix = " " * _visible_length(prefix)
        for detail_line in detail_lines[1:]:
            lines.append(f"{continuation_prefix}{detail_line}")
        if check.guidance:
            guidance_prefix = " " * 4 + _style_text("Tip:", "dim", use_color) + " "
            guidance_width = max(24, terminal_width - 5)
            guidance_lines = textwrap.wrap(check.guidance, width=guidance_width)
            if guidance_lines:
                lines.append(f"{guidance_prefix}{guidance_lines[0]}")
                padding = " " * 9
                for guidance_line in guidance_lines[1:]:
                    lines.append(f"{padding}{guidance_line}")
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


def _build_telegram_token_check(
    settings: Settings,
    *,
    bot_token_env_var: str,
) -> StartupCheck:
    try:
        resolved_token = resolve_telegram_bot_token(
            settings,
            bot_token_env_var=bot_token_env_var,
        )
    except ConfigurationError as exc:
        return StartupCheck(
            status=CheckStatus.ERROR,
            label="Telegram token",
            detail=str(exc),
            guidance=(
                "Run `unclaw onboard` and paste your Telegram bot token again, or "
                f"use the advanced fallback {bot_token_env_var} environment variable."
            ),
        )

    if resolved_token is not None:
        return StartupCheck(
            status=CheckStatus.OK,
            label="Telegram token",
            detail=f"Available from {resolved_token.source_label}.",
        )

    secrets_path = local_secrets_path(settings)
    return StartupCheck(
        status=CheckStatus.ERROR,
        label="Telegram token",
        detail=(
            f"No Telegram bot token was found in {secrets_path} or "
            f"{bot_token_env_var}."
        ),
        guidance=(
            "Run `unclaw onboard` and paste your bot token. Advanced fallback: "
            f"export {bot_token_env_var}=<your bot token>."
        ),
    )


def _format_row(label: str, value: str) -> str:
    return f"{label.upper():<10} {value}"


def _frame_border(fill: str, width: int) -> str:
    return f"+{fill * (width + 2)}+"


def _frame_line(content: str, width: int, *, align: str = "left") -> str:
    return f"| {_pad_content(content, width, align=align)} |"


def _status_badge(status: CheckStatus, use_color: bool) -> str:
    badge_text = {
        CheckStatus.OK: "[OK  ]",
        CheckStatus.WARN: "[WARN]",
        CheckStatus.ERROR: "[FAIL]",
        CheckStatus.INFO: "[INFO]",
    }[status]
    color = {
        CheckStatus.OK: "green",
        CheckStatus.WARN: "yellow",
        CheckStatus.ERROR: "red",
        CheckStatus.INFO: "blue",
    }[status]
    return _style_text(badge_text, color, use_color, bold=True)


def _style_art(text: str, use_color: bool) -> str:
    return _style_text(text, "shrimp", use_color, bold=True)


def _style_text(text: str, color: str, use_color: bool, *, bold: bool = False) -> str:
    if not use_color:
        return text

    color_code = {
        "blue": "34",
        "cyan": "36",
        "green": "32",
        "yellow": "33",
        "red": "31",
        "dim": "2",
        "shrimp": "38;2;245;185;160",
    }[color]
    if color == "dim":
        codes = [color_code]
    else:
        codes = [color_code]
        if bold:
            codes.insert(0, "1")
    if bold and color == "dim":
        codes.insert(0, "1")
    return f"\033[{';'.join(codes)}m{text}\033[0m"


def _visible_length(text: str) -> int:
    length = 0
    in_escape = False
    for character in text:
        if character == "\033":
            in_escape = True
            continue
        if in_escape:
            if character == "m":
                in_escape = False
            continue
        length += _character_width(character)
    return length


def _character_width(character: str) -> int:
    if not character:
        return 0
    if unicodedata.category(character) in {"Cc", "Cf"}:
        return 0
    if unicodedata.combining(character):
        return 0
    if unicodedata.east_asian_width(character) in {"F", "W"}:
        return 2
    return 1


def _pad_content(content: str, width: int, *, align: str) -> str:
    padding = max(0, width - _visible_length(content))
    if align == "center":
        left_padding = padding // 2
        right_padding = padding - left_padding
        return f"{' ' * left_padding}{content}{' ' * right_padding}"
    return f"{content}{' ' * padding}"


def _render_branding() -> tuple[tuple[str, ...], str]:
    if _stdout_supports_text((*_UNICODE_WORDMARK, _BRAND_TAGLINE)):
        return _UNICODE_WORDMARK, _BRAND_TAGLINE
    return _ASCII_WORDMARK, _BRAND_TAGLINE_ASCII


def _stdout_supports_text(lines: tuple[str, ...]) -> bool:
    encoding = sys.stdout.encoding or "utf-8"
    try:
        for line in lines:
            line.encode(encoding)
    except UnicodeEncodeError:
        return False
    return True


def _should_use_color() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("TERM") == "dumb":
        return False
    return sys.stdout.isatty()
