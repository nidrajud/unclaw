"""Helpers for project-local secret storage and Telegram token validation."""

from __future__ import annotations

import os
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from unclaw.errors import ConfigurationError
from unclaw.settings import Settings

LOCAL_SECRETS_FILE_NAME = "secrets.yaml"
DEFAULT_TELEGRAM_TOKEN_ENV_VAR = "TELEGRAM_BOT_TOKEN"

_ENV_VAR_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_TELEGRAM_BOT_TOKEN_PATTERN = re.compile(r"^\d{6,}:[A-Za-z0-9_-]{20,}$")
_TELEGRAM_TOKEN_IN_TEXT_PATTERN = re.compile(r"\d{6,}:[A-Za-z0-9_-]{20,}")
_LOCAL_SECRETS_MODE = 0o600


@dataclass(frozen=True, slots=True)
class LocalSecrets:
    """Project-local secrets stored outside the tracked app config."""

    telegram_bot_token: str | None = None


@dataclass(frozen=True, slots=True)
class ResolvedTelegramBotToken:
    """One Telegram bot token resolved from local config or the environment."""

    value: str
    source_label: str


def local_secrets_path(settings: Settings) -> Path:
    """Return the project-local secrets file path."""

    return settings.paths.config_dir / LOCAL_SECRETS_FILE_NAME


def is_probable_telegram_bot_token(value: str) -> bool:
    """Heuristically detect pasted Telegram bot tokens."""

    return bool(_TELEGRAM_BOT_TOKEN_PATTERN.fullmatch(value.strip()))


def validate_telegram_bot_token(value: str) -> str:
    """Validate a Telegram bot token value."""

    normalized = value.strip()
    if not normalized:
        raise ConfigurationError("Telegram bot token cannot be empty.")
    if not is_probable_telegram_bot_token(normalized):
        raise ConfigurationError(
            "Telegram bot tokens usually look like 123456789:AA... Paste the full "
            "token from BotFather."
        )
    return normalized


def validate_telegram_token_env_var_name(value: str) -> str:
    """Validate the config field that stores the Telegram token env var name."""

    normalized = value.strip()
    if not normalized:
        raise ConfigurationError(
            "Telegram setting 'bot_token_env_var' must be a non-empty string."
        )
    if is_probable_telegram_bot_token(normalized):
        raise ConfigurationError(
            "Telegram setting 'bot_token_env_var' must contain an environment "
            "variable name such as TELEGRAM_BOT_TOKEN, not a pasted bot token. "
            "Store the real token in config/secrets.yaml or in that environment "
            "variable before running `unclaw telegram`."
        )
    if not _ENV_VAR_NAME_PATTERN.fullmatch(normalized):
        raise ConfigurationError(
            "Telegram setting 'bot_token_env_var' must look like an environment "
            "variable name such as TELEGRAM_BOT_TOKEN."
        )
    return normalized


def mask_telegram_bot_token(value: str) -> str:
    """Mask one Telegram bot token while keeping it recognizable."""

    normalized = value.strip()
    if not normalized:
        return normalized

    bot_id, separator, secret = normalized.partition(":")
    if not separator:
        return "***"

    visible_prefix = secret[:2]
    visible_suffix = secret[-4:] if len(secret) > 4 else secret
    if len(secret) <= 6:
        return f"{bot_id}:{visible_prefix}"
    return f"{bot_id}:{visible_prefix}...{visible_suffix}"


def sanitize_telegram_text(value: str, *, known_token: str | None = None) -> str:
    """Mask Telegram bot tokens embedded in URLs, errors, or free-form text."""

    sanitized = value
    if known_token:
        sanitized = sanitized.replace(
            known_token,
            mask_telegram_bot_token(known_token),
        )

    return _TELEGRAM_TOKEN_IN_TEXT_PATTERN.sub(
        lambda match: mask_telegram_bot_token(match.group(0)),
        sanitized,
    )


def load_local_secrets(settings: Settings) -> LocalSecrets:
    """Load the optional project-local secrets file."""

    path = local_secrets_path(settings)
    payload = _load_optional_yaml_mapping(path)
    telegram_section = payload.get("telegram")
    if telegram_section is None:
        return LocalSecrets()
    if not isinstance(telegram_section, dict):
        raise ConfigurationError(
            f"Local secrets file must contain a 'telegram' mapping: {path}"
        )

    raw_bot_token = telegram_section.get("bot_token")
    if raw_bot_token is None:
        return LocalSecrets()
    if not isinstance(raw_bot_token, str):
        raise ConfigurationError(
            f"Telegram secret 'bot_token' must be a string in {path}."
        )

    try:
        telegram_bot_token = validate_telegram_bot_token(raw_bot_token)
    except ConfigurationError as exc:
        raise ConfigurationError(
            f"Invalid Telegram bot token in {path}: {exc}"
        ) from exc
    return LocalSecrets(telegram_bot_token=telegram_bot_token)


def write_local_secrets(settings: Settings, secrets: LocalSecrets) -> None:
    """Persist project-local secrets when present."""

    if secrets.telegram_bot_token is None:
        return

    path = local_secrets_path(settings)
    try:
        payload = _load_optional_yaml_mapping(path)
    except ConfigurationError:
        payload = {}
    telegram_section = payload.get("telegram")
    if telegram_section is None:
        telegram_payload: dict[str, object] = {}
    elif isinstance(telegram_section, dict):
        telegram_payload = dict(telegram_section)
    else:
        telegram_payload = {}

    telegram_payload["bot_token"] = validate_telegram_bot_token(
        secrets.telegram_bot_token
    )
    payload["telegram"] = telegram_payload
    path.write_text(
        yaml.safe_dump(
            payload,
            sort_keys=False,
            allow_unicode=False,
        ),
        encoding="utf-8",
    )
    _ensure_local_secrets_permissions(path)


def resolve_telegram_bot_token(
    settings: Settings,
    *,
    bot_token_env_var: str,
) -> ResolvedTelegramBotToken | None:
    """Resolve the Telegram bot token from local secrets first, then env."""

    secrets = load_local_secrets(settings)
    if secrets.telegram_bot_token is not None:
        return ResolvedTelegramBotToken(
            value=secrets.telegram_bot_token,
            source_label=f"local file {local_secrets_path(settings)}",
        )

    raw_env_value = os.environ.get(bot_token_env_var)
    if raw_env_value is None or not raw_env_value.strip():
        return None

    try:
        token = validate_telegram_bot_token(raw_env_value)
    except ConfigurationError as exc:
        raise ConfigurationError(
            f"Environment variable {bot_token_env_var} is not a valid Telegram bot "
            f"token: {exc}"
        ) from exc

    return ResolvedTelegramBotToken(
        value=token,
        source_label=f"environment variable {bot_token_env_var}",
    )


def _load_optional_yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except OSError as exc:
        raise ConfigurationError(f"Could not read local secrets file: {path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Invalid YAML in local secrets file: {path}") from exc

    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ConfigurationError(
            f"Local secrets file must contain a mapping: {path}"
        )
    return dict(payload)


def _ensure_local_secrets_permissions(path: Path) -> None:
    try:
        os.chmod(path, _LOCAL_SECRETS_MODE)
    except OSError:
        warnings.warn(
            (
                "Unclaw could not enforce owner-only permissions on the local "
                f"secrets file: {path}"
            ),
            stacklevel=2,
        )
