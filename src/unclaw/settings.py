"""Configuration loading and runtime path resolution."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from unclaw.constants import (
    APP_CONFIG_FILE_NAME,
    APP_NAME,
    CACHE_DIRECTORY_NAME,
    CONFIG_DIRECTORY_NAME,
    DATA_DIRECTORY_NAME,
    DATABASE_FILE_NAME,
    DISPLAY_NAME,
    FILES_DIRECTORY_NAME,
    LOG_FILE_NAME,
    LOGS_DIRECTORY_NAME,
    MODELS_CONFIG_FILE_NAME,
    PROJECT_ROOT_ENV_VAR,
    SESSIONS_DIRECTORY_NAME,
)
from unclaw.errors import ConfigurationError, PathResolutionError


@dataclass(frozen=True, slots=True)
class DirectorySettings:
    data_dir: str
    logs_dir: str
    sessions_dir: str
    cache_dir: str
    files_dir: str
    database_file: str


@dataclass(frozen=True, slots=True)
class LoggingSettings:
    level: str
    mode: str
    console_enabled: bool
    file_enabled: bool
    file_name: str
    include_reasoning_text: bool


@dataclass(frozen=True, slots=True)
class ChannelSettings:
    terminal_enabled: bool
    telegram_enabled: bool


@dataclass(frozen=True, slots=True)
class ThinkingSettings:
    default_enabled: bool


@dataclass(frozen=True, slots=True)
class FileToolSecuritySettings:
    allowed_roots: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FetchToolSecuritySettings:
    allow_private_networks: bool


@dataclass(frozen=True, slots=True)
class ToolSecuritySettings:
    files: FileToolSecuritySettings
    fetch: FetchToolSecuritySettings


@dataclass(frozen=True, slots=True)
class SecuritySettings:
    tools: ToolSecuritySettings


@dataclass(frozen=True, slots=True)
class AppSettings:
    name: str
    display_name: str
    environment: str
    directories: DirectorySettings
    logging: LoggingSettings
    channels: ChannelSettings
    default_model_profile: str
    thinking: ThinkingSettings
    security: SecuritySettings


@dataclass(frozen=True, slots=True)
class ModelProfile:
    name: str
    provider: str
    model_name: str
    temperature: float
    thinking_supported: bool
    tool_mode: str


@dataclass(frozen=True, slots=True)
class RuntimePaths:
    project_root: Path
    config_dir: Path
    app_config_path: Path
    models_config_path: Path
    data_dir: Path
    logs_dir: Path
    sessions_dir: Path
    cache_dir: Path
    files_dir: Path
    database_path: Path
    log_file_path: Path

    def runtime_directories(self) -> tuple[Path, ...]:
        # Preserve order while removing duplicate directories.
        return tuple(
            dict.fromkeys(
                (
                    self.data_dir,
                    self.logs_dir,
                    self.sessions_dir,
                    self.cache_dir,
                    self.files_dir,
                    self.database_path.parent,
                    self.log_file_path.parent,
                )
            )
        )


@dataclass(frozen=True, slots=True)
class Settings:
    app: AppSettings
    models: dict[str, ModelProfile]
    paths: RuntimePaths

    @property
    def default_model(self) -> ModelProfile:
        return self.models[self.app.default_model_profile]


def load_settings(project_root: Path | None = None) -> Settings:
    resolved_project_root = resolve_project_root(project_root)
    config_dir = resolved_project_root / CONFIG_DIRECTORY_NAME
    app_config_path = config_dir / APP_CONFIG_FILE_NAME
    models_config_path = config_dir / MODELS_CONFIG_FILE_NAME

    app_payload = _load_yaml_file(app_config_path)
    models_payload = _load_yaml_file(models_config_path)

    app_settings = _build_app_settings(
        app_payload,
        project_root=resolved_project_root,
    )
    model_profiles = _build_model_profiles(models_payload)
    runtime_paths = _build_runtime_paths(
        project_root=resolved_project_root,
        config_dir=config_dir,
        app_config_path=app_config_path,
        models_config_path=models_config_path,
        app_settings=app_settings,
    )

    if app_settings.default_model_profile not in model_profiles:
        raise ConfigurationError(
            "Default model profile "
            f"'{app_settings.default_model_profile}' is not defined in "
            f"{models_config_path}."
        )

    return Settings(app=app_settings, models=model_profiles, paths=runtime_paths)


def resolve_project_root(project_root: Path | None = None) -> Path:
    if project_root is not None:
        return project_root.expanduser().resolve()

    environment_root = os.environ.get(PROJECT_ROOT_ENV_VAR)
    if environment_root:
        return Path(environment_root).expanduser().resolve()

    search_bases = (Path.cwd(), Path(__file__).resolve().parent)
    for base in search_bases:
        for candidate in (base, *base.parents):
            if _looks_like_project_root(candidate):
                return candidate.resolve()

    raise PathResolutionError(
        "Could not resolve the Unclaw project root. "
        f"Run the command from a project checkout or set {PROJECT_ROOT_ENV_VAR}."
    )


def _looks_like_project_root(path: Path) -> bool:
    return (
        (path / CONFIG_DIRECTORY_NAME / APP_CONFIG_FILE_NAME).is_file()
        and (path / CONFIG_DIRECTORY_NAME / MODELS_CONFIG_FILE_NAME).is_file()
    )


def _load_yaml_file(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigurationError(f"Missing configuration file: {path}") from exc
    except OSError as exc:
        raise ConfigurationError(f"Could not read configuration file: {path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Invalid YAML in configuration file: {path}") from exc

    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ConfigurationError(f"Configuration file must contain a mapping: {path}")
    return payload


def _build_app_settings(
    payload: Mapping[str, Any],
    *,
    project_root: Path,
) -> AppSettings:
    del project_root
    app_section = _get_mapping(payload, "app")
    paths_section = _get_mapping(payload, "paths")
    logging_section = _get_mapping(payload, "logging")
    channels_section = _get_mapping(payload, "channels")
    models_section = _get_mapping(payload, "models")
    thinking_section = _get_mapping(payload, "thinking")
    security_section = _get_mapping(payload, "security")
    tool_security_section = _get_mapping(security_section, "tools")
    file_security_section = _get_mapping(tool_security_section, "files")
    fetch_security_section = _get_mapping(tool_security_section, "fetch")

    directories = DirectorySettings(
        data_dir=_get_str(paths_section, "data_dir", DATA_DIRECTORY_NAME),
        logs_dir=_get_str(paths_section, "logs_dir", LOGS_DIRECTORY_NAME),
        sessions_dir=_get_str(paths_section, "sessions_dir", SESSIONS_DIRECTORY_NAME),
        cache_dir=_get_str(paths_section, "cache_dir", CACHE_DIRECTORY_NAME),
        files_dir=_get_str(paths_section, "files_dir", FILES_DIRECTORY_NAME),
        database_file=_get_str(paths_section, "database_file", DATABASE_FILE_NAME),
    )
    logging_settings = LoggingSettings(
        level=_get_str(logging_section, "level", "INFO"),
        mode=_get_choice(logging_section, "mode", "simple", {"simple", "full"}),
        console_enabled=_get_bool(logging_section, "console_enabled", True),
        file_enabled=_get_bool(logging_section, "file_enabled", True),
        file_name=_get_str(logging_section, "file_name", LOG_FILE_NAME),
        include_reasoning_text=_get_bool(
            logging_section,
            "include_reasoning_text",
            False,
        ),
    )
    channel_settings = ChannelSettings(
        terminal_enabled=_get_bool(channels_section, "terminal_enabled", True),
        telegram_enabled=_get_bool(channels_section, "telegram_enabled", False),
    )
    thinking_settings = ThinkingSettings(
        default_enabled=_get_bool(thinking_section, "default_enabled", False),
    )
    security_settings = SecuritySettings(
        tools=ToolSecuritySettings(
            files=FileToolSecuritySettings(
                allowed_roots=_get_str_list(
                    file_security_section,
                    "allowed_roots",
                    default=(".",),
                ),
            ),
            fetch=FetchToolSecuritySettings(
                allow_private_networks=_get_bool(
                    fetch_security_section,
                    "allow_private_networks",
                    False,
                )
            ),
        )
    )

    return AppSettings(
        name=_get_str(app_section, "name", APP_NAME),
        display_name=_get_str(app_section, "display_name", DISPLAY_NAME),
        environment=_get_str(app_section, "environment", "development"),
        directories=directories,
        logging=logging_settings,
        channels=channel_settings,
        default_model_profile=_get_str(models_section, "default_profile"),
        thinking=thinking_settings,
        security=security_settings,
    )


def _build_model_profiles(payload: Mapping[str, Any]) -> dict[str, ModelProfile]:
    profiles_section = _get_mapping(payload, "profiles")
    profiles: dict[str, ModelProfile] = {}

    for profile_name, raw_profile in profiles_section.items():
        if not isinstance(profile_name, str):
            raise ConfigurationError("Model profile names must be strings.")
        if not isinstance(raw_profile, Mapping):
            raise ConfigurationError(
                f"Model profile '{profile_name}' must contain a mapping."
            )

        profiles[profile_name] = ModelProfile(
            name=profile_name,
            provider=_get_str(raw_profile, "provider"),
            model_name=_get_str(raw_profile, "model_name"),
            temperature=_get_float(raw_profile, "temperature"),
            thinking_supported=_get_bool(raw_profile, "thinking_supported"),
            tool_mode=_get_str(raw_profile, "tool_mode"),
        )

    if not profiles:
        raise ConfigurationError("At least one model profile must be defined.")

    return profiles


def _build_runtime_paths(
    *,
    project_root: Path,
    config_dir: Path,
    app_config_path: Path,
    models_config_path: Path,
    app_settings: AppSettings,
) -> RuntimePaths:
    data_dir = _resolve_path(project_root, app_settings.directories.data_dir)
    logs_dir = _resolve_path(data_dir, app_settings.directories.logs_dir)
    sessions_dir = _resolve_path(data_dir, app_settings.directories.sessions_dir)
    cache_dir = _resolve_path(data_dir, app_settings.directories.cache_dir)
    files_dir = _resolve_path(data_dir, app_settings.directories.files_dir)
    database_path = _resolve_path(data_dir, app_settings.directories.database_file)
    log_file_path = _resolve_path(logs_dir, app_settings.logging.file_name)

    return RuntimePaths(
        project_root=project_root,
        config_dir=config_dir,
        app_config_path=app_config_path,
        models_config_path=models_config_path,
        data_dir=data_dir,
        logs_dir=logs_dir,
        sessions_dir=sessions_dir,
        cache_dir=cache_dir,
        files_dir=files_dir,
        database_path=database_path,
        log_file_path=log_file_path,
    )


def _resolve_path(base_path: Path, raw_value: str) -> Path:
    path = Path(raw_value).expanduser()
    if not path.is_absolute():
        path = base_path / path
    return path.resolve()


def _get_mapping(source: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = source.get(key, {})
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"Configuration key '{key}' must contain a mapping.")
    return value


def _get_str(
    source: Mapping[str, Any],
    key: str,
    default: str | None = None,
) -> str:
    value = source.get(key, default)
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"Configuration key '{key}' must be a non-empty string.")
    return value


def _get_bool(
    source: Mapping[str, Any],
    key: str,
    default: bool | None = None,
) -> bool:
    value = source.get(key, default)
    if not isinstance(value, bool):
        raise ConfigurationError(f"Configuration key '{key}' must be a boolean.")
    return value


def _get_float(source: Mapping[str, Any], key: str) -> float:
    value = source.get(key)
    if not isinstance(value, (int, float)):
        raise ConfigurationError(f"Configuration key '{key}' must be numeric.")
    return float(value)


def _get_choice(
    source: Mapping[str, Any],
    key: str,
    default: str,
    allowed_values: set[str],
) -> str:
    value = _get_str(source, key, default)
    normalized_value = value.strip().lower()
    if normalized_value not in allowed_values:
        allowed = ", ".join(sorted(allowed_values))
        raise ConfigurationError(
            f"Configuration key '{key}' must be one of: {allowed}."
        )
    return normalized_value


def _get_str_list(
    source: Mapping[str, Any],
    key: str,
    *,
    default: tuple[str, ...] = (),
) -> tuple[str, ...]:
    value = source.get(key)
    if value is None:
        return default
    if not isinstance(value, list):
        raise ConfigurationError(f"Configuration key '{key}' must be a list.")

    values: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ConfigurationError(
                f"Configuration key '{key}' must contain non-empty strings."
            )
        values.append(item.strip())
    return tuple(values)
