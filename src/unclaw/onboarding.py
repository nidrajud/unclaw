"""Interactive onboarding for first-time and advanced local setup."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import yaml

from unclaw.bootstrap import bootstrap
from unclaw.channels.telegram_bot import TelegramConfig, load_telegram_config
from unclaw.errors import UnclawError
from unclaw.settings import Settings
from unclaw.startup import (
    build_banner,
    find_missing_model_profiles,
    inspect_ollama,
    ollama_install_guidance,
    OllamaStatus,
    start_ollama_server,
    wait_for_ollama,
)

InputFunc: TypeAlias = Callable[[str], str]
OutputFunc: TypeAlias = Callable[[str], None]

_AVAILABLE_CHANNELS = ("terminal", "telegram")
_PROFILE_ORDER = ("fast", "main", "deep", "codex")
_DEFAULT_TELEGRAM_TOKEN_ENV_VAR = "TELEGRAM_BOT_TOKEN"


@dataclass(frozen=True, slots=True)
class ModelProfileDraft:
    """Editable model profile settings used during onboarding."""

    provider: str
    model_name: str
    temperature: float
    thinking_supported: bool
    tool_mode: str


@dataclass(frozen=True, slots=True)
class OnboardingPlan:
    """One complete onboarding result ready to write to config files."""

    beginner_mode: bool
    automatic_configuration: bool
    logging_mode: str
    enabled_channels: tuple[str, ...]
    default_profile: str
    model_profiles: dict[str, ModelProfileDraft]
    telegram_bot_token_env_var: str
    telegram_allowed_chat_ids: tuple[int, ...]
    telegram_polling_timeout_seconds: int


_RECOMMENDED_PROFILES: dict[str, ModelProfileDraft] = {
    "fast": ModelProfileDraft(
        provider="ollama",
        model_name="llama3.2:3b",
        temperature=0.2,
        thinking_supported=False,
        tool_mode="json_plan",
    ),
    "main": ModelProfileDraft(
        provider="ollama",
        model_name="qwen2.5:7b",
        temperature=0.3,
        thinking_supported=True,
        tool_mode="json_plan",
    ),
    "deep": ModelProfileDraft(
        provider="ollama",
        model_name="qwen2.5:14b",
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


def main(project_root: Path | None = None) -> int:
    """Run interactive onboarding from the command line."""

    try:
        settings = bootstrap(project_root=project_root)
        return run_onboarding(settings)
    except (KeyboardInterrupt, EOFError):
        print("\nOnboarding cancelled.", file=sys.stderr)
        return 1
    except UnclawError as exc:
        print(f"Failed to run Unclaw onboarding: {exc}", file=sys.stderr)
        return 1


def run_onboarding(
    settings: Settings,
    *,
    input_func: InputFunc = input,
    output_func: OutputFunc = print,
) -> int:
    """Run the interactive onboarding flow with injectable I/O for tests."""

    telegram_config = _load_existing_telegram_config(settings)
    ollama_status = inspect_ollama()

    output_func(
        build_banner(
            title="Unclaw onboarding",
            subtitle="Friendly local-first setup for terminal and Telegram.",
            rows=(
                ("project", str(settings.paths.project_root)),
                ("config", str(settings.paths.config_dir)),
                (
                    "ollama",
                    _describe_ollama_status(ollama_status),
                ),
            ),
        )
    )
    output_func("This will update your local config files in place.")
    output_func("")

    beginner_mode = _prompt_yes_no(
        "Do you want beginner-friendly guided setup?",
        default=True,
        input_func=input_func,
        output_func=output_func,
    )
    automatic_configuration = _prompt_yes_no(
        "Should Unclaw use automatic recommended configuration where possible?",
        default=beginner_mode,
        input_func=input_func,
        output_func=output_func,
    )
    logging_mode = _prompt_choice(
        "Choose the logging style",
        options=("simple", "full"),
        default=settings.app.logging.mode,
        input_func=input_func,
        output_func=output_func,
    )
    enabled_channels = _prompt_channels(
        default_channels=_default_channels(settings),
        input_func=input_func,
        output_func=output_func,
    )

    if beginner_mode and automatic_configuration:
        use_recommended_profiles = _prompt_yes_no(
            "Use the recommended model profile set?",
            default=True,
            input_func=input_func,
            output_func=output_func,
        )
        if use_recommended_profiles:
            output_func("Using the recommended local model set.")
            model_profiles = recommended_model_profiles()
        else:
            model_profiles = _prompt_model_profiles(
                settings,
                input_func=input_func,
                output_func=output_func,
            )
    else:
        model_profiles = _prompt_model_profiles(
            settings,
            input_func=input_func,
            output_func=output_func,
        )

    telegram_bot_token_env_var = telegram_config.bot_token_env_var
    if "telegram" in enabled_channels:
        output_func("")
        output_func(
            "Telegram needs a bot token in an environment variable before "
            "`unclaw telegram` can start."
        )
        if beginner_mode:
            output_func(
                f"Use `{telegram_bot_token_env_var}` for the token on this machine."
            )
        else:
            telegram_bot_token_env_var = _prompt_text(
                "Telegram bot token environment variable",
                default=telegram_config.bot_token_env_var,
                input_func=input_func,
                output_func=output_func,
            )

    plan = OnboardingPlan(
        beginner_mode=beginner_mode,
        automatic_configuration=automatic_configuration,
        logging_mode=logging_mode,
        enabled_channels=enabled_channels,
        default_profile="main",
        model_profiles=model_profiles,
        telegram_bot_token_env_var=telegram_bot_token_env_var,
        telegram_allowed_chat_ids=tuple(sorted(telegram_config.allowed_chat_ids)),
        telegram_polling_timeout_seconds=telegram_config.polling_timeout_seconds,
    )

    output_func("")
    _print_plan_summary(plan, output_func=output_func)
    should_write_config = _prompt_yes_no(
        "Write this configuration now?",
        default=True,
        input_func=input_func,
        output_func=output_func,
    )
    if not should_write_config:
        output_func("No files were changed.")
        return 1

    write_onboarding_files(settings, plan)
    configured_settings = bootstrap(project_root=settings.paths.project_root)
    output_func("")
    output_func("Configuration saved:")
    output_func(f"- {configured_settings.paths.app_config_path}")
    output_func(f"- {configured_settings.paths.models_config_path}")
    output_func(f"- {configured_settings.paths.config_dir / 'telegram.yaml'}")

    ollama_status = _post_configure_ollama(
        configured_settings,
        plan,
        input_func=input_func,
        output_func=output_func,
    )

    output_func("")
    output_func("Next steps:")
    output_func("- Start the terminal runtime with `unclaw start`.")
    if "telegram" in enabled_channels:
        output_func(
            f"- Export `{plan.telegram_bot_token_env_var}` and run `unclaw telegram`."
        )
    else:
        output_func("- Enable Telegram later by rerunning `unclaw onboard`.")

    if ollama_status is not None:
        missing_profiles = find_missing_model_profiles(
            configured_settings,
            installed_model_names=ollama_status.model_names,
            profile_names=tuple(plan.model_profiles),
        )
        if missing_profiles:
            output_func("- Pull the remaining missing models before heavy use.")

    return 0


def recommended_model_profiles() -> dict[str, ModelProfileDraft]:
    """Return a fresh copy of the recommended local model profiles."""

    return {name: draft for name, draft in _RECOMMENDED_PROFILES.items()}


def build_onboarding_file_payloads(
    settings: Settings,
    plan: OnboardingPlan,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    """Build YAML payloads for app, model, and Telegram configuration."""

    app_payload: dict[str, object] = {
        "app": {
            "name": settings.app.name,
            "display_name": settings.app.display_name,
            "environment": settings.app.environment,
        },
        "paths": {
            "data_dir": settings.app.directories.data_dir,
            "logs_dir": settings.app.directories.logs_dir,
            "sessions_dir": settings.app.directories.sessions_dir,
            "cache_dir": settings.app.directories.cache_dir,
            "files_dir": settings.app.directories.files_dir,
            "database_file": settings.app.directories.database_file,
        },
        "logging": {
            "level": "DEBUG" if plan.logging_mode == "full" else "INFO",
            "mode": plan.logging_mode,
            "console_enabled": settings.app.logging.console_enabled,
            "file_enabled": settings.app.logging.file_enabled,
            "file_name": settings.app.logging.file_name,
        },
        "channels": {
            "terminal_enabled": "terminal" in plan.enabled_channels,
            "telegram_enabled": "telegram" in plan.enabled_channels,
        },
        "models": {
            "default_profile": plan.default_profile,
        },
        "thinking": {
            "default_enabled": settings.app.thinking.default_enabled,
        },
    }

    models_payload: dict[str, object] = {"profiles": {}}
    profiles_section = models_payload["profiles"]
    assert isinstance(profiles_section, dict)
    for profile_name in _PROFILE_ORDER:
        draft = plan.model_profiles[profile_name]
        profiles_section[profile_name] = {
            "provider": draft.provider,
            "model_name": draft.model_name,
            "temperature": draft.temperature,
            "thinking_supported": draft.thinking_supported,
            "tool_mode": draft.tool_mode,
        }

    telegram_payload: dict[str, object] = {
        "bot_token_env_var": plan.telegram_bot_token_env_var,
        "polling_timeout_seconds": plan.telegram_polling_timeout_seconds,
        "allowed_chat_ids": list(plan.telegram_allowed_chat_ids),
    }
    return app_payload, models_payload, telegram_payload


def write_onboarding_files(settings: Settings, plan: OnboardingPlan) -> None:
    """Write the selected onboarding configuration to disk."""

    app_payload, models_payload, telegram_payload = build_onboarding_file_payloads(
        settings,
        plan,
    )
    _write_yaml(settings.paths.app_config_path, app_payload)
    _write_yaml(settings.paths.models_config_path, models_payload)
    _write_yaml(settings.paths.config_dir / "telegram.yaml", telegram_payload)


def _post_configure_ollama(
    settings: Settings,
    plan: OnboardingPlan,
    *,
    input_func: InputFunc,
    output_func: OutputFunc,
) -> OllamaStatus | None:
    ollama_status = inspect_ollama()
    if not ollama_status.is_installed:
        output_func("")
        output_func("Ollama is not installed yet.")
        output_func(ollama_install_guidance())
        return None

    if not ollama_status.is_running:
        output_func("")
        should_start_ollama = _prompt_yes_no(
            "Ollama is not running. Start it now with `ollama serve`?",
            default=plan.automatic_configuration,
            input_func=input_func,
            output_func=output_func,
        )
        if should_start_ollama:
            try:
                log_path = settings.paths.logs_dir / "ollama-serve.log"
                start_ollama_server(log_path)
            except OSError as exc:
                output_func(f"Could not start Ollama automatically: {exc}")
                output_func("Start it manually with `ollama serve`.")
                return None

            ollama_status = wait_for_ollama()
            if ollama_status.is_running:
                output_func("Ollama is now running.")
            else:
                output_func("Ollama still does not look reachable.")
                output_func("Start it manually with `ollama serve`, then rerun onboarding.")
                return None
        else:
            output_func("Start Ollama later with `ollama serve`.")
            return None

    missing_profiles = find_missing_model_profiles(
        settings,
        installed_model_names=ollama_status.model_names,
        profile_names=tuple(plan.model_profiles),
    )
    if not missing_profiles:
        output_func("")
        output_func("All selected models are already available in Ollama.")
        return ollama_status

    missing_models = _unique_model_names(missing_profiles)
    output_func("")
    output_func(
        "Missing local models: "
        + ", ".join(
            f"{profile_name}={model_name}"
            for profile_name, model_name in missing_profiles
        )
    )
    should_pull_models = _prompt_yes_no(
        "Pull the missing models now?",
        default=plan.automatic_configuration or plan.beginner_mode,
        input_func=input_func,
        output_func=output_func,
    )
    if not should_pull_models:
        output_func("Pull them later with:")
        for model_name in missing_models:
            output_func(f"- ollama pull {model_name}")
        return ollama_status

    for model_name in missing_models:
        output_func(f"Pulling {model_name}...")
        completed_process = subprocess.run(["ollama", "pull", model_name], check=False)
        if completed_process.returncode == 0:
            output_func(f"Finished pulling {model_name}.")
            continue
        output_func(f"`ollama pull {model_name}` exited with code {completed_process.returncode}.")

    return inspect_ollama()


def _load_existing_telegram_config(settings: Settings) -> TelegramConfig:
    try:
        return load_telegram_config(settings)
    except UnclawError:
        return TelegramConfig(
            bot_token_env_var=_DEFAULT_TELEGRAM_TOKEN_ENV_VAR,
            polling_timeout_seconds=30,
            allowed_chat_ids=frozenset(),
        )


def _default_channels(settings: Settings) -> tuple[str, ...]:
    enabled_channels: list[str] = []
    if settings.app.channels.terminal_enabled:
        enabled_channels.append("terminal")
    if settings.app.channels.telegram_enabled:
        enabled_channels.append("telegram")
    if enabled_channels:
        return tuple(enabled_channels)
    return ("terminal",)


def _prompt_model_profiles(
    settings: Settings,
    *,
    input_func: InputFunc,
    output_func: OutputFunc,
) -> dict[str, ModelProfileDraft]:
    output_func("")
    output_func("Choose the model used for each profile.")
    output_func("Press Enter to keep the suggested value.")

    profiles: dict[str, ModelProfileDraft] = {}
    recommended = recommended_model_profiles()
    for profile_name in _PROFILE_ORDER:
        current_profile = settings.models.get(profile_name)
        default_model_name = (
            current_profile.model_name
            if current_profile is not None
            else recommended[profile_name].model_name
        )
        model_name = _prompt_text(
            f"Model for {profile_name}",
            default=default_model_name,
            input_func=input_func,
            output_func=output_func,
        )

        if current_profile is None:
            template = recommended[profile_name]
        else:
            template = ModelProfileDraft(
                provider=current_profile.provider,
                model_name=model_name,
                temperature=current_profile.temperature,
                thinking_supported=current_profile.thinking_supported,
                tool_mode=current_profile.tool_mode,
            )

        profiles[profile_name] = ModelProfileDraft(
            provider=template.provider,
            model_name=model_name,
            temperature=template.temperature,
            thinking_supported=template.thinking_supported,
            tool_mode=template.tool_mode,
        )

    return profiles


def _prompt_yes_no(
    prompt: str,
    *,
    default: bool,
    input_func: InputFunc,
    output_func: OutputFunc,
) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    while True:
        response = input_func(f"{prompt} {suffix} ").strip().lower()
        if not response:
            return default
        if response in {"y", "yes"}:
            return True
        if response in {"n", "no"}:
            return False
        output_func("Please answer yes or no.")


def _prompt_choice(
    prompt: str,
    *,
    options: tuple[str, ...],
    default: str,
    input_func: InputFunc,
    output_func: OutputFunc,
) -> str:
    normalized_options = tuple(option.lower() for option in options)
    default_value = default.lower()
    while True:
        response = input_func(
            f"{prompt} ({'/'.join(normalized_options)}) [{default_value}] "
        ).strip().lower()
        if not response:
            return default_value
        if response in normalized_options:
            return response
        output_func(f"Please choose one of: {', '.join(normalized_options)}.")


def _prompt_channels(
    *,
    default_channels: tuple[str, ...],
    input_func: InputFunc,
    output_func: OutputFunc,
) -> tuple[str, ...]:
    default_text = ",".join(default_channels)
    while True:
        response = input_func(
            "Enable channels (comma separated: terminal, telegram) "
            f"[{default_text}] "
        ).strip()
        raw_values = response or default_text
        selected_channels = tuple(
            channel
            for channel in _AVAILABLE_CHANNELS
            if channel in {
                value.strip().lower()
                for value in raw_values.split(",")
                if value.strip()
            }
        )
        if selected_channels:
            return selected_channels
        output_func("Choose at least one channel.")


def _prompt_text(
    prompt: str,
    *,
    default: str,
    input_func: InputFunc,
    output_func: OutputFunc,
) -> str:
    while True:
        response = input_func(f"{prompt} [{default}] ").strip()
        if response:
            return response
        if default.strip():
            return default.strip()
        output_func("This value cannot be empty.")


def _print_plan_summary(plan: OnboardingPlan, *, output_func: OutputFunc) -> None:
    output_func("Planned setup:")
    output_func(
        f"- mode: {'beginner' if plan.beginner_mode else 'advanced'} | "
        f"automatic: {'yes' if plan.automatic_configuration else 'no'}"
    )
    output_func(f"- logging: {plan.logging_mode}")
    output_func(f"- channels: {', '.join(plan.enabled_channels)}")
    for profile_name in _PROFILE_ORDER:
        output_func(
            f"- {profile_name}: {plan.model_profiles[profile_name].model_name}"
        )
    if "telegram" in plan.enabled_channels:
        output_func(f"- telegram env var: {plan.telegram_bot_token_env_var}")


def _write_yaml(path: Path, payload: dict[str, object]) -> None:
    rendered = yaml.safe_dump(
        payload,
        sort_keys=False,
        allow_unicode=False,
    )
    path.write_text(rendered, encoding="utf-8")


def _describe_ollama_status(ollama_status: OllamaStatus) -> str:
    if not ollama_status.is_installed:
        return "not installed"
    if not ollama_status.is_running:
        return "installed, not running"
    return f"running with {len(ollama_status.model_names)} model(s)"


def _unique_model_names(missing_profiles: tuple[tuple[str, str], ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(model_name for _profile_name, model_name in missing_profiles))
