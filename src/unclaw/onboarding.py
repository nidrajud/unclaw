"""Interactive onboarding for first-time and advanced local setup."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TypeAlias

import yaml

from unclaw.bootstrap import bootstrap
from unclaw.channels.telegram_bot import (
    TelegramConfig,
    load_telegram_config,
)
from unclaw.errors import ConfigurationError, UnclawError
from unclaw.local_secrets import (
    DEFAULT_TELEGRAM_TOKEN_ENV_VAR,
    LocalSecrets,
    load_local_secrets,
    local_secrets_path,
    validate_telegram_bot_token,
    write_local_secrets,
)
from unclaw.settings import ModelProfile, Settings
from unclaw.startup import (
    OllamaStatus,
    build_banner,
    find_missing_model_profiles,
    inspect_ollama,
    ollama_install_guidance,
    start_ollama_server,
    wait_for_ollama,
)

try:
    import questionary
except ImportError:  # pragma: no cover - exercised in environments without the UX dependency.
    questionary = None

InputFunc: TypeAlias = Callable[[str], str]
OutputFunc: TypeAlias = Callable[[str], None]

_PROFILE_ORDER = ("fast", "main", "deep", "codex")

_PROFILE_DESCRIPTIONS = {
    "fast": "Quick replies and lighter local hardware usage.",
    "main": "Best default for most conversations and tool work.",
    "deep": "Stronger reasoning when extra latency is acceptable.",
    "codex": "Code-heavy work such as repositories, scripts, and fixes.",
}


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
    telegram_bot_token: str | None
    telegram_bot_token_env_var: str
    telegram_allowed_chat_ids: tuple[int, ...]
    telegram_polling_timeout_seconds: int


@dataclass(frozen=True, slots=True)
class MenuOption:
    """One selectable onboarding option."""

    value: str
    label: str
    description: str | None = None


class PromptUI(Protocol):
    """Minimal prompt surface shared by interactive and fallback onboarding UIs."""

    interactive: bool

    def section(self, title: str, description: str | None = None) -> None:
        ...

    def info(self, message: str = "") -> None:
        ...

    def confirm(
        self,
        prompt: str,
        *,
        default: bool,
        help_text: str | None = None,
    ) -> bool:
        ...

    def select(
        self,
        prompt: str,
        *,
        options: tuple[MenuOption, ...],
        default: str,
        help_text: str | None = None,
    ) -> str:
        ...

    def checkbox(
        self,
        prompt: str,
        *,
        options: tuple[MenuOption, ...],
        default_values: tuple[str, ...],
        help_text: str | None = None,
    ) -> tuple[str, ...]:
        ...

    def text(
        self,
        prompt: str,
        *,
        default: str,
        help_text: str | None = None,
        instruction: str | None = None,
        validator: Callable[[str], str | None] | None = None,
    ) -> str:
        ...


@dataclass(slots=True)
class FallbackPromptUI:
    """Readable plain-text onboarding prompts used in tests and non-TTY flows."""

    input_func: InputFunc
    output_func: OutputFunc
    interactive: bool = False

    def section(self, title: str, description: str | None = None) -> None:
        self.output_func("")
        self.output_func(title)
        self.output_func("-" * max(36, len(title)))
        if description:
            self.output_func(description)
        self.output_func("")

    def info(self, message: str = "") -> None:
        self.output_func(message)

    def confirm(
        self,
        prompt: str,
        *,
        default: bool,
        help_text: str | None = None,
    ) -> bool:
        if help_text:
            self.output_func(help_text)

        suffix = "[Y/n]" if default else "[y/N]"
        while True:
            response = self.input_func(f"{prompt} {suffix} ").strip().lower()
            if not response:
                return default
            if response in {"y", "yes"}:
                return True
            if response in {"n", "no"}:
                return False
            self.output_func("Please answer yes or no.")

    def select(
        self,
        prompt: str,
        *,
        options: tuple[MenuOption, ...],
        default: str,
        help_text: str | None = None,
    ) -> str:
        default_value = _resolve_select_default(options=options, default=default)
        if help_text:
            self.output_func(help_text)
        self.output_func(prompt)

        default_index = 1
        for index, option in enumerate(options, start=1):
            if option.value == default_value:
                default_index = index
            suffix = f" - {option.description}" if option.description else ""
            self.output_func(f"  {index}. {option.label}{suffix}")

        while True:
            response = self.input_func(f"Select an option [{default_index}] ").strip()
            if not response:
                return default_value
            if response.isdigit():
                choice_index = int(response)
                if 1 <= choice_index <= len(options):
                    return options[choice_index - 1].value

            normalized = response.lower()
            for option in options:
                if normalized == option.value.lower():
                    return option.value
            self.output_func("Choose one of the listed options.")

    def checkbox(
        self,
        prompt: str,
        *,
        options: tuple[MenuOption, ...],
        default_values: tuple[str, ...],
        help_text: str | None = None,
    ) -> tuple[str, ...]:
        while True:
            if help_text:
                self.output_func(help_text)
            self.output_func(prompt)

            selected_values: list[str] = []
            default_lookup = set(default_values)
            for option in options:
                enabled = self.confirm(
                    f"Enable {option.label}?",
                    default=option.value in default_lookup,
                    help_text=option.description,
                )
                if enabled:
                    selected_values.append(option.value)

            ordered_values = tuple(
                option.value for option in options if option.value in selected_values
            )
            if ordered_values:
                return ordered_values
            self.output_func("Choose at least one option.")

    def text(
        self,
        prompt: str,
        *,
        default: str,
        help_text: str | None = None,
        instruction: str | None = None,
        validator: Callable[[str], str | None] | None = None,
    ) -> str:
        if help_text:
            self.output_func(help_text)
        if instruction:
            self.output_func(instruction)

        prompt_suffix = f" [{default}]" if default.strip() else ""
        while True:
            response = self.input_func(f"{prompt}{prompt_suffix} ").strip()
            value = response or default.strip()
            if not value:
                self.output_func("This value cannot be empty.")
                continue

            if validator is not None:
                error_message = validator(value)
                if error_message is not None:
                    self.output_func(error_message)
                    continue
            return value


@dataclass(slots=True)
class InteractivePromptUI:
    """Menu-driven onboarding prompts backed by questionary."""

    output_func: OutputFunc
    interactive: bool = True

    def section(self, title: str, description: str | None = None) -> None:
        self.output_func("")
        self.output_func(title)
        self.output_func("-" * max(36, len(title)))
        if description:
            self.output_func(description)
        self.output_func("")

    def info(self, message: str = "") -> None:
        self.output_func(message)

    def confirm(
        self,
        prompt: str,
        *,
        default: bool,
        help_text: str | None = None,
    ) -> bool:
        if help_text:
            self.output_func(help_text)
        answer = questionary.confirm(
            prompt,
            default=default,
            style=_QUESTIONARY_STYLE,
        ).ask()
        if answer is None:
            raise KeyboardInterrupt
        return answer

    def select(
        self,
        prompt: str,
        *,
        options: tuple[MenuOption, ...],
        default: str,
        help_text: str | None = None,
    ) -> str:
        if help_text:
            self.output_func(help_text)

        default_value = _resolve_select_default(options=options, default=default)
        answer = _ask_select_question(
            prompt,
            choices=[_build_questionary_choice(option) for option in options],
            initial_choice=default_value,
            instruction="Use arrow keys, then press Enter.",
        )
        if answer is None:
            raise KeyboardInterrupt
        return answer

    def checkbox(
        self,
        prompt: str,
        *,
        options: tuple[MenuOption, ...],
        default_values: tuple[str, ...],
        help_text: str | None = None,
    ) -> tuple[str, ...]:
        if help_text:
            self.output_func(help_text)

        default_lookup = set(default_values)
        while True:
            answer = questionary.checkbox(
                prompt,
                choices=[
                    questionary.Choice(
                        title=option.label,
                        value=option.value,
                        checked=option.value in default_lookup,
                        description=option.description,
                    )
                    for option in options
                ],
                initial_choice=_resolve_checkbox_initial_choice(
                    options=options,
                    default_values=default_values,
                ),
                instruction="Use arrow keys to move, Space to toggle, Enter to confirm.",
                style=_QUESTIONARY_STYLE,
            ).ask()
            if answer is None:
                raise KeyboardInterrupt

            ordered_values = tuple(
                option.value for option in options if option.value in set(answer)
            )
            if ordered_values:
                return ordered_values
            self.output_func("Choose at least one option.")

    def text(
        self,
        prompt: str,
        *,
        default: str,
        help_text: str | None = None,
        instruction: str | None = None,
        validator: Callable[[str], str | None] | None = None,
    ) -> str:
        while True:
            if help_text:
                self.output_func(help_text)
            answer = questionary.text(
                prompt,
                default=default,
                instruction=instruction or "Press Enter to confirm.",
                style=_QUESTIONARY_STYLE,
            ).ask()
            if answer is None:
                raise KeyboardInterrupt

            value = answer.strip() or default.strip()
            if not value:
                self.output_func("This value cannot be empty.")
                continue
            if validator is not None:
                error_message = validator(value)
                if error_message is not None:
                    self.output_func(error_message)
                    continue
            return value


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

_SETUP_MODE_OPTIONS = (
    MenuOption(
        value="recommended",
        label="Recommended setup",
        description="Start with the recommended local model set and guided defaults.",
    ),
    MenuOption(
        value="advanced",
        label="Advanced setup",
        description="Choose logging, channels, and every model profile manually.",
    ),
)

_LOGGING_OPTIONS = (
    MenuOption(
        value="simple",
        label="Simple logs",
        description="Cleaner day-to-day output with less terminal noise.",
    ),
    MenuOption(
        value="full",
        label="Full logs",
        description="More runtime detail for testing, tuning, and debugging.",
    ),
)

_CHANNEL_PRESET_OPTIONS = (
    MenuOption(
        value="terminal_only",
        label="Terminal only",
        description="Keep Unclaw in the terminal only.",
    ),
    MenuOption(
        value="terminal_and_telegram",
        label="Terminal + Telegram",
        description="Use both the terminal and your Telegram bot.",
    ),
    MenuOption(
        value="telegram_only",
        label="Telegram only",
        description="Run Unclaw only through Telegram.",
    ),
)

if questionary is not None:
    _QUESTIONARY_STYLE = questionary.Style(
        [
            ("qmark", "fg:#f6c7b1 bold"),
            ("question", "fg:#fff4ec bold"),
            ("answer", "fg:#f6c7b1 bold"),
            ("pointer", "fg:#241511 bg:#f6c7b1 bold"),
            ("highlighted", "fg:#241511 bg:#f6c7b1 bold"),
            ("selected", "fg:#241511 bg:#f2b99b bold"),
            ("text", "fg:#dde3e8"),
            ("separator", "fg:#8f7b70"),
            ("instruction", "fg:#c79a85 italic"),
            ("disabled", "fg:#6d6159 italic"),
            ("bottom-toolbar", "noreverse"),
            ("validation-toolbar", "fg:#ffffff bg:#b42318 bold"),
        ]
    )
else:  # pragma: no cover - exercised when questionary is unavailable.
    _QUESTIONARY_STYLE = None


def _build_questionary_choice(option: MenuOption) -> Any:
    return questionary.Choice(
        title=option.label,
        value=option.value,
        description=option.description,
    )


def _resolve_checkbox_initial_choice(
    *,
    options: tuple[MenuOption, ...],
    default_values: tuple[str, ...],
) -> str:
    for value in default_values:
        for option in options:
            if option.value == value:
                return value
    return options[0].value


def _ask_select_question(
    prompt: str,
    *,
    choices: list[Any],
    initial_choice: str,
    instruction: str,
) -> str | None:
    from prompt_toolkit.application import Application
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.keys import Keys
    from questionary.constants import (
        DEFAULT_QUESTION_PREFIX,
        DEFAULT_SELECTED_POINTER,
    )
    from questionary.prompts import common as questionary_common
    from questionary.prompts.common import InquirerControl
    from questionary.question import Question
    from questionary.styles import merge_styles_default

    merged_style = merge_styles_default([_QUESTIONARY_STYLE])
    inquirer_control = InquirerControl(
        choices,
        default=None,
        pointer=DEFAULT_SELECTED_POINTER,
        use_indicator=False,
        use_shortcuts=False,
        show_selected=False,
        show_description=True,
        use_arrow_keys=True,
        initial_choice=initial_choice,
    )

    def get_prompt_tokens() -> list[tuple[str, str]]:
        return [
            ("class:qmark", DEFAULT_QUESTION_PREFIX),
            ("class:question", f" {prompt} "),
            ("class:instruction", instruction),
        ]

    layout = questionary_common.create_inquirer_layout(
        inquirer_control,
        get_prompt_tokens,
    )
    bindings = KeyBindings()

    @bindings.add(Keys.ControlQ, eager=True)
    @bindings.add(Keys.ControlC, eager=True)
    def _abort(event: Any) -> None:
        event.app.exit(exception=KeyboardInterrupt, style="class:aborting")

    def _move_cursor_down(event: Any) -> None:
        inquirer_control.select_next()
        while not inquirer_control.is_selection_valid():
            inquirer_control.select_next()

    def _move_cursor_up(event: Any) -> None:
        inquirer_control.select_previous()
        while not inquirer_control.is_selection_valid():
            inquirer_control.select_previous()

    @bindings.add(Keys.Down, eager=True)
    def _down(event: Any) -> None:
        _move_cursor_down(event)

    @bindings.add(Keys.Up, eager=True)
    def _up(event: Any) -> None:
        _move_cursor_up(event)

    @bindings.add("j", eager=True)
    def _j(event: Any) -> None:
        _move_cursor_down(event)

    @bindings.add("k", eager=True)
    def _k(event: Any) -> None:
        _move_cursor_up(event)

    @bindings.add(Keys.ControlN, eager=True)
    def _ctrl_n(event: Any) -> None:
        _move_cursor_down(event)

    @bindings.add(Keys.ControlP, eager=True)
    def _ctrl_p(event: Any) -> None:
        _move_cursor_up(event)

    @bindings.add(Keys.ControlM, eager=True)
    def _submit(event: Any) -> None:
        inquirer_control.is_answered = True
        event.app.exit(result=inquirer_control.get_pointed_at().value)

    @bindings.add(Keys.Any)
    def _ignore_other_keys(event: Any) -> None:
        del event

    return Question(
        Application(
            layout=layout,
            key_bindings=bindings,
            style=merged_style,
        )
    ).ask()


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

    prompt_ui = _build_prompt_ui(input_func=input_func, output_func=output_func)
    telegram_config, telegram_warning = _load_existing_telegram_config(settings)
    local_secrets, local_secrets_warning = _load_existing_local_secrets(settings)
    ollama_status = inspect_ollama()

    output_func(
        _build_onboarding_banner(settings=settings, ollama_status=ollama_status)
    )
    output_func("Welcome to Unclaw setup.")
    output_func("This guide updates local files in this project only.")
    output_func("Your models, chats, and secrets stay on your machine.")
    if prompt_ui.interactive:
        output_func("Use arrow keys to move, Enter to confirm, and Ctrl-C to cancel.")
    if telegram_warning is not None:
        output_func(telegram_warning)
    if local_secrets_warning is not None:
        output_func(local_secrets_warning)

    prompt_ui.section(
        "🛠 Setup style",
        "Choose how much Unclaw should decide for you during first-time setup.",
    )
    setup_mode = prompt_ui.select(
        "How do you want to configure Unclaw?",
        options=_SETUP_MODE_OPTIONS,
        default="recommended",
    )
    beginner_mode = setup_mode == "recommended"
    automatic_configuration = beginner_mode

    prompt_ui.section(
        "🪵 Logging",
        "Choose how much runtime detail you want during startup and normal use.",
    )
    logging_mode = prompt_ui.select(
        "Which log view should Unclaw use day to day?",
        options=_LOGGING_OPTIONS,
        default="simple" if beginner_mode else settings.app.logging.mode,
    )

    prompt_ui.section(
        "📡 Channels",
        "Choose how people should talk to Unclaw in this project.",
    )
    enabled_channels = _enabled_channels_from_preset(
        _prompt_channel_preset(
            prompt_ui=prompt_ui,
            default_preset=_default_channel_preset(settings),
        )
    )

    prompt_ui.section(
        "🧠 Model profiles",
        "Choose the local models for quick replies, default chat, deeper work, and code tasks.",
    )
    if beginner_mode:
        model_profiles = recommended_model_profiles()
        prompt_ui.info("Using the recommended starter model lineup:")
        for profile_name in _PROFILE_ORDER:
            prompt_ui.info(
                f"- {profile_name}: {model_profiles[profile_name].model_name}"
            )
    else:
        model_profiles = _prompt_model_profiles(
            settings,
            prompt_ui=prompt_ui,
            ollama_status=ollama_status,
        )

    telegram_bot_token = local_secrets.telegram_bot_token
    telegram_bot_token_env_var = telegram_config.bot_token_env_var
    if "telegram" in enabled_channels:
        telegram_bot_token = _prompt_telegram_bot_token(
            existing_token=local_secrets.telegram_bot_token,
            prompt_ui=prompt_ui,
        )

    plan = OnboardingPlan(
        beginner_mode=beginner_mode,
        automatic_configuration=automatic_configuration,
        logging_mode=logging_mode,
        enabled_channels=enabled_channels,
        default_profile="main",
        model_profiles=model_profiles,
        telegram_bot_token=telegram_bot_token,
        telegram_bot_token_env_var=telegram_bot_token_env_var,
        telegram_allowed_chat_ids=tuple(sorted(telegram_config.allowed_chat_ids)),
        telegram_polling_timeout_seconds=telegram_config.polling_timeout_seconds,
    )

    prompt_ui.section(
        "📝 Review",
        "Check the plan before Unclaw writes local configuration files.",
    )
    _print_plan_summary(plan, output_func=output_func)
    should_write_config = prompt_ui.confirm(
        "Write this configuration now?",
        default=True,
    )
    if not should_write_config:
        output_func("No files were changed.")
        return 1

    write_onboarding_files(settings, plan)
    configured_settings = bootstrap(project_root=settings.paths.project_root)
    output_func("")
    output_func("Saved local configuration:")
    output_func(f"- {configured_settings.paths.app_config_path}")
    output_func(f"- {configured_settings.paths.models_config_path}")
    output_func(f"- {configured_settings.paths.config_dir / 'telegram.yaml'}")
    if plan.telegram_bot_token is not None:
        output_func(f"- {local_secrets_path(configured_settings)}")

    ollama_status = _post_configure_ollama(
        configured_settings,
        plan,
        prompt_ui=prompt_ui,
    )

    output_func("")
    output_func("What to do next:")
    output_func("- Start the terminal experience with `unclaw start`.")
    if "telegram" in enabled_channels:
        output_func("- Start your Telegram bot with `unclaw telegram`.")
        output_func(
            f"- Your bot token is stored locally in `{local_secrets_path(configured_settings)}`."
        )
        if plan.telegram_allowed_chat_ids:
            count = len(plan.telegram_allowed_chat_ids)
            label = "chat" if count == 1 else "chats"
            output_func(
                f"- Telegram access: {count} authorized {label} already configured in `config/telegram.yaml`."
            )
        else:
            output_func(
                "- Telegram access: secure deny-by-default until you add chat IDs to `config/telegram.yaml`."
            )
            output_func(
                "- Tip: send one test message, then check `unclaw logs simple` for the rejected `chat_id` to allowlist."
            )
        output_func(
            f"- Advanced fallback: export {plan.telegram_bot_token_env_var}=<your bot token>"
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
            output_func("- Pull the remaining missing models before heavier use.")

    return 0


def recommended_model_profiles() -> dict[str, ModelProfileDraft]:
    """Return a fresh copy of the recommended local model profiles."""

    return {
        name: ModelProfileDraft(
            provider=draft.provider,
            model_name=draft.model_name,
            temperature=draft.temperature,
            thinking_supported=draft.thinking_supported,
            tool_mode=draft.tool_mode,
        )
        for name, draft in _RECOMMENDED_PROFILES.items()
    }


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
            "include_reasoning_text": False,
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
    write_local_secrets(
        settings,
        LocalSecrets(telegram_bot_token=plan.telegram_bot_token),
    )


def _post_configure_ollama(
    settings: Settings,
    plan: OnboardingPlan,
    *,
    prompt_ui: PromptUI,
) -> OllamaStatus | None:
    prompt_ui.section(
        "🦙 Local model runtime",
        "Check Ollama now so startup is smoother the next time you launch Unclaw.",
    )
    ollama_status = inspect_ollama()
    if not ollama_status.is_installed:
        prompt_ui.info("Ollama is not installed yet.")
        prompt_ui.info(ollama_install_guidance())
        return None

    if not ollama_status.is_running:
        should_start_ollama = prompt_ui.confirm(
            "Ollama is not running. Start it now with `ollama serve`?",
            default=plan.automatic_configuration,
        )
        if should_start_ollama:
            try:
                log_path = settings.paths.logs_dir / "ollama-serve.log"
                start_ollama_server(log_path)
            except OSError as exc:
                prompt_ui.info(f"Could not start Ollama automatically: {exc}")
                prompt_ui.info("Start it manually with `ollama serve`.")
                return None

            ollama_status = wait_for_ollama()
            if ollama_status.is_running:
                prompt_ui.info("Ollama is now running.")
            else:
                prompt_ui.info("Ollama still does not look reachable.")
                prompt_ui.info(
                    "Start it manually with `ollama serve`, then rerun onboarding."
                )
                return None
        else:
            prompt_ui.info("Start Ollama later with `ollama serve`.")
            return None

    missing_profiles = find_missing_model_profiles(
        settings,
        installed_model_names=ollama_status.model_names,
        profile_names=tuple(plan.model_profiles),
    )
    if not missing_profiles:
        prompt_ui.info("All selected models are already available in Ollama.")
        return ollama_status

    missing_models = _unique_model_names(missing_profiles)
    prompt_ui.info(
        "Missing local models: "
        + ", ".join(
            f"{profile_name}={model_name}"
            for profile_name, model_name in missing_profiles
        )
    )
    should_pull_models = prompt_ui.confirm(
        "Pull the missing models now?",
        default=plan.automatic_configuration or plan.beginner_mode,
    )
    if not should_pull_models:
        prompt_ui.info("Pull them later with:")
        for model_name in missing_models:
            prompt_ui.info(f"- ollama pull {model_name}")
        return ollama_status

    for model_name in missing_models:
        prompt_ui.info(f"Pulling {model_name}...")
        completed_process = subprocess.run(["ollama", "pull", model_name], check=False)
        if completed_process.returncode == 0:
            prompt_ui.info(f"Finished pulling {model_name}.")
            continue
        prompt_ui.info(
            f"`ollama pull {model_name}` exited with code {completed_process.returncode}."
        )

    return inspect_ollama()


def _load_existing_telegram_config(
    settings: Settings,
) -> tuple[TelegramConfig, str | None]:
    try:
        return load_telegram_config(settings), None
    except UnclawError as exc:
        warning = (
            "Telegram config note: "
            f"{exc} Using {DEFAULT_TELEGRAM_TOKEN_ENV_VAR} as the advanced fallback "
            "environment variable."
        )
        return (
            TelegramConfig(
                bot_token_env_var=DEFAULT_TELEGRAM_TOKEN_ENV_VAR,
                polling_timeout_seconds=30,
                allowed_chat_ids=frozenset(),
            ),
            warning,
        )


def _load_existing_local_secrets(
    settings: Settings,
) -> tuple[LocalSecrets, str | None]:
    try:
        return load_local_secrets(settings), None
    except UnclawError as exc:
        warning = (
            "Local secrets note: "
            f"{exc} Unclaw will ask you to rewrite the Telegram token for this project."
        )
        return LocalSecrets(), warning


def _build_prompt_ui(
    *,
    input_func: InputFunc,
    output_func: OutputFunc,
) -> PromptUI:
    if (
        input_func is input
        and output_func is print
        and questionary is not None
        and sys.stdin.isatty()
        and sys.stdout.isatty()
    ):
        return InteractivePromptUI(output_func=output_func)
    return FallbackPromptUI(input_func=input_func, output_func=output_func)


def _default_channel_preset(settings: Settings) -> str:
    terminal_enabled = settings.app.channels.terminal_enabled
    telegram_enabled = settings.app.channels.telegram_enabled
    if terminal_enabled and telegram_enabled:
        return "terminal_and_telegram"
    if telegram_enabled:
        return "telegram_only"
    return "terminal_only"


def _prompt_channel_preset(
    *,
    prompt_ui: PromptUI,
    default_preset: str,
) -> str:
    return prompt_ui.select(
        "Which channel setup should Unclaw enable?",
        options=_CHANNEL_PRESET_OPTIONS,
        default=default_preset,
        help_text=(
            "Pick one clear setup. If Telegram is enabled, Unclaw will ask for "
            "the bot token and store it locally for this project."
        ),
    )


def _enabled_channels_from_preset(channel_preset: str) -> tuple[str, ...]:
    match channel_preset:
        case "terminal_only":
            return ("terminal",)
        case "terminal_and_telegram":
            return ("terminal", "telegram")
        case "telegram_only":
            return ("telegram",)
        case _:
            raise ValueError(f"Unsupported channel preset: {channel_preset}")


def _prompt_model_profiles(
    settings: Settings,
    *,
    prompt_ui: PromptUI,
    ollama_status: OllamaStatus,
) -> dict[str, ModelProfileDraft]:
    profiles: dict[str, ModelProfileDraft] = {}
    recommended = recommended_model_profiles()
    installed_model_names = (
        ollama_status.model_names if ollama_status.is_running else ()
    )

    for profile_name in _PROFILE_ORDER:
        current_profile = settings.models.get(profile_name)
        current_model_name = (
            current_profile.model_name
            if current_profile is not None
            else recommended[profile_name].model_name
        )
        recommended_model_name = recommended[profile_name].model_name

        options, default_choice = _build_profile_menu_options(
            profile_name=profile_name,
            has_current_profile=current_profile is not None,
            current_model_name=current_model_name,
            recommended_model_name=recommended_model_name,
            installed_model_names=installed_model_names,
        )
        selection = prompt_ui.select(
            f"{profile_name.title()} profile",
            options=options,
            default=default_choice,
            help_text=_build_profile_help_text(
                profile_name=profile_name,
                installed_model_names=installed_model_names,
            ),
        )

        if selection == "custom":
            template = (
                _draft_from_profile(current_profile)
                if current_profile is not None
                else recommended[profile_name]
            )
            model_name = prompt_ui.text(
                f"Custom model name for {profile_name}",
                default=current_model_name,
                help_text=(
                    "Enter the exact local model name you plan to pull or already "
                    "have in Ollama."
                ),
                validator=_validate_model_name,
            )
        elif selection == "recommended":
            template = recommended[profile_name]
            model_name = recommended_model_name
        elif selection == "installed":
            template = (
                _draft_from_profile(current_profile)
                if current_profile is not None
                else recommended[profile_name]
            )
            model_name = _prompt_installed_model_name(
                profile_name=profile_name,
                current_model_name=current_model_name,
                recommended_model_name=recommended_model_name,
                installed_model_names=installed_model_names,
                prompt_ui=prompt_ui,
            )
        else:
            template = (
                _draft_from_profile(current_profile)
                if current_profile is not None
                else recommended[profile_name]
            )
            model_name = current_model_name

        profiles[profile_name] = ModelProfileDraft(
            provider=template.provider,
            model_name=model_name,
            temperature=template.temperature,
            thinking_supported=template.thinking_supported,
            tool_mode=template.tool_mode,
        )

    return profiles


def _build_profile_menu_options(
    *,
    profile_name: str,
    has_current_profile: bool,
    current_model_name: str,
    recommended_model_name: str,
    installed_model_names: tuple[str, ...],
) -> tuple[tuple[MenuOption, ...], str]:
    options: list[MenuOption] = [
        MenuOption(
            value="current",
            label=f"Keep current model: {current_model_name}",
            description=(
                "Leave this profile exactly as it is."
                if has_current_profile
                else "No current profile was found, so this uses the detected fallback."
            ),
        ),
        MenuOption(
            value="recommended",
            label=f"Use recommended model: {recommended_model_name}",
            description=(
                "This already matches the current config."
                if current_model_name == recommended_model_name
                else "Use the updated local-friendly default for this role."
            ),
        ),
    ]
    if installed_model_names:
        options.append(
            MenuOption(
                value="installed",
                label="Choose from installed Ollama models",
                description=f"{len(installed_model_names)} local model(s) detected in Ollama.",
            )
        )
    options.append(
        MenuOption(
            value="custom",
            label="Enter a custom model name",
            description="Type another local model name yourself.",
        )
    )
    return tuple(options), "current" if has_current_profile else "recommended"


def _build_profile_help_text(
    *,
    profile_name: str,
    installed_model_names: tuple[str, ...],
) -> str:
    base_text = _PROFILE_DESCRIPTIONS[profile_name]
    if installed_model_names:
        return (
            f"{base_text} Ollama is running with "
            f"{len(installed_model_names)} installed model(s) available."
        )
    return (
        f"{base_text} Start Ollama if you want to browse installed local models during setup."
    )


def _prompt_installed_model_name(
    *,
    profile_name: str,
    current_model_name: str,
    recommended_model_name: str,
    installed_model_names: tuple[str, ...],
    prompt_ui: PromptUI,
) -> str:
    options = tuple(
        MenuOption(
            value=model_name,
            label=model_name,
            description=_describe_installed_model(
                model_name=model_name,
                current_model_name=current_model_name,
                recommended_model_name=recommended_model_name,
            ),
        )
        for model_name in installed_model_names
    )
    default_model_name = _resolve_installed_model_default(
        installed_model_names=installed_model_names,
        current_model_name=current_model_name,
        recommended_model_name=recommended_model_name,
    )
    return prompt_ui.select(
        f"Installed Ollama model for {profile_name}",
        options=options,
        default=default_model_name,
        help_text="Choose one of the local models already available in Ollama.",
    )


def _prompt_telegram_bot_token(
    *,
    existing_token: str | None,
    prompt_ui: PromptUI,
) -> str:
    prompt_ui.section(
        "🤖 Telegram bot",
        (
            "Connect your Telegram bot with the token from BotFather. "
            "Unclaw stores it locally in `config/secrets.yaml` for this project."
        ),
    )
    prompt_ui.info("Get this token from BotFather after you create or open your bot.")
    prompt_ui.info("Example format: 123456789:AA...")
    prompt_ui.info("The token stays visible while you type so you can verify the paste.")
    prompt_ui.info(
        "Secure by default: `allowed_chat_ids: []` denies all Telegram chats until you authorize one explicitly."
    )
    prompt_ui.info(
        "Advanced fallback: `unclaw telegram` can still read "
        f"`{DEFAULT_TELEGRAM_TOKEN_ENV_VAR}` from the environment."
    )

    if existing_token is not None:
        keep_existing = prompt_ui.confirm(
            "Keep the Telegram bot token already stored in `config/secrets.yaml`?",
            default=True,
            help_text="Choose No if you want to replace it with a fresh token from BotFather.",
        )
        if keep_existing:
            return existing_token

    return prompt_ui.text(
        "Telegram bot token",
        default="",
        help_text=(
            "Paste the full token from BotFather. Unclaw will write it to "
            "`config/secrets.yaml` for this project."
        ),
        instruction="Visible while typing. Press Enter to save it locally.",
        validator=_validate_telegram_bot_token,
    )


def _validate_model_name(value: str) -> str | None:
    normalized = value.strip()
    if not normalized:
        return "The model name cannot be empty."
    return None


def _validate_telegram_bot_token(value: str) -> str | None:
    try:
        validate_telegram_bot_token(value)
    except ConfigurationError as exc:
        return str(exc)
    return None


def _draft_from_profile(profile: ModelProfile) -> ModelProfileDraft:
    return ModelProfileDraft(
        provider=profile.provider,
        model_name=profile.model_name,
        temperature=profile.temperature,
        thinking_supported=profile.thinking_supported,
        tool_mode=profile.tool_mode,
    )


def _print_plan_summary(plan: OnboardingPlan, *, output_func: OutputFunc) -> None:
    output_func(
        "- Setup style: "
        f"{'recommended guided' if plan.beginner_mode else 'advanced custom'}"
    )
    output_func(f"- Logging: {plan.logging_mode}")
    output_func(
        "- Channels: "
        + ", ".join(
            channel_name for channel_name in ("terminal", "telegram") if channel_name in plan.enabled_channels
        )
    )
    output_func("- Model lineup:")
    for profile_name in _PROFILE_ORDER:
        output_func(f"  {profile_name}: {plan.model_profiles[profile_name].model_name}")
    if "telegram" in plan.enabled_channels:
        output_func("- Telegram token: stored locally in config/secrets.yaml")
        output_func(
            f"- Telegram env fallback: {plan.telegram_bot_token_env_var}"
        )
        if plan.telegram_allowed_chat_ids:
            count = len(plan.telegram_allowed_chat_ids)
            label = "chat" if count == 1 else "chats"
            output_func(f"- Telegram access: allowlist with {count} authorized {label}")
        else:
            output_func("- Telegram access: secure deny-by-default (no chats authorized yet)")


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


def _build_onboarding_banner(*, settings: Settings, ollama_status: OllamaStatus) -> str:
    return build_banner(
        title="Unclaw setup",
        subtitle="Guided local setup for channels, models, and startup defaults.",
        rows=(
            ("project", str(settings.paths.project_root)),
            ("config", str(settings.paths.config_dir)),
            ("ollama", _describe_ollama_status(ollama_status)),
        ),
    )


def _unique_model_names(missing_profiles: tuple[tuple[str, str], ...]) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(model_name for _profile_name, model_name in missing_profiles)
    )


def _resolve_select_default(*, options: tuple[MenuOption, ...], default: str) -> str:
    for option in options:
        if option.value == default:
            return default
    return options[0].value


def _resolve_installed_model_default(
    *,
    installed_model_names: tuple[str, ...],
    current_model_name: str,
    recommended_model_name: str,
) -> str:
    if current_model_name in installed_model_names:
        return current_model_name
    if recommended_model_name in installed_model_names:
        return recommended_model_name
    return installed_model_names[0]


def _describe_installed_model(
    *,
    model_name: str,
    current_model_name: str,
    recommended_model_name: str,
) -> str | None:
    tags: list[str] = []
    if model_name == current_model_name:
        tags.append("current")
    if model_name == recommended_model_name:
        tags.append("recommended")
    if not tags:
        return None
    return ", ".join(tags)
