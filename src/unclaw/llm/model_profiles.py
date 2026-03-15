"""Helpers for resolving configured model profiles."""

from __future__ import annotations

from unclaw.errors import ConfigurationError
from unclaw.llm.base import ModelCapabilities, ResolvedModelProfile
from unclaw.settings import ModelProfile, Settings


def resolve_model_profile(
    settings: Settings,
    profile_name: str,
) -> ResolvedModelProfile:
    """Resolve one configured model profile by name."""
    try:
        profile = settings.models[profile_name]
    except KeyError as exc:
        raise ConfigurationError(
            f"Model profile '{profile_name}' is not defined in settings."
        ) from exc

    return _to_resolved_profile(profile)


def get_default_model_profile(settings: Settings) -> ResolvedModelProfile:
    """Resolve the default model profile from application settings."""
    return resolve_model_profile(settings, settings.app.default_model_profile)


def _to_resolved_profile(profile: ModelProfile) -> ResolvedModelProfile:
    return ResolvedModelProfile(
        name=profile.name,
        provider=profile.provider,
        model_name=profile.model_name,
        temperature=profile.temperature,
        capabilities=_derive_capabilities(profile),
    )


def _derive_capabilities(profile: ModelProfile) -> ModelCapabilities:
    tool_mode_normalized = profile.tool_mode.strip().lower()
    supports_tools = tool_mode_normalized != "none"

    return ModelCapabilities(
        thinking_supported=profile.thinking_supported,
        tool_mode=profile.tool_mode,
        supports_tools=supports_tools,
        supports_native_tool_calling=tool_mode_normalized == "native",
        supports_agent_mode=supports_tools,
    )
