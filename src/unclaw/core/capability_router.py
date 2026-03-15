"""Lightweight capability routing for plain-language user turns."""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from enum import StrEnum

from unclaw.core.capabilities import RuntimeCapabilitySummary
from unclaw.llm.base import LLMMessage, LLMRole, ResolvedModelProfile
from unclaw.llm.ollama_provider import OllamaProvider
from unclaw.settings import Settings

_LOCAL_FILE_FALLBACK_REPLY = (
    "Tell me which local file or folder you want to inspect. "
    "You can also use /read <path> or /ls [path]."
)
_AMBIGUOUS_FALLBACK_REPLY = (
    "I am not sure whether you want a direct answer, web research, "
    "or help with a local file. Please clarify."
)
_AUTOMATION_FALLBACK_REPLY = (
    "That looks like a system or automation request. "
    "Unclaw does not execute that autonomously in this runtime path yet."
)
_ROUTING_CONFIDENCE_VALUES = frozenset({"high", "medium", "low"})
_LOCAL_FILE_EXTENSIONS = frozenset(
    {
        "c",
        "cfg",
        "cpp",
        "css",
        "csv",
        "go",
        "h",
        "html",
        "ini",
        "java",
        "js",
        "json",
        "md",
        "mjs",
        "py",
        "pyi",
        "rb",
        "rs",
        "sh",
        "sql",
        "toml",
        "tsx",
        "ts",
        "txt",
        "xml",
        "yaml",
        "yml",
    }
)
_WINDOWS_PATH_PATTERN = re.compile(r"^[A-Za-z]:\\\\")
_URL_PREFIXES = ("http://", "https://")


class CapabilityKind(StrEnum):
    """High-level capability paths supported in Block B."""

    DIRECT_ANSWER = "direct_answer"
    WEB_RESEARCH = "web_research"
    LOCAL_FILE_INTENT = "local_file_intent"
    AMBIGUOUS = "ambiguous"
    AUTOMATION_INTENT = "automation_intent"


@dataclass(frozen=True, slots=True)
class CapabilityDecision:
    """Structured routing decision returned by one capability router."""

    kind: CapabilityKind
    confidence: str
    source: str
    follow_up_message: str | None = None


class CapabilityRouter(ABC):
    """Abstract capability classifier for one user turn."""

    @abstractmethod
    def route(
        self,
        *,
        settings: Settings,
        profile: ResolvedModelProfile,
        user_message: str,
        capability_summary: RuntimeCapabilitySummary,
    ) -> CapabilityDecision:
        """Classify one turn into a bounded capability path."""


@dataclass(slots=True)
class LLMCapabilityRouter(CapabilityRouter):
    """Use the active local model as a tiny structured router."""

    routing_timeout_seconds: float = 15.0

    def route(
        self,
        *,
        settings: Settings,
        profile: ResolvedModelProfile,
        user_message: str,
        capability_summary: RuntimeCapabilitySummary,
    ) -> CapabilityDecision:
        normalized_user_message = user_message.strip()
        if not normalized_user_message:
            return CapabilityDecision(
                kind=CapabilityKind.AMBIGUOUS,
                confidence="high",
                source="empty_input_fallback",
                follow_up_message=_AMBIGUOUS_FALLBACK_REPLY,
            )

        try:
            provider = self._create_provider(
                profile.provider,
                settings=settings,
            )
            router_profile = replace(profile, temperature=0.0)
            response = provider.chat(
                profile=router_profile,
                messages=_build_router_messages(
                    user_message=normalized_user_message,
                    capability_summary=capability_summary,
                ),
                timeout_seconds=min(
                    settings.app.providers.ollama.timeout_seconds,
                    self.routing_timeout_seconds,
                ),
                thinking_enabled=False,
            )
        except Exception:
            return _fallback_decision_for_message(normalized_user_message)

        parsed_decision = _parse_router_response(response.content)
        if parsed_decision is None:
            return _fallback_decision_for_message(normalized_user_message)

        return _apply_local_file_safety_guardrail(
            normalized_user_message,
            parsed_decision,
        )

    def _create_provider(
        self,
        provider_name: str,
        *,
        settings: Settings,
    ) -> OllamaProvider:
        if provider_name == OllamaProvider.provider_name:
            return OllamaProvider(
                default_timeout_seconds=settings.app.providers.ollama.timeout_seconds
            )

        raise ValueError(f"Unsupported provider for capability routing: {provider_name}")


def _build_router_messages(
    *,
    user_message: str,
    capability_summary: RuntimeCapabilitySummary,
) -> list[LLMMessage]:
    search_status = (
        "available"
        if capability_summary.web_search_available
        else "unavailable"
    )
    file_status = (
        "available"
        if capability_summary.local_file_read_available
        or capability_summary.local_directory_listing_available
        else "unavailable"
    )

    system_lines = (
        "You classify one user turn for Unclaw, a local-first assistant runtime.",
        "Return valid JSON only.",
        (
            "Choose one capability: direct_answer, web_research, "
            "local_file_intent, ambiguous, automation_intent."
        ),
        "Definitions:",
        "- direct_answer: answer directly from general knowledge or reasoning.",
        (
            "- web_research: the user needs current, external, or source-backed "
            "web information."
        ),
        (
            "- local_file_intent: the user wants help with local files, folders, "
            "repository content, paths, or code already on disk."
        ),
        (
            "- ambiguous: it is unclear whether the user wants a direct answer, "
            "web research, or local file help."
        ),
        (
            "- automation_intent: the user wants system actions, file edits, "
            "command execution, or other automation."
        ),
        "Safety rules:",
        (
            "- If the turn refers to local files, paths, directories, repository "
            "content, or code on disk, prefer local_file_intent over web_research."
        ),
        "- If you are unsure, choose ambiguous.",
        (
            "- follow_up must be one short sentence in the user's language when "
            "the capability is local_file_intent, ambiguous, or automation_intent."
        ),
        (
            "- Use null for follow_up when the capability is direct_answer or "
            "web_research."
        ),
        "Current runtime facts:",
        f"- Web research availability: {search_status}.",
        f"- Local file inspection availability: {file_status}.",
        (
            'Output schema: {"capability":"...", "confidence":"high|medium|low", '
            '"follow_up":"... or null"}'
        ),
    )

    return [
        LLMMessage(role=LLMRole.SYSTEM, content="\n".join(system_lines)),
        LLMMessage(role=LLMRole.USER, content=user_message),
    ]


def _parse_router_response(raw_text: str) -> CapabilityDecision | None:
    payload = _extract_json_payload(raw_text)
    if payload is None:
        return None

    raw_capability = payload.get("capability")
    if not isinstance(raw_capability, str):
        return None

    try:
        kind = CapabilityKind(raw_capability.strip().lower())
    except ValueError:
        return None

    raw_confidence = payload.get("confidence")
    confidence = (
        raw_confidence.strip().lower()
        if isinstance(raw_confidence, str)
        else "medium"
    )
    if confidence not in _ROUTING_CONFIDENCE_VALUES:
        confidence = "medium"

    raw_follow_up = payload.get("follow_up")
    follow_up = raw_follow_up.strip() if isinstance(raw_follow_up, str) else None
    if follow_up == "":
        follow_up = None

    if kind is CapabilityKind.LOCAL_FILE_INTENT and follow_up is None:
        follow_up = _LOCAL_FILE_FALLBACK_REPLY
    elif kind is CapabilityKind.AMBIGUOUS and follow_up is None:
        follow_up = _AMBIGUOUS_FALLBACK_REPLY
    elif kind is CapabilityKind.AUTOMATION_INTENT and follow_up is None:
        follow_up = _AUTOMATION_FALLBACK_REPLY

    return CapabilityDecision(
        kind=kind,
        confidence=confidence,
        source="llm_router",
        follow_up_message=follow_up,
    )


def _extract_json_payload(raw_text: str) -> dict[str, object] | None:
    stripped_text = raw_text.strip()
    if not stripped_text:
        return None

    for candidate in (
        stripped_text,
        _slice_first_json_object(stripped_text),
    ):
        if candidate is None:
            continue
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload

    return None


def _slice_first_json_object(raw_text: str) -> str | None:
    start_index = raw_text.find("{")
    end_index = raw_text.rfind("}")
    if start_index < 0 or end_index <= start_index:
        return None
    return raw_text[start_index : end_index + 1]


def _apply_local_file_safety_guardrail(
    user_message: str,
    decision: CapabilityDecision,
) -> CapabilityDecision:
    if not _has_strong_local_artifact_signal(user_message):
        return decision

    if decision.kind in {
        CapabilityKind.LOCAL_FILE_INTENT,
        CapabilityKind.AUTOMATION_INTENT,
    }:
        return decision

    return CapabilityDecision(
        kind=CapabilityKind.LOCAL_FILE_INTENT,
        confidence="high",
        source="local_artifact_guardrail",
        follow_up_message=decision.follow_up_message or _LOCAL_FILE_FALLBACK_REPLY,
    )


def _fallback_decision_for_message(user_message: str) -> CapabilityDecision:
    if _has_strong_local_artifact_signal(user_message):
        return CapabilityDecision(
            kind=CapabilityKind.LOCAL_FILE_INTENT,
            confidence="high",
            source="local_artifact_guardrail",
            follow_up_message=_LOCAL_FILE_FALLBACK_REPLY,
        )

    return CapabilityDecision(
        kind=CapabilityKind.AMBIGUOUS,
        confidence="low",
        source="router_fallback",
        follow_up_message=_AMBIGUOUS_FALLBACK_REPLY,
    )


def _has_strong_local_artifact_signal(user_message: str) -> bool:
    for raw_token in user_message.split():
        token = raw_token.strip("`'\"()[]{}<>.,:;!?")
        if not token:
            continue
        if token.startswith(_URL_PREFIXES):
            continue
        if token.startswith(("./", "../", "~/", "/")):
            return True
        if _WINDOWS_PATH_PATTERN.match(token):
            return True
        if ("/" in token or "\\" in token) and not token.startswith(_URL_PREFIXES):
            return True
        if _looks_like_local_filename(token):
            return True

    return False


def _looks_like_local_filename(token: str) -> bool:
    if "." not in token or token.startswith("."):
        return False
    if token.lower().startswith("www."):
        return False

    stem, _, extension = token.rpartition(".")
    if not stem or not extension:
        return False

    return extension.lower() in _LOCAL_FILE_EXTENSIONS
