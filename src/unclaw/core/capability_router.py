"""Lightweight capability routing for plain-language user turns."""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import StrEnum

from unclaw.core.capabilities import RuntimeCapabilitySummary
from unclaw.llm.base import LLMMessage, LLMRole, ResolvedModelProfile
from unclaw.llm.ollama_provider import OllamaProvider
from unclaw.schemas.chat import ChatMessage, MessageRole
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
        recent_history: Sequence[ChatMessage] = (),
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
        recent_history: Sequence[ChatMessage] = (),
    ) -> CapabilityDecision:
        normalized_user_message = user_message.strip()
        if not normalized_user_message:
            return CapabilityDecision(
                kind=CapabilityKind.AMBIGUOUS,
                confidence="high",
                source="empty_input_fallback",
                follow_up_message=_AMBIGUOUS_FALLBACK_REPLY,
            )

        # Pending clarification: if the assistant just asked a narrow question
        # and the user gives a short direct answer, resolve it immediately.
        clarification_decision = _is_pending_clarification_answer(
            normalized_user_message, recent_history,
        )
        if clarification_decision is not None:
            return clarification_decision

        # Follow-up resolution: short pronoun/reference messages that clearly
        # refer to a recent conversation topic should stay as direct_answer
        # instead of being classified as ambiguous by the LLM router.
        if _is_obvious_follow_up(normalized_user_message, recent_history):
            return CapabilityDecision(
                kind=CapabilityKind.DIRECT_ANSWER,
                confidence="high",
                source="follow_up_resolution",
            )

        # Subject continuation: "cherche sur Wikipédia" with active subject
        if _is_subject_continuation(normalized_user_message, recent_history):
            if capability_summary.web_search_available:
                return CapabilityDecision(
                    kind=CapabilityKind.WEB_RESEARCH,
                    confidence="high",
                    source="subject_continuation",
                )

        # Freshness pre-routing: obvious freshness signals bypass LLM router
        if (
            capability_summary.web_search_available
            and _has_obvious_freshness_signal(normalized_user_message)
        ):
            return CapabilityDecision(
                kind=CapabilityKind.WEB_RESEARCH,
                confidence="high",
                source="freshness_heuristic",
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
        "Freshness signals — prefer web_research when the question:",
        "- asks about current events, recent news, or today's information,",
        "- asks who currently holds a role, title, or position,",
        "- asks about prices, scores, weather, or live data,",
        "- asks about upcoming or future events, releases, or announcements,",
        "- uses words like 'current', 'latest', 'today', 'now', 'upcoming', 'next', 'recent'.",
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


# ---------------------------------------------------------------------------
# Follow-up reference resolution
# ---------------------------------------------------------------------------

_FOLLOW_UP_PRONOUN_PATTERN = re.compile(
    r"\b(?:"
    r"(?:he|she|him|her|his|hers|they|them|their|theirs|it|its)"
    r"|(?:il|elle|lui|eux|elles|son|sa|ses|leur|leurs)"
    r"|(?:er|sie|ihm|ihr|ihnen|sein|seine|seinen|seiner)"
    r")\b",
    flags=re.IGNORECASE,
)

_FOLLOW_UP_REFERENCE_PATTERN = re.compile(
    r"\b(?:"
    r"(?:that|this|those|these)"
    r"|(?:the same|the one|same person|same thing)"
    r"|(?:more detail|more about|tell me more|describe|explain)"
    r"|(?:and (?:what|how|where|when|why|who))"
    r"|(?:et (?:son|sa|ses|quel|quelle|où|quand|comment|pourquoi))"
    r"|(?:fais|donne|parle|decris|décris)"
    r"|(?:carrière|career|age|âge|born|née?|biography)"
    r")\b",
    flags=re.IGNORECASE,
)

_SEARCH_TOOL_HISTORY_PREFIX = "Tool: search_web\n"

_MAX_FOLLOW_UP_WORDS = 15

# ---------------------------------------------------------------------------
# Pending clarification resolution
# ---------------------------------------------------------------------------

_CLARIFICATION_QUESTION_PATTERN = re.compile(
    r"\b(?:"
    r"(?:souhaitez[- ]vous|voulez[- ]vous|préférez[- ]vous|dois[- ]je)"
    r"|(?:would you (?:like|prefer)|should I|do you want)"
    r"|(?:internet ou|web ou|en ligne ou|local ou|fichier ou)"
    r"|(?:internet or|web or|online or|local or|file or)"
    r")\b",
    flags=re.IGNORECASE,
)

_CLARIFICATION_WEB_ANSWERS = frozenset({
    "internet", "web", "en ligne", "online", "oui", "yes",
    "recherche", "search", "cherche", "go ahead", "vas-y", "ok",
})

_CLARIFICATION_LOCAL_ANSWERS = frozenset({
    "local", "fichier", "file", "disque", "disk", "dossier",
    "folder", "non", "no", "pdf", "le pdf", "le fichier",
})

_CLARIFICATION_WIKIPEDIA_ANSWERS = frozenset({
    "wikipedia", "wikipédia", "wiki",
})


def _is_pending_clarification_answer(
    user_message: str,
    recent_history: Sequence[ChatMessage],
) -> CapabilityDecision | None:
    """Detect when the user is answering a pending clarification question.

    Returns a routing decision if the last assistant message was a clarification
    question and the current user message is a short direct answer to it.
    Returns None otherwise.
    """
    if not recent_history:
        return None

    words = user_message.split()
    if len(words) > 5:
        return None

    normalized = user_message.strip().lower().rstrip(".,!?")

    last_assistant_message = _find_last_assistant_message(recent_history)
    if last_assistant_message is None:
        return None

    assistant_text = last_assistant_message.content.strip()
    if not assistant_text.endswith("?"):
        return None

    is_clarification = (
        _CLARIFICATION_QUESTION_PATTERN.search(assistant_text) is not None
    )
    if not is_clarification:
        return None

    if normalized in _CLARIFICATION_WEB_ANSWERS:
        return CapabilityDecision(
            kind=CapabilityKind.WEB_RESEARCH,
            confidence="high",
            source="pending_clarification",
        )

    if normalized in _CLARIFICATION_LOCAL_ANSWERS:
        return CapabilityDecision(
            kind=CapabilityKind.LOCAL_FILE_INTENT,
            confidence="high",
            source="pending_clarification",
            follow_up_message=_LOCAL_FILE_FALLBACK_REPLY,
        )

    if normalized in _CLARIFICATION_WIKIPEDIA_ANSWERS:
        return CapabilityDecision(
            kind=CapabilityKind.WEB_RESEARCH,
            confidence="high",
            source="pending_clarification",
        )

    return None


def _find_last_assistant_message(
    history: Sequence[ChatMessage],
) -> ChatMessage | None:
    """Return the most recent assistant message, skipping trailing user messages.

    The caller passes the full recent history which may include the current
    user message at the end.  We skip those trailing USER messages to find
    the assistant turn that immediately preceded this user turn.
    """
    skipping_trailing_user = True
    for message in reversed(history):
        if skipping_trailing_user and message.role is MessageRole.USER:
            continue
        skipping_trailing_user = False
        if message.role is MessageRole.ASSISTANT:
            return message
        # Hit a TOOL or something else — stop
        return None
    return None


# ---------------------------------------------------------------------------
# Subject continuation for "search on X" style follow-ups
# ---------------------------------------------------------------------------

_SUBJECT_CONTINUATION_PATTERN = re.compile(
    r"\b(?:"
    r"cherche(?:r)?\s+(?:sur|dans|via)\b"
    r"|search(?:es)?\s+(?:on|in|via|for)\b"
    r"|look\s+(?:on|up\s+on|it\s+up\s+on)\b"
    r"|regarde(?:r)?\s+(?:sur|dans)\b"
    r"|trouve(?:r)?\s+(?:sur|dans)\b"
    r")\b",
    flags=re.IGNORECASE,
)


def _is_subject_continuation(
    user_message: str,
    recent_history: Sequence[ChatMessage],
) -> bool:
    """Detect short follow-up messages that continue the current subject.

    Handles cases like "cherche sur Wikipédia" or "search on Wikipedia" where
    the user doesn't use pronouns but clearly refers to the current subject.
    """
    if not recent_history:
        return False

    words = user_message.split()
    if len(words) > _MAX_FOLLOW_UP_WORDS:
        return False

    if _SUBJECT_CONTINUATION_PATTERN.search(user_message) is None:
        return False

    return _has_recent_substantive_context(recent_history)


# ---------------------------------------------------------------------------
# Freshness pre-routing heuristic
# ---------------------------------------------------------------------------

_FRESHNESS_SIGNAL_PATTERN = re.compile(
    r"\b(?:"
    # French freshness signals
    r"actualit[eé]s?\s+(?:du jour|d'aujourd|importantes?|r[eé]centes?)"
    r"|(?:derni[eè]res?\s+(?:nouvelles?|infos?|actualit[eé]s?))"
    r"|(?:r[eé]sum[eé]\s+(?:des?\s+)?actualit[eé]s?)"
    r"|(?:quoi de neuf|qu'?est[- ]ce qui se passe)"
    r"|(?:news?\s+du jour|news?\s+d'aujourd)"
    # English freshness signals
    r"|(?:today'?s?\s+(?:news|headlines|events|updates?))"
    r"|(?:latest\s+(?:news|headlines|updates?|developments?))"
    r"|(?:current\s+(?:news|events|price|status|score|weather))"
    r"|(?:recent\s+(?:news|headlines|updates?|developments?|events?))"
    r"|(?:what(?:'s| is)\s+(?:happening|going on)\s+(?:today|now|right now))"
    r"|(?:breaking\s+news)"
    r")\b",
    flags=re.IGNORECASE,
)


def _has_obvious_freshness_signal(user_message: str) -> bool:
    """Detect messages that clearly need current/live web data.

    This pre-LLM heuristic catches obvious freshness requests to ensure
    reliable routing without depending on the local model's classification.
    """
    return _FRESHNESS_SIGNAL_PATTERN.search(user_message) is not None


# ---------------------------------------------------------------------------
# Follow-up reference resolution
# ---------------------------------------------------------------------------


def _is_obvious_follow_up(
    user_message: str,
    recent_history: Sequence[ChatMessage],
) -> bool:
    """Detect short follow-up messages that clearly refer to recent context.

    Returns True only when:
    1. The message is short (≤ 15 words)
    2. The message contains pronouns or follow-up reference language
    3. The recent history has substantial context (assistant reply or search results)
    """
    if not recent_history:
        return False

    words = user_message.split()
    if len(words) > _MAX_FOLLOW_UP_WORDS:
        return False

    has_pronoun = _FOLLOW_UP_PRONOUN_PATTERN.search(user_message) is not None
    has_reference = _FOLLOW_UP_REFERENCE_PATTERN.search(user_message) is not None
    if not has_pronoun and not has_reference:
        return False

    return _has_recent_substantive_context(recent_history)


def _has_recent_substantive_context(
    history: Sequence[ChatMessage],
) -> bool:
    """Check whether the last few turns contain assistant replies or search results."""
    # Look at the last 6 messages for substantive context
    recent = history[-6:] if len(history) > 6 else history
    for message in reversed(recent):
        if message.role is MessageRole.ASSISTANT and len(message.content.strip()) > 30:
            return True
        if (
            message.role is MessageRole.TOOL
            and message.content.startswith(_SEARCH_TOOL_HISTORY_PREFIX)
        ):
            return True
    return False
