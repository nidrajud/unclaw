"""Grounding helpers for search-backed answers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
import re
import unicodedata
from typing import Any

from unclaw.schemas.chat import ChatMessage, MessageRole

_MAX_COMPOSED_FACTS = 3
_SEARCH_TOOL_PREFIX = "Tool: search_web\n"
_STALE_AS_OF_PATTERN = re.compile(
    r"\bas of\s+(?:"
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|"
    r"nov(?:ember)?|dec(?:ember)?)\s+\d{4}"
    r"|"
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|"
    r"nov(?:ember)?|dec(?:ember)?)\s+\d{1,2},\s+\d{4}"
    r")\b",
    flags=re.IGNORECASE,
)
_AGE_QUERY_PATTERNS = (
    re.compile(r"\bhow old\b"),
    re.compile(r"\bage\b"),
    re.compile(r"\bdate of birth\b"),
    re.compile(r"\bbirth date\b"),
    re.compile(r"\bborn\b"),
    re.compile(r"\bquel age\b"),
    re.compile(r"\bage de\b"),
    re.compile(r"\bdate de naissance\b"),
)
_PERSON_QUERY_PATTERNS = (
    re.compile(r"\bwho is\b"),
    re.compile(r"\bwho[' ]?s\b"),
    re.compile(r"\btell me(?: everything)? you know about\b"),
    re.compile(r"\btell me about\b"),
    re.compile(r"\bwhat has\b"),
    re.compile(r"\bwhat did\b"),
    re.compile(r"\bprofile\b"),
    re.compile(r"\bqui est\b"),
    re.compile(r"\bparle moi de\b"),
    re.compile(r"\bfais moi un resume\b"),
    re.compile(r"\bresume sur\b"),
)
_LOW_VALUE_DETAIL_PATTERNS = (
    re.compile(r"\bjourneys?\b"),
    re.compile(r"\binspiring\b"),
    re.compile(r"\bpassion(?:ate)?\b"),
    re.compile(r"\bit seems\b"),
    re.compile(r"\bseems to\b"),
    re.compile(r"\bappears to\b"),
    re.compile(r"\bprobably\b"),
    re.compile(r"\boften\b"),
    re.compile(r"\bpodcasts?\b"),
    re.compile(r"\bepisodes?\b"),
    re.compile(r"\bnewsletters?\b"),
    re.compile(r"\bblogs?\b"),
    re.compile(r"\bsubstack\b"),
)
_HANDLE_HINT_PATTERN = re.compile(
    r"(?<!\w)@[a-z0-9_.]{2,32}\b",
    flags=re.IGNORECASE,
)
_HANDLE_TERM_PATTERN = re.compile(
    r"\b(?:username|handle|instagram|twitter|x\.com|tiktok|youtube|linkedin|threads)\b",
    flags=re.IGNORECASE,
)
_AGE_CLAIM_PATTERN = re.compile(
    r"\b\d{1,3}\s+years?\s+old\b|\baged\s+\d{1,3}\b",
    flags=re.IGNORECASE,
)
_SOURCE_BULLET_PATTERN = re.compile(r"^- (?P<title>.*?): (?P<url>https?://\S+)$")
_FINDING_BULLET_PATTERN = re.compile(
    r"^- \[(?P<label>[a-z]+); (?P<support>\d+) source(?:s)?\] (?P<text>.+)$",
    flags=re.IGNORECASE,
)
_FOLLOW_UP_QUERY_PATTERN = re.compile(
    r"\b(?:that|this|those|these|it|them|shorter|briefly|recap|summari[sz]e|"
    r"clarify|expand|verify|confirm|source|sources|what about|and what)\b",
    flags=re.IGNORECASE,
)
_MONTH_DATE_PATTERNS = (
    "%B %d, %Y",
    "%b %d, %Y",
    "%d %B %Y",
    "%d %b %Y",
)


@dataclass(frozen=True, slots=True)
class SearchGroundingFinding:
    """One grounded fact extracted from retrieved search context."""

    text: str
    support_count: int
    score: float
    source_titles: tuple[str, ...]
    source_urls: tuple[str, ...]
    confidence: str


@dataclass(frozen=True, slots=True)
class SearchGroundingContext:
    """Normalized search grounding data used for prompt shaping and rewrites."""

    query: str
    current_date: date
    supported_findings: tuple[SearchGroundingFinding, ...]
    uncertain_findings: tuple[SearchGroundingFinding, ...]
    display_sources: tuple[tuple[str, str], ...]
    birth_date: date | None = None


def build_search_answer_contract(*, current_date: date | None = None) -> str:
    """Build the system note for grounded search-backed answering."""
    resolved_date = current_date or date.today()
    return "\n".join(
        (
            "Search-backed answer contract:",
            f"- Current date: {resolved_date.isoformat()}",
            (
                "- If a `search_web` tool message is present, answer only from the "
                "facts it supports."
            ),
            "- Prefer directly supported facts and omit decorative filler.",
            (
                "- If a detail is weakly supported or unconfirmed, leave it out or "
                "label it clearly as unconfirmed."
            ),
            (
                "- If the evidence is weak, contradictory, or not clearly confirmed, "
                "say so explicitly. Do not overclaim or invent facts."
            ),
            (
                "- Do not invent relative date framing such as `as of May 2024` or "
                "similar stale anchors."
            ),
            (
                "- Give age only when you can compute it from a retrieved birth date "
                "and the current date above. Otherwise give the birth date only, or "
                "say the age is not confirmed."
            ),
            (
                "- For person summaries, prioritize identity, role or domain, best-"
                "supported achievements, and notable public activity."
            ),
            (
                "- Skip low-value blog, podcast, or narrative filler unless it adds "
                "a concrete fact supported by the retrieved sources."
            ),
            "- Keep the answer natural prose. Do not dump raw tool output.",
            (
                "- Never expose internal planning, tool commands, or routing "
                "mechanics in your reply. Answer directly and naturally."
            ),
        )
    )


def has_search_grounding_context(history: Sequence[ChatMessage]) -> bool:
    """Return whether the recent history already contains search-backed grounding."""
    return any(_is_search_tool_message(message) for message in history)


def build_search_grounding_context(
    payload: Mapping[str, Any] | None,
    *,
    query: str = "",
    current_date: date | None = None,
) -> SearchGroundingContext | None:
    """Normalize a search tool payload into grounded findings and sources."""
    if not isinstance(payload, Mapping):
        return None

    resolved_query = _read_query(payload, fallback=query)
    source_quality_by_url = _build_source_quality_index(payload)
    raw_findings = _read_findings(payload)
    if not raw_findings and not resolved_query and not _read_display_sources(payload):
        return None

    birth_date = _extract_birth_date(payload)
    normalized_findings = tuple(
        _classify_finding(
            raw_finding,
            source_quality_by_url=source_quality_by_url,
            birth_date=birth_date,
        )
        for raw_finding in raw_findings
    )
    supported_findings = tuple(
        finding
        for finding in normalized_findings
        if finding.confidence in {"strong", "supported"}
    )
    uncertain_findings = tuple(
        finding for finding in normalized_findings if finding.confidence == "uncertain"
    )

    return SearchGroundingContext(
        query=resolved_query,
        current_date=current_date or date.today(),
        supported_findings=supported_findings,
        uncertain_findings=uncertain_findings,
        display_sources=_read_display_sources(payload),
        birth_date=birth_date,
    )


def build_search_tool_history_summary(
    *,
    payload: Mapping[str, Any] | None,
    query: str = "",
    current_date: date | None = None,
) -> tuple[str, ...]:
    """Build compact, parseable search grounding lines for tool history."""
    grounding = build_search_grounding_context(
        payload,
        query=query,
        current_date=current_date,
    )
    if grounding is None:
        return ()

    lines: list[str] = []
    if grounding.query:
        lines.append(f"Search request: {grounding.query}")
    lines.extend(
        (
            f"Grounding date: {grounding.current_date.isoformat()}",
            "",
            "Grounding rules:",
            "- Treat Supported facts as the evidence-backed facts for this search.",
            "- If a detail appears only under Uncertain details, it is not confirmed.",
            (
                "- Do not invent relative dates. Give age only if it can be computed "
                "from a retrieved birth date and the grounding date above."
            ),
            "",
            "Supported facts:",
        )
    )

    if grounding.supported_findings:
        for finding in grounding.supported_findings[:_MAX_COMPOSED_FACTS + 2]:
            lines.append(
                f"- [{finding.confidence}; {finding.support_count} source"
                f"{'' if finding.support_count == 1 else 's'}] {finding.text}"
            )
    else:
        lines.append("- No firm details were confirmed across the retrieved sources.")

    if grounding.uncertain_findings:
        lines.extend(("", "Uncertain details:"))
        for finding in grounding.uncertain_findings[:3]:
            lines.append(
                f"- [uncertain; {finding.support_count} source"
                f"{'' if finding.support_count == 1 else 's'}] {finding.text}"
            )

    if grounding.birth_date is not None:
        lines.extend(("", f"Retrieved birth date: {grounding.birth_date.isoformat()}"))

    if grounding.display_sources:
        lines.extend(("", "Sources:"))
        for title, url in grounding.display_sources:
            if title:
                lines.append(f"- {title}: {url}")
            else:
                lines.append(f"- {url}")

    return tuple(lines)


def shape_search_backed_reply(
    reply_text: str,
    *,
    payload: Mapping[str, Any] | None,
    query: str,
    current_date: date | None = None,
) -> str:
    """Rewrite risky search-backed replies into grounded, compact prose."""
    stripped_reply = reply_text.strip()
    grounding = build_search_grounding_context(
        payload,
        query=query,
        current_date=current_date,
    )
    if grounding is None or not stripped_reply:
        return stripped_reply

    if not _reply_needs_rewrite(stripped_reply, grounding=grounding, query=query):
        return _sanitize_reply(stripped_reply)

    return _compose_grounded_answer(query=query, grounding=grounding)


def parse_search_tool_history(content: str) -> SearchGroundingContext | None:
    """Parse search grounding details back out of stored tool history text."""
    if not content.startswith(_SEARCH_TOOL_PREFIX):
        return None

    query = ""
    current_date_value = date.today()
    supported_findings: list[SearchGroundingFinding] = []
    uncertain_findings: list[SearchGroundingFinding] = []
    display_sources: list[tuple[str, str]] = []
    birth_date: date | None = None
    section = ""

    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("Search request:"):
            query = stripped.partition(":")[2].strip()
            continue
        if stripped.startswith("Grounding date:"):
            parsed_date = _parse_iso_date(stripped.partition(":")[2].strip())
            if parsed_date is not None:
                current_date_value = parsed_date
            continue
        if stripped.startswith("Retrieved birth date:"):
            birth_date = _parse_iso_date(stripped.partition(":")[2].strip())
            continue
        if stripped == "Supported facts:":
            section = "supported"
            continue
        if stripped == "Uncertain details:":
            section = "uncertain"
            continue
        if stripped == "Sources:":
            section = "sources"
            continue
        if not stripped:
            continue

        if section in {"supported", "uncertain"}:
            match = _FINDING_BULLET_PATTERN.match(stripped)
            if match is None:
                continue
            finding = SearchGroundingFinding(
                text=match.group("text").strip(),
                support_count=max(int(match.group("support")), 1),
                score=0.0,
                source_titles=(),
                source_urls=(),
                confidence=match.group("label").casefold(),
            )
            if section == "supported":
                supported_findings.append(finding)
            else:
                uncertain_findings.append(finding)
            continue

        if section == "sources":
            source_match = _SOURCE_BULLET_PATTERN.match(stripped)
            if source_match is None:
                continue
            display_sources.append(
                (
                    source_match.group("title").strip(),
                    source_match.group("url").strip(),
                )
            )

    return SearchGroundingContext(
        query=query,
        current_date=current_date_value,
        supported_findings=tuple(supported_findings),
        uncertain_findings=tuple(uncertain_findings),
        display_sources=tuple(display_sources),
        birth_date=birth_date,
    )


def should_apply_search_grounding(
    *,
    query: str,
    grounding: SearchGroundingContext | None,
) -> bool:
    """Return whether the current user turn likely refers to recent search context."""
    if grounding is None:
        return False

    normalized_query = _fold_for_match(query)
    if not normalized_query:
        return False

    if grounding.query and normalized_query == _fold_for_match(grounding.query):
        return True

    fact_tokens = {
        token
        for finding in (*grounding.supported_findings, *grounding.uncertain_findings)
        for token in _content_tokens(finding.text)
    }
    query_tokens = set(_content_tokens(query))
    if fact_tokens and query_tokens & fact_tokens:
        return True

    return _FOLLOW_UP_QUERY_PATTERN.search(query) is not None


def _read_query(payload: Mapping[str, Any], *, fallback: str) -> str:
    value = payload.get("query")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return fallback.strip()


def _read_findings(payload: Mapping[str, Any]) -> tuple[SearchGroundingFinding, ...]:
    raw_findings = payload.get("synthesized_findings")
    findings: list[SearchGroundingFinding] = []
    if isinstance(raw_findings, list):
        for entry in raw_findings:
            if not isinstance(entry, Mapping):
                continue
            text = entry.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            support_count = entry.get("support_count")
            score = entry.get("score")
            findings.append(
                SearchGroundingFinding(
                    text=text.strip(),
                    support_count=(
                        support_count if isinstance(support_count, int) and support_count > 0 else 1
                    ),
                    score=float(score) if isinstance(score, int | float) else 0.0,
                    source_titles=_read_string_tuple(entry.get("source_titles")),
                    source_urls=_read_string_tuple(entry.get("source_urls")),
                    confidence="supported",
                )
            )
    if findings:
        return tuple(findings)

    summary_points = payload.get("summary_points")
    if not isinstance(summary_points, list):
        return ()

    fallback_findings = [
        SearchGroundingFinding(
            text=item.strip(),
            support_count=1,
            score=6.0,
            source_titles=(),
            source_urls=(),
            confidence="supported",
        )
        for item in summary_points
        if isinstance(item, str) and item.strip()
    ]
    return tuple(fallback_findings)


def _read_string_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(
        item.strip() for item in value if isinstance(item, str) and item.strip()
    )


def _read_display_sources(payload: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    raw_sources = payload.get("display_sources")
    if not isinstance(raw_sources, list):
        raw_sources = payload.get("results")
        if not isinstance(raw_sources, list):
            return ()

    seen_urls: set[str] = set()
    display_sources: list[tuple[str, str]] = []
    for entry in raw_sources:
        if not isinstance(entry, Mapping):
            continue
        title = entry.get("title")
        url = entry.get("url")
        if not isinstance(url, str) or not url.strip():
            continue
        normalized_url = url.strip()
        if normalized_url in seen_urls:
            continue
        seen_urls.add(normalized_url)
        display_sources.append(
            (
                title.strip() if isinstance(title, str) else "",
                normalized_url,
            )
        )
    return tuple(display_sources)


def _build_source_quality_index(payload: Mapping[str, Any]) -> dict[str, float]:
    raw_sources = payload.get("results")
    if not isinstance(raw_sources, list):
        return {}

    quality_by_url: dict[str, float] = {}
    for entry in raw_sources:
        if not isinstance(entry, Mapping):
            continue
        url = entry.get("url")
        if not isinstance(url, str) or not url.strip():
            continue
        usefulness = entry.get("usefulness")
        evidence_count = entry.get("evidence_count")
        fetched = entry.get("fetched")
        used_snippet_fallback = entry.get("used_snippet_fallback")
        quality = float(usefulness) if isinstance(usefulness, int | float) else 0.0
        if isinstance(evidence_count, int):
            quality += min(evidence_count, 3) * 1.25
        if fetched is True:
            quality += 0.75
        if used_snippet_fallback is True:
            quality -= 0.75
        quality_by_url[url.strip()] = quality
    return quality_by_url


def _classify_finding(
    finding: SearchGroundingFinding,
    *,
    source_quality_by_url: Mapping[str, float],
    birth_date: date | None,
) -> SearchGroundingFinding:
    source_quality = max(
        (source_quality_by_url.get(url, 0.0) for url in finding.source_urls),
        default=0.0,
    )
    confidence = "uncertain"
    if finding.support_count >= 2:
        confidence = "strong"
    elif finding.support_count == 1 and (finding.score >= 6.0 or source_quality >= 6.0):
        confidence = "supported"

    if _finding_requires_extra_caution(finding.text, birth_date=birth_date):
        confidence = "uncertain"
    elif confidence == "uncertain" and finding.support_count == 1 and finding.score >= 4.0:
        confidence = "supported"

    return SearchGroundingFinding(
        text=finding.text,
        support_count=finding.support_count,
        score=finding.score,
        source_titles=finding.source_titles,
        source_urls=finding.source_urls,
        confidence=confidence,
    )


def _finding_requires_extra_caution(text: str, *, birth_date: date | None) -> bool:
    if _HANDLE_HINT_PATTERN.search(text) or _HANDLE_TERM_PATTERN.search(text):
        return True
    if _STALE_AS_OF_PATTERN.search(text):
        return True
    if _AGE_CLAIM_PATTERN.search(text):
        return birth_date is None
    if any(pattern.search(text) for pattern in _LOW_VALUE_DETAIL_PATTERNS):
        return True
    return False


def _extract_birth_date(payload: Mapping[str, Any]) -> date | None:
    candidate_texts: list[str] = []
    for finding in _read_findings(payload):
        candidate_texts.append(finding.text)

    raw_evidence = payload.get("evidence")
    if isinstance(raw_evidence, list):
        for entry in raw_evidence:
            if not isinstance(entry, Mapping):
                continue
            text = entry.get("text")
            if isinstance(text, str) and text.strip():
                candidate_texts.append(text.strip())

    for text in candidate_texts:
        parsed_date = _extract_birth_date_from_text(text)
        if parsed_date is not None:
            return parsed_date
    return None


def _extract_birth_date_from_text(text: str) -> date | None:
    match = re.search(
        r"\bborn(?: on)? (?P<date>[A-Z][a-z]+ \d{1,2}, \d{4}|"
        r"\d{4}-\d{2}-\d{2}|"
        r"\d{1,2} [A-Z][a-z]+ \d{4})\b",
        text,
    )
    if match is None:
        return None
    return _parse_human_date(match.group("date"))


def _parse_human_date(value: str) -> date | None:
    normalized_value = value.strip()
    iso_date = _parse_iso_date(normalized_value)
    if iso_date is not None:
        return iso_date
    for date_pattern in _MONTH_DATE_PATTERNS:
        try:
            return datetime.strptime(normalized_value, date_pattern).date()
        except ValueError:
            continue
    return None


def _parse_iso_date(value: str) -> date | None:
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _reply_needs_rewrite(
    reply_text: str,
    *,
    grounding: SearchGroundingContext,
    query: str,
) -> bool:
    if _STALE_AS_OF_PATTERN.search(reply_text):
        return True
    if _contains_unconfirmed_handle_claim(reply_text, grounding=grounding):
        return True
    if _contains_unsupported_age_claim(reply_text, grounding=grounding):
        return True
    if _contains_unqualified_uncertain_finding(reply_text, grounding=grounding):
        return True
    if _query_is_person_profile(query) and _contains_low_value_filler(reply_text):
        return True
    if _contains_unnecessary_hedging(reply_text):
        return True
    return False


def _contains_unconfirmed_handle_claim(
    reply_text: str,
    *,
    grounding: SearchGroundingContext,
) -> bool:
    if not (_HANDLE_HINT_PATTERN.search(reply_text) or _HANDLE_TERM_PATTERN.search(reply_text)):
        return False
    strong_text = " ".join(finding.text for finding in grounding.supported_findings)
    return not (
        _HANDLE_HINT_PATTERN.search(strong_text) or _HANDLE_TERM_PATTERN.search(strong_text)
    )


def _contains_unsupported_age_claim(
    reply_text: str,
    *,
    grounding: SearchGroundingContext,
) -> bool:
    if not _AGE_CLAIM_PATTERN.search(reply_text):
        return False
    return grounding.birth_date is None


def _contains_low_value_filler(reply_text: str) -> bool:
    return any(pattern.search(reply_text) for pattern in _LOW_VALUE_DETAIL_PATTERNS)


def _contains_unqualified_uncertain_finding(
    reply_text: str,
    *,
    grounding: SearchGroundingContext,
) -> bool:
    if _has_uncertainty_language(reply_text):
        return False

    reply_tokens = set(_content_tokens(reply_text))
    if not reply_tokens:
        return False

    for finding in grounding.uncertain_findings:
        finding_tokens = set(_content_tokens(finding.text))
        if len(finding_tokens) < 2:
            continue
        if _overlap_ratio(finding_tokens, reply_tokens) >= 0.6:
            return True
    return False


def _contains_unnecessary_hedging(reply_text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:probably|it seems|seems to|appears to|often)\b",
            reply_text,
            flags=re.IGNORECASE,
        )
    )


def _has_uncertainty_language(reply_text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:unconfirmed|not confirmed|not consistently confirmed|"
            r"one source|single source|some sources|reportedly|reported|"
            r"could not confirm|couldn't confirm|uncertain)\b",
            reply_text,
            flags=re.IGNORECASE,
        )
    )


def _sanitize_reply(reply_text: str) -> str:
    text = _STALE_AS_OF_PATTERN.sub("", reply_text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    return text.strip(" \n")


def _compose_grounded_answer(
    *,
    query: str,
    grounding: SearchGroundingContext,
) -> str:
    if _query_is_age_request(query):
        age_answer = _compose_age_answer(grounding)
        if age_answer:
            return age_answer

    supported_facts = _select_supported_facts(grounding)
    if not supported_facts and grounding.uncertain_findings:
        return _compose_uncertain_only_answer(grounding)
    if not supported_facts:
        return "I couldn't confirm any strong details from the retrieved sources."

    if _query_is_person_profile(query):
        return _compose_person_answer(
            supported_facts=supported_facts,
            grounding=grounding,
        )

    sentences = [fact.text.rstrip(".") + "." for fact in supported_facts[:_MAX_COMPOSED_FACTS]]
    if grounding.uncertain_findings:
        sentences.append(_compose_uncertainty_note(grounding))
    return " ".join(sentences).strip()


def _compose_age_answer(grounding: SearchGroundingContext) -> str | None:
    if grounding.birth_date is None:
        if grounding.uncertain_findings:
            return (
                "I found some biographical details, but I could not confirm a birth "
                "date strongly enough to compute a current age."
            )
        return None

    age = _compute_age(grounding.birth_date, grounding.current_date)
    return (
        f"I found a birth date of {grounding.birth_date.isoformat()}. "
        f"On {grounding.current_date.isoformat()}, that makes them {age} years old."
    )


def _compose_person_answer(
    *,
    supported_facts: Sequence[SearchGroundingFinding],
    grounding: SearchGroundingContext,
) -> str:
    selected_facts = list(supported_facts[:_MAX_COMPOSED_FACTS])
    sentences = [fact.text.rstrip(".") + "." for fact in selected_facts]
    if grounding.uncertain_findings:
        sentences.append(_compose_uncertainty_note(grounding))
    return " ".join(sentences).strip()


def _compose_uncertain_only_answer(grounding: SearchGroundingContext) -> str:
    if any(
        _HANDLE_HINT_PATTERN.search(finding.text) or _HANDLE_TERM_PATTERN.search(finding.text)
        for finding in grounding.uncertain_findings
    ):
        return (
            "I found some biographical details, but I did not see consistent support "
            "for specific social handles or usernames."
        )
    return (
        "I found a few mentions, but the details were not consistently confirmed "
        "across the retrieved sources."
    )


def _compose_uncertainty_note(grounding: SearchGroundingContext) -> str:
    if any(
        _HANDLE_HINT_PATTERN.search(finding.text) or _HANDLE_TERM_PATTERN.search(finding.text)
        for finding in grounding.uncertain_findings
    ):
        return (
            "Some profile details, including possible social handles, were not "
            "consistently confirmed."
        )
    return "Some lower-confidence details were omitted because the sources did not confirm them consistently."


def _select_supported_facts(
    grounding: SearchGroundingContext,
) -> tuple[SearchGroundingFinding, ...]:
    supported = list(grounding.supported_findings)
    supported.sort(
        key=lambda finding: (
            0 if finding.confidence == "strong" else 1,
            -finding.support_count,
            -finding.score,
            finding.text.casefold(),
        )
    )

    selected: list[SearchGroundingFinding] = []
    selected_tokens: list[set[str]] = []
    identity_facts = [fact for fact in supported if _looks_like_identity_fact(fact.text)]
    other_facts = [fact for fact in supported if fact not in identity_facts]
    ordered_facts = identity_facts + other_facts

    for fact in ordered_facts:
        fact_tokens = set(_content_tokens(fact.text))
        if fact_tokens and any(
            _overlap_ratio(fact_tokens, existing_tokens) >= 0.7
            for existing_tokens in selected_tokens
        ):
            continue
        selected.append(fact)
        selected_tokens.append(fact_tokens)
        if len(selected) >= _MAX_COMPOSED_FACTS:
            break

    return tuple(selected)


def _looks_like_identity_fact(text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:is|was|est|etait|était)\b",
            text,
            flags=re.IGNORECASE,
        )
    )


def _query_is_person_profile(query: str) -> bool:
    folded_query = _fold_for_match(query)
    return any(pattern.search(folded_query) for pattern in _PERSON_QUERY_PATTERNS)


def _query_is_age_request(query: str) -> bool:
    folded_query = _fold_for_match(query)
    return any(pattern.search(folded_query) for pattern in _AGE_QUERY_PATTERNS)


def _compute_age(birth_date: date, current_date: date) -> int:
    age = current_date.year - birth_date.year
    if (current_date.month, current_date.day) < (birth_date.month, birth_date.day):
        age -= 1
    return age


def _fold_for_match(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.casefold())
    without_accents = "".join(
        character
        for character in normalized
        if not unicodedata.combining(character)
    )
    return " ".join(re.findall(r"[a-z0-9]+", without_accents))


def _content_tokens(text: str) -> tuple[str, ...]:
    return tuple(
        token for token in _fold_for_match(text).split() if len(token) > 2 or token.isdigit()
    )


def _overlap_ratio(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    intersection = len(left & right)
    return intersection / min(len(left), len(right))


def _is_search_tool_message(message: ChatMessage) -> bool:
    return (
        message.role is MessageRole.TOOL
        and message.content.startswith(_SEARCH_TOOL_PREFIX)
    )


__all__ = [
    "SearchGroundingContext",
    "SearchGroundingFinding",
    "build_search_answer_contract",
    "build_search_grounding_context",
    "build_search_tool_history_summary",
    "has_search_grounding_context",
    "parse_search_tool_history",
    "shape_search_backed_reply",
    "should_apply_search_grounding",
]
