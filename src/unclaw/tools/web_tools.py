"""Lightweight web tools for the early Unclaw runtime."""

from __future__ import annotations

import ipaddress
import re
import socket
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, unquote, urlencode, urljoin, urlparse, urlunparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

from unclaw.tools.contracts import (
    ToolCall,
    ToolDefinition,
    ToolPermissionLevel,
    ToolResult,
)
from unclaw.tools.registry import ToolRegistry

_DEFAULT_MAX_FETCH_CHARS = 8_000
_MAX_FETCH_BYTES = 1_000_000
_DEFAULT_TIMEOUT_SECONDS = 10.0
_DEFAULT_MAX_SEARCH_RESULTS = 20
_MAX_SEARCH_RESULTS = 20
_DEFAULT_MAX_SEARCH_FETCHES = 30
_DEFAULT_MAX_CRAWL_DEPTH = 2
_MAX_CHILD_LINKS_PER_PAGE = 3
_DEFAULT_SEARCH_FETCH_CHARS = 12_000
_MAX_SUMMARY_POINTS = 5
_MAX_SUMMARY_POINT_CHARS = 260
_MAX_SOURCE_NOTE_CHARS = 220
_MAX_PAGE_EVIDENCE_ITEMS = 3
_MAX_KEPT_EVIDENCE_ITEMS = 8
_MAX_OUTPUT_SOURCES = 8
_DUCKDUCKGO_HTML_SEARCH_URL = "https://html.duckduckgo.com/html/"
_SEARCH_PROVIDER_NAME = "DuckDuckGo HTML"
_LOW_VALUE_RESULT_TITLES = {"accueil", "home", "homepage", "index"}
_ARTICLE_PATH_CUES = frozenset(
    {
        "analysis",
        "article",
        "articles",
        "blog",
        "blogs",
        "entry",
        "feature",
        "features",
        "post",
        "posts",
        "recap",
        "report",
        "reports",
        "story",
        "stories",
        "update",
        "updates",
    }
)
_LIVE_STREAMING_PATH_CUES = frozenset(
    {
        "direct",
        "directs",
        "emission",
        "emissions",
        "en-direct",
        "live",
        "player",
        "programme",
        "programmes",
        "regarder",
        "replay",
        "stream",
        "streaming",
        "tv",
        "watch",
    }
)
_HUB_PATH_CUES = frozenset(
    {
        "archive",
        "archives",
        "category",
        "categories",
        "index",
        "latest",
        "listing",
        "live",
        "page",
        "section",
        "tag",
        "tags",
        "topics",
        "updates",
    }
)
_LOW_VALUE_PATH_CUES = frozenset(
    {
        "about",
        "account",
        "contact",
        "donate",
        "help",
        "join",
        "legal",
        "login",
        "logout",
        "privacy",
        "register",
        "settings",
        "share",
        "signin",
        "signup",
        "subscribe",
        "support",
        "terms",
    }
)
_GENERIC_LINK_TEXTS = frozenset(
    {
        "continue reading",
        "home",
        "learn more",
        "menu",
        "more",
        "next",
        "older posts",
        "previous",
        "read more",
        "see more",
        "view more",
    }
)
_MATCH_BOILERPLATE_PREFIXES = (
    "all rights reserved",
    "cookie ",
    "copyright ",
    "menu ",
    "sign in",
    "skip to",
)
_NOISE_SIGNAL_PHRASES = frozenset(
    {
        "accept cookies",
        "accepter les cookies",
        "acceder aux notifications",
        "account settings",
        "all rights reserved",
        "already a subscriber",
        "article reserve",
        "conditions generales",
        "consentement",
        "contenu reserve",
        "cookie policy",
        "cookie preferences",
        "creer un compte",
        "data protection",
        "deja abonne",
        "deja inscrit",
        "donnees personnelles",
        "en poursuivant",
        "en savoir plus et gerer",
        "gerer les cookies",
        "gerer mes preferences",
        "inscription gratuite",
        "log in to",
        "manage preferences",
        "manage your subscription",
        "mon compte",
        "newsletter signup",
        "nos partenaires",
        "notre politique",
        "nous utilisons des cookies",
        "offre numerique",
        "offre speciale",
        "parametres de confidentialite",
        "partenaires data",
        "politique de confidentialite",
        "privacy policy",
        "privacy settings",
        "profitez de",
        "se connecter",
        "sign in to",
        "sign up for",
        "subscribe to",
        "subscription required",
        "terms of service",
        "tous droits reserves",
        "use of cookies",
        "utilisation de cookies",
        "votre abonnement",
        "votre consentement",
        "your privacy",
        "your subscription",
    }
)
_QUERY_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "au",
        "aujourdhui",
        "aujourd",
        "ce",
        "ces",
        "cette",
        "d",
        "de",
        "des",
        "du",
        "en",
        "est",
        "et",
        "for",
        "il",
        "is",
        "hui",
        "la",
        "le",
        "les",
        "l",
        "of",
        "on",
        "ou",
        "pour",
        "q",
        "qu",
        "que",
        "quelle",
        "quelles",
        "quel",
        "quels",
        "s",
        "the",
        "to",
        "today",
        "un",
        "une",
        "what",
    }
)
_COPULAR_TOKENS = frozenset(
    {
        "am",
        "are",
        "be",
        "been",
        "being",
        "est",
        "etaient",
        "etait",
        "furent",
        "is",
        "sera",
        "seront",
        "sont",
        "was",
        "were",
    }
)
_BLOCKED_FETCH_HOSTS = {
    "instance-data",
    "instance-data.ec2.internal",
    "localhost",
    "localhost.localdomain",
    "metadata",
    "metadata.google.internal",
}
_BLOCKED_FETCH_IPS = {"100.100.100.200"}
_BLOCK_TAGS = {
    "article",
    "aside",
    "blockquote",
    "br",
    "div",
    "footer",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "header",
    "li",
    "main",
    "nav",
    "p",
    "section",
    "tr",
}
_IGNORED_TAGS = {"noscript", "script", "style"}
_LOW_VALUE_EXTENSIONS = (
    ".css",
    ".ico",
    ".jpg",
    ".jpeg",
    ".js",
    ".json",
    ".pdf",
    ".png",
    ".rss",
    ".svg",
    ".xml",
    ".zip",
)

FETCH_URL_TEXT_DEFINITION = ToolDefinition(
    name="fetch_url_text",
    description="Fetch a URL and extract readable text content.",
    permission_level=ToolPermissionLevel.NETWORK,
    arguments={
        "url": "HTTP or HTTPS URL to fetch.",
        "max_chars": "Optional maximum number of characters to return.",
        "timeout_seconds": "Optional request timeout in seconds.",
    },
)

SEARCH_WEB_DEFINITION = ToolDefinition(
    name="search_web",
    description=(
        "Search the public web with bounded iterative retrieval and return a compact summary."
    ),
    permission_level=ToolPermissionLevel.NETWORK,
    arguments={
        "query": "Plain-language search query.",
        "max_results": "Optional maximum number of initial search results to consider, between 1 and 20.",
        "timeout_seconds": "Optional request timeout in seconds.",
    },
)


def register_web_tools(
    registry: ToolRegistry,
    *,
    allow_private_networks: bool = False,
) -> None:
    """Register the built-in lightweight web tools."""

    def fetch_handler(call: ToolCall) -> ToolResult:
        return fetch_url_text(
            call,
            allow_private_networks=allow_private_networks,
        )

    def search_handler(call: ToolCall) -> ToolResult:
        return search_web(call)

    registry.register(FETCH_URL_TEXT_DEFINITION, fetch_handler)
    registry.register(SEARCH_WEB_DEFINITION, search_handler)


def fetch_url_text(
    call: ToolCall,
    *,
    allow_private_networks: bool = False,
) -> ToolResult:
    """Fetch a URL and return a compact text version of the response body."""
    tool_name = FETCH_URL_TEXT_DEFINITION.name

    try:
        url = _read_string_argument(call.arguments, "url")
        max_chars = _read_positive_int_argument(
            call.arguments,
            "max_chars",
            default=_DEFAULT_MAX_FETCH_CHARS,
        )
        timeout_seconds = _read_positive_number_argument(
            call.arguments,
            "timeout_seconds",
            default=_DEFAULT_TIMEOUT_SECONDS,
        )
    except ValueError as exc:
        return ToolResult.failure(tool_name=tool_name, error=str(exc))

    if not _is_supported_url(url):
        return ToolResult.failure(
            tool_name=tool_name,
            error="Argument 'url' must be a valid HTTP or HTTPS URL.",
        )

    try:
        document = _fetch_text_document(
            url,
            max_chars=max_chars,
            timeout_seconds=timeout_seconds,
            allow_private_networks=allow_private_networks,
            accept_header=(
                "text/plain, text/html, application/json;q=0.9, */*;q=0.1"
            ),
        )
    except ValueError as exc:
        return ToolResult.failure(tool_name=tool_name, error=str(exc))
    except _BlockedFetchTargetError as exc:
        return ToolResult.failure(tool_name=tool_name, error=str(exc))
    except HTTPError as exc:
        return ToolResult.failure(
            tool_name=tool_name,
            error=f"HTTP error {exc.code} while fetching '{url}': {exc.reason}",
        )
    except URLError as exc:
        return ToolResult.failure(
            tool_name=tool_name,
            error=f"Could not fetch '{url}': {exc.reason}",
        )
    except OSError as exc:
        return ToolResult.failure(
            tool_name=tool_name,
            error=f"Could not fetch '{url}': {exc}",
        )

    output_text = "\n".join(
        [
            f"URL: {document.resolved_url}",
            f"Status: {document.status_code or 'unknown'}",
            f"Content-Type: {document.content_type}",
            "",
            _format_text_excerpt(
                document.text_excerpt,
                truncated=document.truncated,
            ),
        ]
    )
    return ToolResult.ok(
        tool_name=tool_name,
        output_text=output_text,
        payload={
            "requested_url": url,
            "resolved_url": document.resolved_url,
            "status_code": document.status_code,
            "content_type": document.content_type,
            "truncated": document.truncated,
        },
    )


def search_web(call: ToolCall) -> ToolResult:
    """Search the public web, iteratively fetch bounded sources, and summarize evidence."""
    tool_name = SEARCH_WEB_DEFINITION.name

    try:
        query = _read_string_argument(call.arguments, "query")
        max_results = _read_limited_positive_int_argument(
            call.arguments,
            "max_results",
            default=_DEFAULT_MAX_SEARCH_RESULTS,
            maximum=_MAX_SEARCH_RESULTS,
        )
        timeout_seconds = _read_positive_number_argument(
            call.arguments,
            "timeout_seconds",
            default=_DEFAULT_TIMEOUT_SECONDS,
        )
    except ValueError as exc:
        return ToolResult.failure(tool_name=tool_name, error=str(exc))

    try:
        response_text = _search_public_web(
            query=query,
            timeout_seconds=timeout_seconds,
        )
    except _BlockedFetchTargetError as exc:
        return ToolResult.failure(tool_name=tool_name, error=str(exc))
    except HTTPError as exc:
        return ToolResult.failure(
            tool_name=tool_name,
            error=(
                f"Search provider returned HTTP error {exc.code} "
                f"for '{query}': {exc.reason}"
            ),
        )
    except URLError as exc:
        return ToolResult.failure(
            tool_name=tool_name,
            error=f"Could not search the web for '{query}': {exc.reason}",
        )
    except OSError as exc:
        return ToolResult.failure(
            tool_name=tool_name,
            error=f"Could not search the web for '{query}': {exc}",
        )
    except ValueError as exc:
        return ToolResult.failure(tool_name=tool_name, error=str(exc))

    search_query = _build_search_query(query)
    raw_results = _parse_duckduckgo_html_results(response_text, max_results=max_results)
    ranked_results = _rank_search_results(
        _deduplicate_search_results(raw_results),
        query=search_query,
    )
    outcome = _run_iterative_retrieval(
        results=ranked_results,
        query=search_query,
        timeout_seconds=timeout_seconds,
        budget=_RetrievalBudget(max_initial_results=max_results),
    )
    synthesis = _synthesize_search_knowledge(
        outcome.evidence_items,
        query=search_query,
    )
    summary_points = tuple(finding.text for finding in synthesis.findings)
    output_text = _format_search_results(
        query=query,
        outcome=outcome,
        summary_points=summary_points,
        synthesis=synthesis,
    )

    return ToolResult.ok(
        tool_name=tool_name,
        output_text=output_text,
        payload={
            "query": query,
            "provider": _SEARCH_PROVIDER_NAME,
            "initial_result_count": outcome.initial_result_count,
            "considered_candidate_count": outcome.considered_candidate_count,
            "fetch_attempt_count": outcome.fetch_attempt_count,
            "fetch_success_count": outcome.fetch_success_count,
            "evidence_count": len(outcome.evidence_items),
            "statement_count": len(synthesis.statements),
            "fact_cluster_count": len(synthesis.fact_clusters),
            "finding_count": len(synthesis.findings),
            "summary_points": list(summary_points),
            "synthesized_findings": [
                {
                    "text": finding.text,
                    "score": finding.score,
                    "support_count": finding.support_count,
                    "source_titles": list(finding.source_titles),
                    "source_urls": list(finding.source_urls),
                }
                for finding in synthesis.findings
            ],
            "results": [
                {
                    "title": source.title,
                    "url": source.url,
                    "takeaway": source.takeaway,
                    "depth": source.depth,
                    "fetched": source.fetched,
                    "evidence_count": source.evidence_count,
                    "fetch_error": source.fetch_error,
                    "used_snippet_fallback": source.used_snippet_fallback,
                    "usefulness": source.usefulness,
                }
                for source in outcome.sources
            ],
            "evidence": [
                {
                    "text": evidence.text,
                    "url": evidence.url,
                    "source_title": evidence.source_title,
                    "score": evidence.score,
                    "depth": evidence.depth,
                    "query_relevance": evidence.query_relevance,
                    "evidence_quality": evidence.evidence_quality,
                    "novelty": evidence.novelty,
                    "supporting_urls": list(evidence.supporting_urls),
                    "supporting_titles": list(evidence.supporting_titles),
                }
                for evidence in outcome.evidence_items
            ],
        },
    )


class _BlockedFetchTargetError(ValueError):
    """Raised when a fetch target is blocked by the safe default policy."""


class _SafeRedirectHandler(HTTPRedirectHandler):
    """Reject redirects that escape the public-network fetch policy."""

    def __init__(self, *, allow_private_networks: bool) -> None:
        super().__init__()
        self._allow_private_networks = allow_private_networks

    def redirect_request(
        self,
        req: Request,
        fp,  # type: ignore[no-untyped-def]
        code: int,
        msg: str,
        headers,
        newurl: str,
    ) -> Request | None:
        _ensure_fetch_target_allowed(
            newurl,
            allow_private_networks=self._allow_private_networks,
        )
        return super().redirect_request(req, fp, code, msg, headers, newurl)


@dataclass(slots=True)
class _HTMLLinkBuilder:
    """Collect one anchor while parsing HTML."""

    href: str
    text_parts: list[str] = field(default_factory=list)


class _HTMLPageExtractor(HTMLParser):
    """Collect readable text, title, and anchor links from a basic HTML document."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._ignored_depth = 0
        self._title_depth = 0
        self._parts: list[str] = []
        self._title_parts: list[str] = []
        self._current_link: _HTMLLinkBuilder | None = None
        self._links: list[_HTMLLink] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        attributes = dict(attrs)
        if tag in _IGNORED_TAGS:
            self._ignored_depth += 1
            return
        if self._ignored_depth > 0:
            return
        if tag == "title":
            self._title_depth += 1
        if tag == "a":
            self._parts.append("\n")
            self._current_link = _HTMLLinkBuilder(href=attributes.get("href") or "")
        if tag in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _IGNORED_TAGS:
            if self._ignored_depth > 0:
                self._ignored_depth -= 1
            return
        if self._ignored_depth > 0:
            return
        if tag == "title" and self._title_depth > 0:
            self._title_depth -= 1
        if tag == "a" and self._current_link is not None:
            href = self._current_link.href.strip()
            text = _normalize_text(" ".join(self._current_link.text_parts))
            if href:
                self._links.append(_HTMLLink(url=href, text=text))
            self._current_link = None
            self._parts.append("\n")
        if tag in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._ignored_depth > 0:
            return
        text = data.strip()
        if not text:
            return
        self._parts.append(data)
        if self._title_depth > 0:
            self._title_parts.append(text)
        if self._current_link is not None:
            self._current_link.text_parts.append(text)

    @property
    def title(self) -> str:
        return _normalize_text(" ".join(self._title_parts))

    @property
    def text(self) -> str:
        return _normalize_text("".join(self._parts))

    @property
    def links(self) -> tuple[_HTMLLink, ...]:
        return tuple(self._links)


@dataclass(slots=True)
class _SearchResultBuilder:
    """Collect one parsed DuckDuckGo HTML result block."""

    url: str
    title_parts: list[str] = field(default_factory=list)
    snippet_parts: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class _HTMLLink:
    """One normalized link extracted from a fetched HTML page."""

    url: str
    text: str


@dataclass(frozen=True, slots=True)
class _RawFetchedDocument:
    """Decoded network response body and basic metadata."""

    requested_url: str
    resolved_url: str
    status_code: int | None
    content_type: str
    decoded_text: str


@dataclass(slots=True)
class _FetchedTextDocument:
    """Compact extracted text payload for one fetched public URL."""

    requested_url: str
    resolved_url: str
    status_code: int | None
    content_type: str
    text_excerpt: str
    truncated: bool


@dataclass(frozen=True, slots=True)
class _SearchQuery:
    """Normalized query tokens for generic retrieval scoring."""

    raw_query: str
    normalized_query: str
    keyword_tokens: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _SearchCandidate:
    """One URL candidate waiting to be fetched."""

    url: str
    title: str
    snippet: str
    depth: int
    priority: float
    order: int
    parent_url: str | None = None
    anchor_text: str = ""


@dataclass(frozen=True, slots=True)
class _FetchedSearchPage:
    """Fetched page content enriched with extracted links for retrieval."""

    requested_url: str
    resolved_url: str
    status_code: int | None
    content_type: str
    title: str
    text: str
    truncated: bool
    links: tuple[_HTMLLink, ...]


@dataclass(frozen=True, slots=True)
class _EvidenceItem:
    """One compact evidence unit retained for summary synthesis."""

    text: str
    url: str
    source_title: str
    score: float
    depth: int
    query_relevance: float
    evidence_quality: float
    novelty: float
    supporting_urls: tuple[str, ...]
    supporting_titles: tuple[str, ...]
    signature_tokens: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _EvidenceStatement:
    """One sentence-level evidence statement used for synthesis clustering."""

    text: str
    url: str
    source_title: str
    depth: int
    score: float
    query_relevance: float
    evidence_quality: float
    novelty: float
    signature_tokens: tuple[str, ...]
    content_tokens: tuple[str, ...]
    subject_tokens: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _FactCluster:
    """One aggregated cluster of overlapping evidence statements."""

    merged_text: str
    evidence: tuple[_EvidenceStatement, ...]
    supporting_urls: tuple[str, ...]
    source_titles: tuple[str, ...]
    score: float
    query_relevance: float
    evidence_quality: float
    novelty: float
    support_count: int


@dataclass(frozen=True, slots=True)
class _SynthesizedFinding:
    """One user-facing finding built from an aggregated fact cluster."""

    text: str
    score: float
    support_count: int
    source_titles: tuple[str, ...]
    source_urls: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _SynthesisOutcome:
    """Temporary synthesized knowledge built from extracted evidence."""

    statements: tuple[_EvidenceStatement, ...]
    fact_clusters: tuple[_FactCluster, ...]
    findings: tuple[_SynthesizedFinding, ...]


@dataclass(slots=True)
class _RetrievedSource:
    """One source considered during iterative retrieval."""

    title: str
    url: str
    depth: int
    fetched: bool
    takeaway: str
    usefulness: float
    evidence_count: int
    fetch_error: str | None = None
    used_snippet_fallback: bool = False
    relevance: float = 0.0
    density: float = 0.0
    novelty: float = 0.0
    hub_score: float = 0.0
    child_link_count: int = 0


@dataclass(frozen=True, slots=True)
class _PageScores:
    """Generic scoring signals for one fetched page."""

    relevance: float
    density: float
    novelty: float
    usefulness: float
    hub_score: float
    terminal_score: float
    informative_passage_count: int
    internal_link_count: int


@dataclass(frozen=True, slots=True)
class _RetrievalBudget:
    """Bounded retrieval budgets to keep /search deterministic and lightweight."""

    max_initial_results: int = _DEFAULT_MAX_SEARCH_RESULTS
    max_total_fetches: int = _DEFAULT_MAX_SEARCH_FETCHES
    max_depth: int = _DEFAULT_MAX_CRAWL_DEPTH
    max_child_links_per_page: int = _MAX_CHILD_LINKS_PER_PAGE
    max_kept_evidence_items: int = _MAX_KEPT_EVIDENCE_ITEMS


@dataclass(frozen=True, slots=True)
class _RetrievalOutcome:
    """Final retrieval state used to build tool output."""

    initial_result_count: int
    considered_candidate_count: int
    fetch_attempt_count: int
    fetch_success_count: int
    evidence_items: tuple[_EvidenceItem, ...]
    sources: tuple[_RetrievedSource, ...]


class _DuckDuckGoHTMLSearchParser(HTMLParser):
    """Parse compact search results from DuckDuckGo's HTML endpoint."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._current_result: _SearchResultBuilder | None = None
        self._capture_target: tuple[str, str] | None = None
        self._results: list[dict[str, str]] = []

    @property
    def results(self) -> list[dict[str, str]]:
        return self._results

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        attributes = dict(attrs)
        classes = set((attributes.get("class") or "").split())

        if tag == "a" and "result__a" in classes:
            self._finalize_current_result()
            self._current_result = _SearchResultBuilder(
                url=_normalize_search_result_url(attributes.get("href"))
            )
            self._capture_target = ("title", tag)
            return

        if self._current_result is not None and "result__snippet" in classes:
            self._capture_target = ("snippet", tag)

    def handle_endtag(self, tag: str) -> None:
        if self._capture_target is None:
            return
        _kind, captured_tag = self._capture_target
        if tag == captured_tag:
            self._capture_target = None

    def handle_data(self, data: str) -> None:
        if self._current_result is None or self._capture_target is None:
            return

        text = data.strip()
        if not text:
            return

        kind, _tag = self._capture_target
        if kind == "title":
            self._current_result.title_parts.append(text)
            return

        self._current_result.snippet_parts.append(text)

    def close(self) -> None:
        super().close()
        self._finalize_current_result()

    def _finalize_current_result(self) -> None:
        if self._current_result is None:
            return

        title = _normalize_text(" ".join(self._current_result.title_parts))
        url = self._current_result.url.strip()
        snippet = _normalize_text(" ".join(self._current_result.snippet_parts))

        if title and url:
            self._results.append(
                {
                    "title": title,
                    "url": url,
                    "snippet": snippet,
                }
            )

        self._capture_target = None
        self._current_result = None


def _decode_content(raw_content: bytes, charset: str) -> str:
    try:
        return raw_content.decode(charset)
    except (LookupError, UnicodeDecodeError):
        return raw_content.decode("utf-8", errors="replace")


def _extract_page_content(
    content: str,
    content_type: str,
) -> tuple[str, str, tuple[_HTMLLink, ...]] | None:
    if content_type in {"text/html", "application/xhtml+xml"}:
        parser = _HTMLPageExtractor()
        parser.feed(content)
        parser.close()
        return (parser.title, parser.text, parser.links)

    if _is_text_content_type(content_type):
        return ("", _normalize_text(content), ())

    return None


def _normalize_text(text: str) -> str:
    normalized_lines: list[str] = []
    previous_blank = False

    for raw_line in text.splitlines():
        line = " ".join(raw_line.split())
        if not line:
            if normalized_lines and not previous_blank:
                normalized_lines.append("")
            previous_blank = True
            continue

        normalized_lines.append(line)
        previous_blank = False

    return "\n".join(normalized_lines).strip()


def _parse_duckduckgo_html_results(
    response_text: str,
    *,
    max_results: int,
) -> list[dict[str, str]]:
    parser = _DuckDuckGoHTMLSearchParser()
    parser.feed(response_text)
    parser.close()
    return parser.results[:max_results]


def _search_public_web(*, query: str, timeout_seconds: float) -> str:
    request_url = f"{_DUCKDUCKGO_HTML_SEARCH_URL}?{urlencode({'q': query})}"
    _ensure_fetch_target_allowed(
        request_url,
        allow_private_networks=False,
    )

    request = Request(
        request_url,
        headers={
            "User-Agent": "unclaw/0.1 (+https://local-first.invalid)",
            "Accept": "text/html, application/xhtml+xml;q=0.9, */*;q=0.1",
        },
    )

    with _open_request(
        request,
        timeout_seconds=timeout_seconds,
        allow_private_networks=False,
    ) as response:
        content_type = response.headers.get_content_type()
        charset = response.headers.get_content_charset() or "utf-8"
        raw_content = response.read(_MAX_FETCH_BYTES + 1)

    if len(raw_content) > _MAX_FETCH_BYTES:
        raw_content = raw_content[:_MAX_FETCH_BYTES]

    if content_type not in {"text/html", "application/xhtml+xml"}:
        raise ValueError(
            f"Search provider returned unsupported content type: {content_type}"
        )

    return _decode_content(raw_content, charset)


def _fetch_raw_document(
    url: str,
    *,
    timeout_seconds: float,
    allow_private_networks: bool,
    accept_header: str,
) -> _RawFetchedDocument:
    _ensure_fetch_target_allowed(
        url,
        allow_private_networks=allow_private_networks,
    )

    request = Request(
        url,
        headers={
            "User-Agent": "unclaw/0.1 (+https://local-first.invalid)",
            "Accept": accept_header,
        },
    )

    with _open_request(
        request,
        timeout_seconds=timeout_seconds,
        allow_private_networks=allow_private_networks,
    ) as response:
        content_type = response.headers.get_content_type()
        charset = response.headers.get_content_charset() or "utf-8"
        status_code = getattr(response, "status", None)
        resolved_url = response.geturl()
        raw_content = response.read(_MAX_FETCH_BYTES + 1)

    if len(raw_content) > _MAX_FETCH_BYTES:
        raw_content = raw_content[:_MAX_FETCH_BYTES]

    return _RawFetchedDocument(
        requested_url=url,
        resolved_url=resolved_url,
        status_code=status_code,
        content_type=content_type,
        decoded_text=_decode_content(raw_content, charset),
    )


def _fetch_text_document(
    url: str,
    *,
    max_chars: int,
    timeout_seconds: float,
    allow_private_networks: bool,
    accept_header: str,
) -> _FetchedTextDocument:
    raw_document = _fetch_raw_document(
        url,
        timeout_seconds=timeout_seconds,
        allow_private_networks=allow_private_networks,
        accept_header=accept_header,
    )
    extracted_content = _extract_page_content(
        raw_document.decoded_text,
        raw_document.content_type,
    )
    if extracted_content is None:
        raise ValueError(
            f"Unsupported content type for text extraction: {raw_document.content_type}"
        )

    _title, extracted_text, _links = extracted_content
    if not extracted_text:
        extracted_text = "[empty response body]"

    truncated = len(extracted_text) > max_chars
    text_excerpt = extracted_text[:max_chars].rstrip() if truncated else extracted_text
    return _FetchedTextDocument(
        requested_url=url,
        resolved_url=raw_document.resolved_url,
        status_code=raw_document.status_code,
        content_type=raw_document.content_type,
        text_excerpt=text_excerpt,
        truncated=truncated,
    )


def _fetch_search_page(
    url: str,
    *,
    max_chars: int,
    timeout_seconds: float,
) -> _FetchedSearchPage:
    raw_document = _fetch_raw_document(
        url,
        timeout_seconds=timeout_seconds,
        allow_private_networks=False,
        accept_header="text/plain, text/html, application/json;q=0.9, */*;q=0.1",
    )
    extracted_content = _extract_page_content(
        raw_document.decoded_text,
        raw_document.content_type,
    )
    if extracted_content is None:
        raise ValueError(
            f"Unsupported content type for text extraction: {raw_document.content_type}"
        )

    page_title, extracted_text, links = extracted_content
    if not extracted_text:
        extracted_text = "[empty response body]"

    truncated = len(extracted_text) > max_chars
    page_text = extracted_text[:max_chars].rstrip() if truncated else extracted_text
    return _FetchedSearchPage(
        requested_url=url,
        resolved_url=raw_document.resolved_url,
        status_code=raw_document.status_code,
        content_type=raw_document.content_type,
        title=page_title,
        text=page_text,
        truncated=truncated,
        links=links,
    )


def _format_text_excerpt(text: str, *, truncated: bool) -> str:
    if not truncated:
        return text
    return f"{text}\n\n[truncated]"


def _build_search_query(query: str) -> _SearchQuery:
    keyword_tokens = tuple(
        token
        for token in _text_tokens(query)
        if len(token) >= 3 and token not in _QUERY_STOPWORDS
    )
    if not keyword_tokens:
        keyword_tokens = tuple(token for token in _text_tokens(query) if len(token) >= 2)
    return _SearchQuery(
        raw_query=query,
        normalized_query=_fold_for_match(query),
        keyword_tokens=keyword_tokens,
    )


def _deduplicate_search_results(results: list[dict[str, str]]) -> list[dict[str, str]]:
    deduplicated: list[dict[str, str]] = []
    seen_urls: set[str] = set()

    for result in results:
        url = result["url"].strip()
        canonical_url = _canonicalize_url(url)
        if not url or not canonical_url or canonical_url in seen_urls:
            continue
        deduplicated.append(dict(result))
        seen_urls.add(canonical_url)

    return deduplicated


def _rank_search_results(
    results: list[dict[str, str]],
    *,
    query: _SearchQuery,
) -> list[dict[str, str]]:
    scored_results: list[tuple[float, int, dict[str, str]]] = []

    for index, result in enumerate(results):
        score = _score_search_result(result, query=query)
        scored_results.append((score, index, dict(result)))

    scored_results.sort(key=lambda item: (-item[0], item[1]))
    return [result for _score, _index, result in scored_results]


def _score_search_result(
    result: Mapping[str, str],
    *,
    query: _SearchQuery,
) -> float:
    title = result.get("title", "")
    snippet = result.get("snippet", "")
    url = result.get("url", "")
    parsed_url = urlparse(url)
    folded_metadata = _fold_for_match(f"{title} {snippet} {url}")
    path_segments = _url_path_segments(url)
    score = 0.0

    score += _keyword_overlap_score(folded_metadata, query.keyword_tokens) * 3.0
    score += min(len(snippet.split()) / 12.0, 2.0)
    score += min(len(path_segments), 3)

    if _url_looks_article_like(url):
        score += 4.0
    if _url_looks_archive_like(url):
        score += 1.0
    if _url_looks_homepage_like(url):
        score -= 6.0
    if _url_looks_live_or_streaming(url):
        score -= 5.0
    if _looks_generic_result_title(title=title, hostname=parsed_url.hostname or ""):
        score -= 3.0
    if _passage_has_noise_signals(snippet):
        score -= 2.0

    return score


def _run_iterative_retrieval(
    *,
    results: list[dict[str, str]],
    query: _SearchQuery,
    timeout_seconds: float,
    budget: _RetrievalBudget,
) -> _RetrievalOutcome:
    queue: list[_SearchCandidate] = []
    considered_candidate_urls: set[str] = set()
    sources_by_url: dict[str, _RetrievedSource] = {}
    evidence_items: list[_EvidenceItem] = []
    initial_results = results[: budget.max_initial_results]
    order = 0

    for result in initial_results:
        candidate = _SearchCandidate(
            url=result["url"],
            title=result["title"],
            snippet=result.get("snippet", ""),
            depth=1,
            priority=_score_search_result(result, query=query),
            order=order,
        )
        order += 1
        canonical_url = _canonicalize_url(candidate.url)
        if not canonical_url or canonical_url in considered_candidate_urls:
            continue
        queue.append(candidate)
        considered_candidate_urls.add(canonical_url)

    fetch_attempt_count = 0
    fetch_success_count = 0

    while queue and fetch_attempt_count < budget.max_total_fetches:
        queue.sort(key=lambda item: (-item.priority, item.depth, item.order))
        candidate = queue.pop(0)
        candidate_key = _canonicalize_url(candidate.url)
        if not candidate_key:
            continue

        fetch_attempt_count += 1

        try:
            page = _fetch_search_page(
                candidate.url,
                max_chars=_DEFAULT_SEARCH_FETCH_CHARS,
                timeout_seconds=timeout_seconds,
            )
        except (_BlockedFetchTargetError, HTTPError, URLError, OSError, ValueError) as exc:
            fetch_error = _summarize_source_fetch_error(exc)
            fallback_evidence = _build_snippet_evidence(
                snippet=candidate.snippet,
                url=candidate.url,
                title=_select_source_title("", candidate.title, candidate.anchor_text),
                depth=candidate.depth,
                query=query,
                existing_evidence=evidence_items,
            )
            kept_count = 0
            takeaway = _build_fallback_takeaway(
                snippet=candidate.snippet,
                fetch_error=fetch_error,
            )
            if fallback_evidence is not None:
                kept_count = int(
                    _merge_evidence_item(
                        evidence_items,
                        fallback_evidence,
                        maximum=budget.max_kept_evidence_items,
                    )
                )
                takeaway = fallback_evidence.text

            sources_by_url[candidate_key] = _RetrievedSource(
                title=_select_source_title("", candidate.title, candidate.anchor_text),
                url=candidate.url,
                depth=candidate.depth,
                fetched=False,
                takeaway=takeaway,
                usefulness=max(candidate.priority - 1.0, 0.0),
                evidence_count=kept_count,
                fetch_error=fetch_error,
                used_snippet_fallback=fallback_evidence is not None,
            )
            continue

        fetch_success_count += 1
        page_key = _canonicalize_url(page.resolved_url)
        if not page_key:
            page_key = candidate_key

        source_title = _select_source_title(page.title, candidate.title, candidate.anchor_text)
        evidence_candidates = _extract_page_evidence(
            text=page.text,
            title=source_title,
            url=page.resolved_url,
            depth=candidate.depth,
            query=query,
            existing_evidence=evidence_items,
        )
        page_scores = _score_fetched_page(
            page=page,
            title=source_title,
            query=query,
            evidence_candidates=evidence_candidates,
            existing_evidence=evidence_items,
        )
        kept_count = 0
        for evidence_candidate in evidence_candidates:
            kept_count += int(
                _merge_evidence_item(
                    evidence_items,
                    evidence_candidate,
                    maximum=budget.max_kept_evidence_items,
                )
            )

        best_takeaway = ""
        if evidence_candidates:
            best_takeaway = _clip_text(
                evidence_candidates[0].text,
                limit=_MAX_SOURCE_NOTE_CHARS,
            )
        else:
            fallback_evidence = _build_snippet_evidence(
                snippet=candidate.snippet,
                url=page.resolved_url,
                title=source_title,
                depth=candidate.depth,
                query=query,
                existing_evidence=evidence_items,
            )
            if fallback_evidence is not None:
                kept_count += int(
                    _merge_evidence_item(
                        evidence_items,
                        fallback_evidence,
                        maximum=budget.max_kept_evidence_items,
                    )
                )
                best_takeaway = fallback_evidence.text

        if not best_takeaway:
            best_takeaway = _build_fallback_takeaway(
                snippet=candidate.snippet,
                url=page.resolved_url,
                title=source_title,
                fetched_text=page.text,
            )

        source = _RetrievedSource(
            title=source_title,
            url=page.resolved_url,
            depth=candidate.depth,
            fetched=True,
            takeaway=best_takeaway,
            usefulness=page_scores.usefulness,
            evidence_count=kept_count,
            used_snippet_fallback=not bool(evidence_candidates),
            relevance=page_scores.relevance,
            density=page_scores.density,
            novelty=page_scores.novelty,
            hub_score=page_scores.hub_score,
        )

        if candidate_key != page_key and candidate_key in sources_by_url:
            del sources_by_url[candidate_key]
        sources_by_url[page_key] = source

        if candidate.depth < budget.max_depth and _should_expand_page(page_scores):
            child_candidates = _rank_child_candidates(
                page=page,
                parent_candidate=candidate,
                query=query,
                start_order=order,
            )
            child_slots = min(
                budget.max_child_links_per_page,
                budget.max_total_fetches - fetch_attempt_count,
            )
            for child_candidate in child_candidates[:child_slots]:
                child_key = _canonicalize_url(child_candidate.url)
                if not child_key or child_key in considered_candidate_urls:
                    continue
                queue.append(child_candidate)
                considered_candidate_urls.add(child_key)
                source.child_link_count += 1
                order = max(order, child_candidate.order + 1)

        if _should_stop_retrieval(
            evidence_items=evidence_items,
            sources=sources_by_url.values(),
            fetch_attempt_count=fetch_attempt_count,
        ):
            break

    final_evidence = tuple(
        sorted(
            evidence_items,
            key=lambda item: (-item.score, item.depth, item.url, item.text.casefold()),
        )[: budget.max_kept_evidence_items]
    )
    final_sources = tuple(
        sorted(
            (
                source
                for source in sources_by_url.values()
                if _source_worth_showing(source)
            ),
            key=lambda source: (
                -source.usefulness,
                source.depth,
                source.title.casefold(),
                source.url,
            ),
        )[:_MAX_OUTPUT_SOURCES]
    )
    return _RetrievalOutcome(
        initial_result_count=len(initial_results),
        considered_candidate_count=len(considered_candidate_urls),
        fetch_attempt_count=fetch_attempt_count,
        fetch_success_count=fetch_success_count,
        evidence_items=final_evidence,
        sources=final_sources,
    )


def _extract_page_evidence(
    *,
    text: str,
    title: str,
    url: str,
    depth: int,
    query: _SearchQuery,
    existing_evidence: list[_EvidenceItem],
) -> list[_EvidenceItem]:
    candidates: list[_EvidenceItem] = []

    for passage in _iter_passages(text):
        evidence_text = _truncate_sentences(
            passage,
            max_sentences=2,
            max_chars=_MAX_SOURCE_NOTE_CHARS,
        )
        if not evidence_text:
            continue
        query_relevance = float(
            _keyword_overlap_score(_fold_for_match(evidence_text), query.keyword_tokens)
        )
        base_score = _score_evidence_text(
            evidence_text,
            title=title,
            query=query,
        )
        if base_score < 1.0:
            continue
        signature_tokens = _build_signature_tokens(evidence_text)
        novelty = _novelty_against_evidence(signature_tokens, existing_evidence)
        candidates.append(
            _EvidenceItem(
                text=evidence_text,
                url=url,
                source_title=title,
                score=base_score + novelty * 2.0,
                depth=depth,
                query_relevance=query_relevance,
                evidence_quality=max(base_score - query_relevance * 2.5, 0.0),
                novelty=novelty,
                supporting_urls=(url,),
                supporting_titles=(title,),
                signature_tokens=signature_tokens,
            )
        )

    deduplicated = _deduplicate_evidence_items(candidates)
    return deduplicated[:_MAX_PAGE_EVIDENCE_ITEMS]


def _build_snippet_evidence(
    *,
    snippet: str,
    url: str,
    title: str,
    depth: int,
    query: _SearchQuery,
    existing_evidence: list[_EvidenceItem],
) -> _EvidenceItem | None:
    normalized_snippet = _normalize_text(snippet)
    if not normalized_snippet:
        return None

    evidence_text = _clip_text(normalized_snippet, limit=_MAX_SOURCE_NOTE_CHARS)
    query_relevance = float(
        _keyword_overlap_score(_fold_for_match(evidence_text), query.keyword_tokens)
    )
    base_score = _score_evidence_text(
        evidence_text,
        title=title,
        query=query,
    )
    if base_score < 0.5:
        return None

    signature_tokens = _build_signature_tokens(evidence_text)
    novelty = _novelty_against_evidence(signature_tokens, existing_evidence)
    return _EvidenceItem(
        text=evidence_text,
        url=url,
        source_title=title,
        score=base_score + novelty,
        depth=depth,
        query_relevance=query_relevance,
        evidence_quality=max(base_score - query_relevance * 2.5, 0.0),
        novelty=novelty,
        supporting_urls=(url,),
        supporting_titles=(title,),
        signature_tokens=signature_tokens,
    )


def _score_evidence_text(
    text: str,
    *,
    title: str,
    query: _SearchQuery,
) -> float:
    if not _is_informative_passage(text, title=title):
        return -1.0

    tokens = _text_tokens(text)
    folded_text = _fold_for_match(text)
    unique_ratio = len(set(tokens)) / max(len(tokens), 1)
    score = min(len(tokens) / 12.0, 4.0)
    score += _keyword_overlap_score(folded_text, query.keyword_tokens) * 2.5
    score += unique_ratio * 2.0
    if re.search(r"\b\d+(?:[.,]\d+)?%?\b", text):
        score += 0.5
    if _looks_like_title_echo(text, title):
        score -= 2.5
    if _looks_boilerplate_text(text):
        score -= 3.0
    score -= _passage_noise_score(text)
    if _looks_site_descriptive(text):
        score -= 2.0
    if _looks_promotional(text):
        score -= 3.0
    # Penalize very short passages with no keyword overlap (vague/low-signal)
    keyword_overlap = _keyword_overlap_score(folded_text, query.keyword_tokens)
    if len(tokens) < 12 and keyword_overlap == 0:
        score -= 2.0
    return score


def _deduplicate_evidence_items(candidates: list[_EvidenceItem]) -> list[_EvidenceItem]:
    deduplicated: list[_EvidenceItem] = []
    for candidate in sorted(
        candidates,
        key=lambda item: (-item.score, item.depth, item.url, item.text.casefold()),
    ):
        if any(
            _evidence_similarity(candidate.signature_tokens, existing.signature_tokens) >= 0.8
            for existing in deduplicated
        ):
            continue
        deduplicated.append(candidate)
    return deduplicated


def _merge_evidence_item(
    evidence_items: list[_EvidenceItem],
    candidate: _EvidenceItem,
    *,
    maximum: int,
) -> bool:
    for index, existing in enumerate(evidence_items):
        similarity = _evidence_similarity(
            candidate.signature_tokens,
            existing.signature_tokens,
        )
        if similarity >= 0.8:
            merged_urls = _merge_unique_strings(
                existing.supporting_urls,
                candidate.supporting_urls,
            )
            merged_titles = _merge_unique_strings(
                existing.supporting_titles,
                candidate.supporting_titles,
            )
            preferred = candidate if candidate.score > existing.score else existing
            evidence_items[index] = _EvidenceItem(
                text=preferred.text,
                url=preferred.url,
                source_title=preferred.source_title,
                score=max(existing.score, candidate.score),
                depth=min(existing.depth, candidate.depth),
                query_relevance=max(existing.query_relevance, candidate.query_relevance),
                evidence_quality=max(
                    existing.evidence_quality,
                    candidate.evidence_quality,
                ),
                novelty=max(existing.novelty, candidate.novelty),
                supporting_urls=merged_urls,
                supporting_titles=merged_titles,
                signature_tokens=preferred.signature_tokens,
            )
            evidence_items.sort(
                key=lambda item: (-item.score, item.depth, item.url, item.text.casefold())
            )
            return True

    evidence_items.append(candidate)
    evidence_items.sort(
        key=lambda item: (-item.score, item.depth, item.url, item.text.casefold())
    )
    if len(evidence_items) > maximum:
        evidence_items[:] = evidence_items[:maximum]
    return candidate in evidence_items


def _build_signature_tokens(text: str) -> tuple[str, ...]:
    return tuple(_text_tokens(text)[:24])


def _evidence_similarity(
    left_tokens: tuple[str, ...],
    right_tokens: tuple[str, ...],
) -> float:
    if not left_tokens or not right_tokens:
        return 0.0
    left_set = set(left_tokens)
    right_set = set(right_tokens)
    union_size = len(left_set | right_set)
    if union_size == 0:
        return 0.0
    return len(left_set & right_set) / union_size


def _novelty_against_evidence(
    signature_tokens: tuple[str, ...],
    evidence_items: list[_EvidenceItem],
) -> float:
    if not signature_tokens or not evidence_items:
        return 1.0
    max_similarity = max(
        _evidence_similarity(signature_tokens, existing.signature_tokens)
        for existing in evidence_items
    )
    return max(0.0, 1.0 - max_similarity)


def _score_fetched_page(
    *,
    page: _FetchedSearchPage,
    title: str,
    query: _SearchQuery,
    evidence_candidates: list[_EvidenceItem],
    existing_evidence: list[_EvidenceItem],
) -> _PageScores:
    page_tokens = _text_tokens(page.text)
    informative_passage_count = sum(
        1
        for passage in _iter_passages(page.text)
        if _is_informative_passage(passage, title=title)
    )
    internal_link_count = sum(
        1
        for link in page.links
        if _is_internal_link(
            parent_url=page.resolved_url,
            child_url=urljoin(page.resolved_url, link.url),
        )
    )
    folded_metadata = _fold_for_match(f"{title} {page.text}")
    relevance = float(_keyword_overlap_score(folded_metadata, query.keyword_tokens))
    density = min(len(page_tokens) / 120.0, 3.0)
    density += min(informative_passage_count, 4) * 0.8
    density += min(len(set(page_tokens)) / max(len(page_tokens), 1), 1.0)
    if evidence_candidates:
        novelty = sum(
            _novelty_against_evidence(candidate.signature_tokens, existing_evidence)
            for candidate in evidence_candidates
        ) / len(evidence_candidates)
    else:
        novelty = 1.0

    hub_score = 0.0
    if _url_looks_homepage_like(page.resolved_url):
        hub_score += 2.5
    if _url_looks_archive_like(page.resolved_url):
        hub_score += 1.5
    hub_score += min(internal_link_count / 5.0, 3.5)
    if informative_passage_count <= 1:
        hub_score += 1.5
    if len(page_tokens) < 120 and internal_link_count >= 4:
        hub_score += 1.5
    if internal_link_count >= 12 and len(page_tokens) < 200:
        hub_score += 2.0
    if _url_looks_article_like(page.resolved_url):
        hub_score -= 0.8
    if _url_looks_live_or_streaming(page.resolved_url):
        hub_score += 2.0

    terminal_score = 0.0
    if _url_looks_article_like(page.resolved_url):
        terminal_score += 2.0
    terminal_score += min(informative_passage_count, 4) * 0.9
    if len(page_tokens) >= 120:
        terminal_score += 1.0
    if internal_link_count <= 3:
        terminal_score += 0.5

    if _looks_low_value_page(
        url=page.resolved_url,
        text=page.text,
        title=title,
        link_count=internal_link_count,
    ):
        terminal_score -= 1.5
    if _url_looks_live_or_streaming(page.resolved_url):
        terminal_score -= 2.0

    usefulness = relevance * 3.0 + density * 1.5 + novelty * 2.0 + terminal_score - hub_score
    return _PageScores(
        relevance=relevance,
        density=density,
        novelty=novelty,
        usefulness=usefulness,
        hub_score=hub_score,
        terminal_score=terminal_score,
        informative_passage_count=informative_passage_count,
        internal_link_count=internal_link_count,
    )


def _should_expand_page(page_scores: _PageScores) -> bool:
    return (
        page_scores.hub_score >= 2.0
        and page_scores.internal_link_count >= 2
        and page_scores.hub_score >= page_scores.terminal_score
    )


def _rank_child_candidates(
    *,
    page: _FetchedSearchPage,
    parent_candidate: _SearchCandidate,
    query: _SearchQuery,
    start_order: int,
) -> list[_SearchCandidate]:
    scored_candidates: list[tuple[float, int, _SearchCandidate]] = []
    seen_urls: set[str] = set()

    for offset, link in enumerate(page.links):
        child_url = urljoin(page.resolved_url, link.url)
        canonical_url = _canonicalize_url(child_url)
        if not canonical_url or canonical_url in seen_urls:
            continue
        if not _is_supported_url(child_url):
            continue
        if not _is_internal_link(parent_url=page.resolved_url, child_url=child_url):
            continue
        if canonical_url == _canonicalize_url(page.resolved_url):
            continue
        if _url_looks_low_value(child_url):
            continue

        score = _score_child_link(
            child_url=child_url,
            anchor_text=link.text,
            parent_url=page.resolved_url,
            query=query,
        )
        candidate = _SearchCandidate(
            url=child_url,
            title=link.text or page.title,
            snippet="",
            depth=parent_candidate.depth + 1,
            priority=score,
            order=start_order + offset,
            parent_url=page.resolved_url,
            anchor_text=link.text,
        )
        scored_candidates.append((score, offset, candidate))
        seen_urls.add(canonical_url)

    scored_candidates.sort(key=lambda item: (-item[0], item[1]))
    return [candidate for _score, _offset, candidate in scored_candidates]


def _score_child_link(
    *,
    child_url: str,
    anchor_text: str,
    parent_url: str,
    query: _SearchQuery,
) -> float:
    folded_metadata = _fold_for_match(f"{anchor_text} {child_url}")
    score = _keyword_overlap_score(folded_metadata, query.keyword_tokens) * 3.0
    score += min(len(_url_path_segments(child_url)), 3)
    if len(_text_tokens(anchor_text)) >= 3:
        score += 1.0
    if _url_looks_article_like(child_url):
        score += 4.0
    if _child_looks_more_content_rich(child_url=child_url, parent_url=parent_url):
        score += 2.0
    if _link_text_looks_generic(anchor_text):
        score -= 3.0
    if _url_looks_archive_like(child_url):
        score -= 0.5
    return score


def _child_looks_more_content_rich(*, child_url: str, parent_url: str) -> bool:
    child_segments = _url_path_segments(child_url)
    parent_segments = _url_path_segments(parent_url)
    if _url_looks_article_like(child_url) and not _url_looks_article_like(parent_url):
        return True
    return len(child_segments) > len(parent_segments)


def _source_worth_showing(source: _RetrievedSource) -> bool:
    """Return True if a source is useful enough to display in final output."""
    # Sources with genuine evidence are always worth showing
    if source.evidence_count > 0 and not source.used_snippet_fallback:
        return True
    # Sources with strong snippet fallback evidence are acceptable
    if source.evidence_count > 0 and source.usefulness >= 3.0:
        return True
    # Live/streaming pages are low value
    if _url_looks_live_or_streaming(source.url):
        return False
    # Sources that were fetched but produced no evidence need high usefulness
    if source.fetched and source.evidence_count == 0:
        return source.usefulness >= 8.0
    # Unfetched sources need at least a reasonable usefulness
    if not source.fetched and source.evidence_count == 0:
        return False
    return source.usefulness >= 2.0


def _should_stop_retrieval(
    *,
    evidence_items: list[_EvidenceItem],
    sources,
    fetch_attempt_count: int,
) -> bool:
    informative_source_count = sum(1 for source in sources if source.evidence_count > 0)
    strong_evidence_count = sum(1 for evidence in evidence_items if evidence.score >= 6.0)
    return (
        fetch_attempt_count >= 4
        and informative_source_count >= 2
        and strong_evidence_count >= _MAX_SUMMARY_POINTS
    )


def _summarize_source_fetch_error(exc: Exception) -> str:
    if isinstance(exc, HTTPError):
        return f"HTTP {exc.code}: {exc.reason}"
    if isinstance(exc, URLError):
        return f"Network error: {exc.reason}"
    return str(exc)


def _select_source_title(page_title: str, search_title: str, anchor_text: str) -> str:
    for candidate in (search_title, page_title, anchor_text):
        normalized_candidate = _normalize_text(candidate)
        if normalized_candidate and normalized_candidate.casefold() not in {"untitled", "index"}:
            return normalized_candidate
    return "Untitled source"


def _build_fallback_takeaway(
    snippet: str,
    *,
    fetch_error: str | None = None,
    url: str = "",
    title: str = "",
    fetched_text: str = "",
) -> str:
    normalized_snippet = _normalize_text(snippet)
    if normalized_snippet:
        return _clip_text(normalized_snippet, limit=_MAX_SOURCE_NOTE_CHARS)
    if fetch_error:
        return _clip_text(
            f"Could not read this page directly: {fetch_error}",
            limit=_MAX_SOURCE_NOTE_CHARS,
        )
    if fetched_text and not _looks_low_value_page(
        url=url,
        text=fetched_text,
        title=title,
        link_count=0,
    ):
        return _truncate_sentences(
            fetched_text,
            max_sentences=2,
            max_chars=_MAX_SOURCE_NOTE_CHARS,
        )
    return "Fetched page, but no strong evidence was kept."


def _iter_passages(text: str) -> tuple[str, ...]:
    passages = tuple(
        paragraph.strip()
        for paragraph in text.split("\n\n")
        if paragraph.strip()
    )
    if passages:
        return passages
    return tuple(line.strip() for line in text.splitlines() if line.strip())


def _is_informative_passage(text: str, *, title: str) -> bool:
    words = text.split()
    if len(words) < 8 or _looks_like_title_echo(text, title):
        return False

    lowered = _fold_for_match(text)
    if lowered.startswith(_MATCH_BOILERPLATE_PREFIXES):
        return False
    if _passage_has_noise_signals(text):
        return False
    if _looks_site_descriptive(text):
        return False
    if _looks_promotional(text):
        return False
    return True


def _looks_low_value_page(
    *,
    url: str,
    text: str,
    title: str,
    link_count: int,
) -> bool:
    if not text or text == "[empty response body]":
        return True
    token_count = len(_text_tokens(text))
    if token_count < 18:
        return True
    if _looks_like_title_echo(text, title):
        return True
    if _url_looks_homepage_like(url) and token_count < 80:
        return True
    if link_count >= 8 and token_count < 120:
        return True
    # Hub/index pages with very high link-to-content ratio
    if link_count >= 12 and token_count < 200:
        return True
    # Pages that are mostly navigation noise
    if _passage_has_noise_signals(text) and token_count < 100:
        return True
    return False


def _looks_like_title_echo(text: str, title: str) -> bool:
    title_tokens = _text_tokens(title)
    text_tokens = _text_tokens(text)
    if not title_tokens or not text_tokens:
        return False

    short_text = len(text_tokens) <= max(len(title_tokens) + 6, 18)
    if not short_text:
        return False

    title_signature = " ".join(title_tokens)
    text_signature = " ".join(text_tokens[: len(title_tokens) + 6])
    if text_signature.startswith(title_signature):
        return True

    title_token_set = set(title_tokens)
    overlap = sum(1 for token in text_tokens if token in title_token_set)
    return overlap >= max(3, len(title_tokens) - 1)


def _looks_boilerplate_text(text: str) -> bool:
    folded_text = _fold_for_match(text)
    return folded_text.startswith(_MATCH_BOILERPLATE_PREFIXES)


def _looks_site_descriptive(text: str) -> bool:
    """Return True if the passage merely describes the site or section rather than conveying facts."""
    folded = _fold_for_match(text)
    _SITE_DESCRIPTIVE_CUES = (
        "retrouvez toute l",
        "retrouvez toutes les",
        "suivez l actualite",
        "toute l actualite",
        "toutes les actualites",
        "toute l information",
        "decouvrez les dernieres",
        "decouvrez toute l",
        "decouvrez toutes les",
        "bienvenue sur",
        "welcome to our",
        "visit our",
        "follow us on",
        "suivez nous sur",
        "retrouvez nous sur",
        "abonnez vous",
        "stay up to date",
        "stay informed",
        "suivez en direct",
        "suivez en temps reel",
        "regardez en direct",
        "a suivre en direct",
        "en direct sur",
        "en continu sur",
        "your source for",
        "your daily source",
        "votre source d",
        "tout savoir sur",
        "l essentiel de l actualite",
        "toute l info",
    )
    return any(cue in folded for cue in _SITE_DESCRIPTIVE_CUES)


def _looks_promotional(text: str) -> bool:
    """Return True if the passage is promotional or service-oriented rather than factual."""
    folded = _fold_for_match(text)
    _PROMOTIONAL_CUES = (
        "abonnez vous",
        "commencez votre",
        "creez votre compte",
        "decouvrez nos offres",
        "essai gratuit",
        "essayez gratuitement",
        "free trial",
        "get started",
        "inscrivez vous",
        "join now",
        "offre d abonnement",
        "offre speciale",
        "profitez de notre",
        "sign up today",
        "start your",
        "subscribe now",
        "try for free",
        "upgrade your",
    )
    return any(cue in folded for cue in _PROMOTIONAL_CUES)


def _passage_has_noise_signals(text: str) -> bool:
    """Return True if the passage contains UI, consent, or navigation noise."""
    folded = _fold_for_match(text)
    return any(phrase in folded for phrase in _NOISE_SIGNAL_PHRASES)


def _passage_noise_score(text: str) -> float:
    """Return a penalty score (0.0 = clean, higher = noisier) for UI/consent noise."""
    folded = _fold_for_match(text)
    hits = sum(1 for phrase in _NOISE_SIGNAL_PHRASES if phrase in folded)
    if hits == 0:
        return 0.0
    tokens = folded.split()
    if not tokens:
        return 0.0
    # Scale penalty: more hits or shorter text = higher noise ratio
    return min(hits * 3.0, 9.0)


def _truncate_sentences(text: str, *, max_sentences: int, max_chars: int) -> str:
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text)
        if sentence.strip()
    ]
    if not sentences:
        return _clip_text(text, limit=max_chars)

    selected = " ".join(sentences[:max_sentences]).strip()
    if selected:
        return _clip_text(selected, limit=max_chars)
    return _clip_text(text, limit=max_chars)


_MIN_SUMMARY_EVIDENCE_SCORE = 3.0


def _synthesize_search_knowledge(
    evidence_items: tuple[_EvidenceItem, ...],
    *,
    query: _SearchQuery,
) -> _SynthesisOutcome:
    if not evidence_items:
        return _SynthesisOutcome(
            statements=(),
            fact_clusters=(),
            findings=(),
        )

    statements = _build_evidence_statements(
        evidence_items,
        query=query,
    )
    if not statements:
        return _SynthesisOutcome(
            statements=(),
            fact_clusters=(),
            findings=(),
        )

    fact_clusters = _build_fact_clusters(
        statements,
        query=query,
    )
    findings = _select_synthesized_findings(fact_clusters)
    return _SynthesisOutcome(
        statements=statements,
        fact_clusters=fact_clusters,
        findings=findings,
    )


def _build_evidence_statements(
    evidence_items: tuple[_EvidenceItem, ...],
    *,
    query: _SearchQuery,
) -> tuple[_EvidenceStatement, ...]:
    statements: list[_EvidenceStatement] = []
    seen_keys: set[tuple[str, str]] = set()

    for evidence in sorted(
        evidence_items,
        key=lambda item: (-item.score, item.depth, item.url, item.text.casefold()),
    ):
        if evidence.score < _MIN_SUMMARY_EVIDENCE_SCORE and evidence.evidence_quality < 1.5:
            continue

        sentence_candidates = _split_sentences(evidence.text) or (evidence.text,)
        for sentence in sentence_candidates:
            statement_text = _normalize_text(sentence)
            if not statement_text:
                continue
            if _looks_site_descriptive(statement_text):
                continue
            if _looks_promotional(statement_text):
                continue
            if _passage_has_noise_signals(statement_text):
                continue

            statement_quality = _score_evidence_text(
                statement_text,
                title=evidence.source_title,
                query=query,
            )
            if statement_quality < 2.0:
                continue

            query_relevance = float(
                _keyword_overlap_score(_fold_for_match(statement_text), query.keyword_tokens)
            )
            signature_tokens = _build_signature_tokens(statement_text)
            content_tokens = _content_tokens(statement_text)
            if not content_tokens:
                continue

            support_pairs = tuple(
                zip(
                    evidence.supporting_urls,
                    evidence.supporting_titles,
                    strict=False,
                )
            ) or ((evidence.url, evidence.source_title),)
            for support_url, support_title in support_pairs:
                support_key = (_canonicalize_url(support_url) or support_url, statement_text)
                if support_key in seen_keys:
                    continue
                statements.append(
                    _EvidenceStatement(
                        text=statement_text,
                        url=support_url,
                        source_title=support_title,
                        depth=evidence.depth,
                        score=statement_quality + evidence.novelty,
                        query_relevance=query_relevance,
                        evidence_quality=max(statement_quality - query_relevance * 2.5, 0.0),
                        novelty=evidence.novelty,
                        signature_tokens=signature_tokens,
                        content_tokens=content_tokens,
                        subject_tokens=_extract_subject_tokens(statement_text),
                    )
                )
                seen_keys.add(support_key)

    return tuple(
        sorted(
            statements,
            key=lambda item: (-item.score, item.depth, item.url, item.text.casefold()),
        )
    )


def _build_fact_clusters(
    statements: tuple[_EvidenceStatement, ...],
    *,
    query: _SearchQuery,
) -> tuple[_FactCluster, ...]:
    clustered_statements: list[list[_EvidenceStatement]] = []

    for statement in statements:
        best_index: int | None = None
        best_similarity = 0.0
        for index, cluster_members in enumerate(clustered_statements):
            similarity = max(
                _statement_similarity(statement, existing)
                for existing in cluster_members
            )
            if similarity > best_similarity:
                best_similarity = similarity
                best_index = index

        if best_index is not None and best_similarity >= 0.54:
            clustered_statements[best_index].append(statement)
            continue

        clustered_statements.append([statement])

    fact_clusters: list[_FactCluster] = []
    for cluster_members in clustered_statements:
        ordered_members = tuple(
            sorted(
                cluster_members,
                key=lambda item: (-item.score, item.depth, item.url, item.text.casefold()),
            )
        )
        supporting_urls = _unique_statement_urls(ordered_members)
        source_titles = _unique_statement_titles(ordered_members)
        support_count = len(supporting_urls)
        distinct_domain_count = len(
            {
                _registered_domain(hostname)
                for hostname in (
                    urlparse(statement.url).hostname or ""
                    for statement in ordered_members
                )
                if hostname
            }
        )
        query_relevance = (
            max((statement.query_relevance for statement in ordered_members), default=0.0)
            + (
                sum(statement.query_relevance for statement in ordered_members)
                / len(ordered_members)
            )
            * 0.5
        )
        evidence_quality = sum(
            statement.evidence_quality for statement in ordered_members[:2]
        ) / min(len(ordered_members), 2)
        novelty = sum(statement.novelty for statement in ordered_members) / len(ordered_members)
        cluster_score = (
            query_relevance * 2.5
            + support_count * 2.0
            + distinct_domain_count * 0.75
            + evidence_quality * 1.5
            + novelty * 1.25
        )

        fact_clusters.append(
            _FactCluster(
                merged_text=_merge_cluster_text(ordered_members, query=query),
                evidence=ordered_members,
                supporting_urls=supporting_urls,
                source_titles=source_titles,
                score=cluster_score,
                query_relevance=query_relevance,
                evidence_quality=evidence_quality,
                novelty=novelty,
                support_count=support_count,
            )
        )

    return tuple(
        sorted(
            fact_clusters,
            key=lambda cluster: (
                -cluster.score,
                -cluster.support_count,
                -cluster.query_relevance,
                cluster.merged_text.casefold(),
            ),
        )
    )


def _select_synthesized_findings(
    fact_clusters: tuple[_FactCluster, ...],
) -> tuple[_SynthesizedFinding, ...]:
    if not fact_clusters:
        return ()

    findings: list[_SynthesizedFinding] = []
    selected_signatures: list[frozenset[str]] = []

    for cluster in fact_clusters:
        cluster_signature = frozenset(_content_tokens(cluster.merged_text))
        if cluster_signature and any(
            _overlap_coefficient(cluster_signature, signature) >= 0.72
            for signature in selected_signatures
        ):
            continue

        adjusted_score = cluster.score
        if selected_signatures:
            adjusted_score -= max(
                _overlap_coefficient(cluster_signature, signature) * 2.5
                for signature in selected_signatures
            )

        if adjusted_score < 3.0 and findings:
            continue

        findings.append(
            _SynthesizedFinding(
                text=_clip_summary_text(cluster.merged_text, limit=_MAX_SUMMARY_POINT_CHARS),
                score=cluster.score,
                support_count=cluster.support_count,
                source_titles=cluster.source_titles,
                source_urls=cluster.supporting_urls,
            )
        )
        selected_signatures.append(cluster_signature)

        if len(findings) >= _MAX_SUMMARY_POINTS:
            break

    if findings:
        return tuple(findings)

    best_cluster = fact_clusters[0]
    return (
        _SynthesizedFinding(
            text=_clip_summary_text(best_cluster.merged_text, limit=_MAX_SUMMARY_POINT_CHARS),
            score=best_cluster.score,
            support_count=best_cluster.support_count,
            source_titles=best_cluster.source_titles,
            source_urls=best_cluster.supporting_urls,
        ),
    )


def _split_sentences(text: str) -> tuple[str, ...]:
    return tuple(
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text)
        if sentence.strip()
    )


def _content_tokens(text: str) -> tuple[str, ...]:
    return tuple(
        token
        for token in _text_tokens(text)
        if token not in _QUERY_STOPWORDS and (len(token) > 2 or token.isdigit())
    )


def _extract_subject_tokens(text: str) -> tuple[str, ...]:
    tokens = _text_tokens(text)
    for index, token in enumerate(tokens):
        if token not in _COPULAR_TOKENS:
            continue
        subject = tuple(
            candidate
            for candidate in tokens[:index]
            if candidate not in _QUERY_STOPWORDS
        )
        if len(subject) == 1 and subject[0] in {"elle", "he", "il", "it", "she", "they"}:
            return ()
        if 2 <= len(subject) <= 6:
            return subject
        return ()
    return ()


def _statement_similarity(
    left: _EvidenceStatement,
    right: _EvidenceStatement,
) -> float:
    similarity = max(
        _evidence_similarity(left.content_tokens, right.content_tokens),
        _overlap_coefficient(left.content_tokens, right.content_tokens),
        _substring_similarity(left.text, right.text),
    )

    if left.subject_tokens and right.subject_tokens:
        subject_overlap = _overlap_coefficient(left.subject_tokens, right.subject_tokens)
        if left.subject_tokens == right.subject_tokens:
            similarity = max(similarity, 0.62)
        elif subject_overlap >= 1.0 and len(set(left.subject_tokens) & set(right.subject_tokens)) >= 2:
            similarity = max(similarity, 0.56)

    return similarity


def _overlap_coefficient(
    left_tokens,
    right_tokens,
) -> float:
    left_set = set(left_tokens)
    right_set = set(right_tokens)
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / min(len(left_set), len(right_set))


def _substring_similarity(left_text: str, right_text: str) -> float:
    left_folded = _fold_for_match(left_text)
    right_folded = _fold_for_match(right_text)
    if not left_folded or not right_folded:
        return 0.0
    if left_folded in right_folded or right_folded in left_folded:
        return 0.78
    return 0.0


def _merge_cluster_text(
    cluster_members: tuple[_EvidenceStatement, ...],
    *,
    query: _SearchQuery,
) -> str:
    copular_merge = _merge_copular_cluster(cluster_members)
    if copular_merge:
        return _clip_summary_text(copular_merge, limit=_MAX_SUMMARY_POINT_CHARS)

    lead_statement = cluster_members[0].text.strip()
    lead_signature = frozenset(cluster_members[0].content_tokens)
    combined_parts = [_strip_terminal_punctuation(lead_statement)]

    for statement in cluster_members[1:]:
        candidate_signature = frozenset(statement.content_tokens)
        if _overlap_coefficient(candidate_signature, lead_signature) >= 0.85:
            continue

        candidate_text = _strip_terminal_punctuation(statement.text)
        proposed = _join_summary_parts((*combined_parts, candidate_text))
        if len(proposed) <= _MAX_SUMMARY_POINT_CHARS:
            combined_parts.append(candidate_text)
            break

    return _clip_summary_text(
        _join_summary_parts(tuple(combined_parts)),
        limit=_MAX_SUMMARY_POINT_CHARS,
    )


def _merge_copular_cluster(
    cluster_members: tuple[_EvidenceStatement, ...],
) -> str:
    parsed_statements: list[tuple[str, str, str, _EvidenceStatement]] = []
    for statement in cluster_members:
        parsed = _split_copular_statement(statement.text)
        if parsed is None:
            continue
        subject_text, verb_text, complement_text = parsed
        if statement.subject_tokens and statement.subject_tokens == _extract_subject_tokens(
            statement.text
        ):
            parsed_statements.append((subject_text, verb_text, complement_text, statement))

    if len(parsed_statements) < 2:
        return ""

    best_subject, best_verb, best_complement, best_statement = parsed_statements[0]
    best_subject_tokens = best_statement.subject_tokens
    complement_parts = [_strip_terminal_punctuation(best_complement)]
    complement_signatures = [frozenset(_content_tokens(best_complement))]

    for subject_text, _verb_text, complement_text, statement in parsed_statements[1:]:
        if statement.subject_tokens != best_subject_tokens:
            continue
        if subject_text.casefold() != best_subject.casefold():
            continue
        complement_signature = frozenset(_content_tokens(complement_text))
        if any(
            _overlap_coefficient(complement_signature, signature) >= 0.72
            for signature in complement_signatures
        ):
            continue
        candidate_part = _strip_terminal_punctuation(complement_text)
        proposed = f"{best_subject} {best_verb} {'; '.join((*complement_parts, candidate_part))}."
        if len(proposed) > _MAX_SUMMARY_POINT_CHARS:
            continue
        complement_parts.append(candidate_part)
        complement_signatures.append(complement_signature)
        if len(complement_parts) >= 3:
            break

    if len(complement_parts) < 2:
        return ""

    return f"{best_subject} {best_verb} {'; '.join(complement_parts)}."


def _split_copular_statement(text: str) -> tuple[str, str, str] | None:
    match = re.match(
        (
            r"^\s*"
            r"([^,.!?]{2,80}?)"
            r"\s+"
            r"(am|are|be|been|being|est|etait|etaient|étaient|était|furent|is|sera|seront|sont|was|were)"
            r"\s+"
            r"(.+?)"
            r"\s*$"
        ),
        text,
        flags=re.IGNORECASE,
    )
    if match is None:
        return None

    subject_text = match.group(1).strip(" ,;:")
    verb_text = match.group(2).strip()
    complement_text = match.group(3).strip()
    if not subject_text or not complement_text:
        return None
    return (subject_text, verb_text, complement_text)


def _join_summary_parts(parts: tuple[str, ...]) -> str:
    cleaned_parts = [part.strip() for part in parts if part.strip()]
    if not cleaned_parts:
        return ""
    joined = ". ".join(_strip_terminal_punctuation(part) for part in cleaned_parts)
    if joined and joined[-1] not in ".!?":
        joined += "."
    return joined


def _strip_terminal_punctuation(text: str) -> str:
    return text.strip().rstrip(" ,;:.!?")


def _clip_summary_text(value: str, *, limit: int) -> str:
    normalized = " ".join(value.split()).strip()
    if len(normalized) <= limit:
        return normalized

    for separator in (". ", "; ", ", "):
        boundary = normalized.rfind(separator, 0, limit)
        if boundary >= int(limit * 0.65):
            clipped = normalized[: boundary + (1 if separator == ". " else 0)].rstrip()
            if clipped and clipped[-1] not in ".!?":
                clipped += "."
            return clipped

    return _clip_text(normalized, limit=limit)


def _merge_unique_strings(
    left: tuple[str, ...],
    right: tuple[str, ...],
) -> tuple[str, ...]:
    merged: list[str] = []
    seen: set[str] = set()
    for value in (*left, *right):
        key = value.casefold()
        if key in seen:
            continue
        merged.append(value)
        seen.add(key)
    return tuple(merged)


def _unique_statement_urls(
    cluster_members: tuple[_EvidenceStatement, ...],
) -> tuple[str, ...]:
    unique_urls: list[str] = []
    seen_urls: set[str] = set()
    for statement in cluster_members:
        canonical_url = _canonicalize_url(statement.url) or statement.url
        if canonical_url in seen_urls:
            continue
        unique_urls.append(statement.url)
        seen_urls.add(canonical_url)
    return tuple(unique_urls)


def _unique_statement_titles(
    cluster_members: tuple[_EvidenceStatement, ...],
) -> tuple[str, ...]:
    unique_titles: list[str] = []
    seen_titles: set[str] = set()
    for statement in cluster_members:
        title_key = statement.source_title.casefold()
        if title_key in seen_titles:
            continue
        unique_titles.append(statement.source_title)
        seen_titles.add(title_key)
    return tuple(unique_titles)


def _format_search_results(
    *,
    query: str,
    outcome: _RetrievalOutcome,
    summary_points: tuple[str, ...],
    synthesis: _SynthesisOutcome,
) -> str:
    display_sources = _select_output_sources(
        sources=outcome.sources,
        synthesis=synthesis,
    )
    lines = [
        f"Search query: {query}",
        f"Sources considered: {outcome.considered_candidate_count}",
        (
            "Sources fetched: "
            f"{outcome.fetch_success_count} of {outcome.fetch_attempt_count} attempted"
        ),
        f"Evidence kept: {len(outcome.evidence_items)}",
        "",
    ]

    if outcome.initial_result_count == 0:
        lines.append("No public web results found.")
        return "\n".join(lines)

    lines.append("Summary:")
    if not summary_points:
        lines.append("- No clear findings could be extracted from the pages fetched.")
    for point in summary_points:
        lines.append(f"- {point}")

    lines.append("")
    lines.append("Sources:")
    if not display_sources:
        lines.append("1. No source was strong enough to retain in the final answer.")
    for index, source in enumerate(display_sources, start=1):
        lines.append(f"{index}. {source.title}")
        lines.append(f"   URL: {source.url}")
        if index != len(display_sources):
            lines.append("")

    return "\n".join(lines)


def _select_output_sources(
    *,
    sources: tuple[_RetrievedSource, ...],
    synthesis: _SynthesisOutcome,
) -> tuple[_RetrievedSource, ...]:
    if not sources:
        return ()

    finding_urls = {
        _canonicalize_url(url) or url
        for finding in synthesis.findings
        for url in finding.source_urls
    }
    if not finding_urls:
        return sources

    selected_sources = tuple(
        source
        for source in sources
        if (_canonicalize_url(source.url) or source.url) in finding_urls
    )
    if selected_sources:
        return selected_sources
    return sources


def _link_text_looks_generic(text: str) -> bool:
    normalized_text = _fold_for_match(text)
    return normalized_text in _GENERIC_LINK_TEXTS


def _fold_for_match(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.casefold())
    without_accents = "".join(
        character
        for character in normalized
        if not unicodedata.combining(character)
    )
    return " ".join(re.findall(r"[a-z0-9]+", without_accents))


def _text_tokens(text: str) -> tuple[str, ...]:
    return tuple(_fold_for_match(text).split())


def _keyword_overlap_score(text: str, keywords: tuple[str, ...]) -> int:
    if not keywords:
        return 0
    text_tokens = set(text.split())
    return sum(1 for keyword in keywords if keyword in text_tokens)


def _url_path_segments(url: str) -> tuple[str, ...]:
    parsed = urlparse(url)
    return tuple(segment for segment in parsed.path.split("/") if segment)


def _url_looks_homepage_like(url: str) -> bool:
    parsed = urlparse(url)
    path = parsed.path.rstrip("/")
    if path in {"", "/"}:
        return True
    return path.casefold() in {"/accueil", "/home", "/index.html"}


def _url_looks_article_like(url: str) -> bool:
    parsed = urlparse(url)
    path = parsed.path
    path_segments = _url_path_segments(url)
    if re.search(r"/20\d{2}/\d{1,2}/\d{1,2}/", path):
        return True
    if re.search(r"/20\d{2}-\d{2}-\d{2}", path):
        return True
    if any(segment.casefold() in _ARTICLE_PATH_CUES for segment in path_segments):
        return True
    if path_segments and path_segments[-1].count("-") >= 3:
        return True
    return False


def _url_looks_archive_like(url: str) -> bool:
    path_segments = {segment.casefold() for segment in _url_path_segments(url)}
    return bool(path_segments & _HUB_PATH_CUES)


def _url_looks_live_or_streaming(url: str) -> bool:
    path_segments = {segment.casefold() for segment in _url_path_segments(url)}
    return bool(path_segments & _LIVE_STREAMING_PATH_CUES)


def _url_looks_low_value(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.fragment:
        return True
    path = parsed.path.casefold()
    if any(path.endswith(extension) for extension in _LOW_VALUE_EXTENSIONS):
        return True
    path_segments = {segment.casefold() for segment in _url_path_segments(url)}
    if path_segments & _LOW_VALUE_PATH_CUES:
        return True
    return False


def _looks_generic_result_title(*, title: str, hostname: str) -> bool:
    folded_title = _fold_for_match(title)
    if folded_title in _LOW_VALUE_RESULT_TITLES:
        return True

    hostname_tokens = [
        token
        for token in re.split(r"[.\-]+", hostname.casefold())
        if token and token not in {"com", "fr", "net", "org", "www"}
    ]
    if hostname_tokens and folded_title == " ".join(hostname_tokens):
        return True
    return False


def _clip_text(value: str, *, limit: int) -> str:
    normalized = " ".join(value.split()).strip()
    if len(normalized) <= limit:
        return normalized

    clipped = normalized[: limit - 3].rstrip(" ,;:.")
    return f"{clipped}..."


def _is_supported_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _canonicalize_url(url: str) -> str:
    if not _is_supported_url(url):
        return ""
    parsed = urlparse(url)
    normalized_path = parsed.path or "/"
    if normalized_path != "/" and normalized_path.endswith("/"):
        normalized_path = normalized_path.rstrip("/")
    return urlunparse(
        (
            parsed.scheme.casefold(),
            parsed.netloc.casefold(),
            normalized_path,
            "",
            parsed.query,
            "",
        )
    )


def _normalize_search_result_url(raw_url: str | None) -> str:
    if raw_url is None or not raw_url.strip():
        return ""

    resolved_url = urljoin(_DUCKDUCKGO_HTML_SEARCH_URL, raw_url.strip())
    parsed = urlparse(resolved_url)
    if parsed.netloc.endswith("duckduckgo.com") and parsed.path.rstrip("/") == "/l":
        redirect_targets = parse_qs(parsed.query).get("uddg", ())
        if redirect_targets:
            target_url = unquote(redirect_targets[0]).strip()
            if target_url:
                return target_url

    return resolved_url


def _registered_domain(hostname: str) -> str:
    normalized_hostname = hostname.rstrip(".").lower()
    if normalized_hostname.startswith("www."):
        normalized_hostname = normalized_hostname[4:]
    parts = normalized_hostname.split(".")
    if len(parts) >= 2:
        return ".".join(parts[-2:])
    return normalized_hostname


def _is_internal_link(*, parent_url: str, child_url: str) -> bool:
    parent_hostname = urlparse(parent_url).hostname
    child_hostname = urlparse(child_url).hostname
    if parent_hostname is None or child_hostname is None:
        return False
    return _registered_domain(parent_hostname) == _registered_domain(child_hostname)


def _open_request(
    request: Request,
    *,
    timeout_seconds: float,
    allow_private_networks: bool,
):
    opener = build_opener(
        _SafeRedirectHandler(
            allow_private_networks=allow_private_networks,
        )
    )
    return opener.open(request, timeout=timeout_seconds)


def _ensure_fetch_target_allowed(
    url: str,
    *,
    allow_private_networks: bool,
) -> None:
    if not _is_supported_url(url):
        raise _BlockedFetchTargetError(
            "Only HTTP and HTTPS fetch targets are supported."
        )
    if allow_private_networks:
        return

    parsed = urlparse(url)
    hostname = parsed.hostname
    if hostname is None:
        raise _BlockedFetchTargetError(
            "Could not determine which host to fetch."
        )

    normalized_host = hostname.rstrip(".").lower()
    if _is_blocked_hostname(normalized_host):
        raise _BlockedFetchTargetError(
            _build_blocked_fetch_message(
                target=normalized_host,
                reason="local or metadata-style hosts are blocked by default",
            )
        )

    literal_ip = _parse_ip_address(normalized_host)
    if literal_ip is not None:
        _raise_if_blocked_ip(literal_ip, target=normalized_host)
        return

    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    for address in _resolve_host_addresses(normalized_host, port):
        _raise_if_blocked_ip(address, target=normalized_host)


def _is_blocked_hostname(hostname: str) -> bool:
    return hostname in _BLOCKED_FETCH_HOSTS or hostname.endswith(".localhost")


def _parse_ip_address(value: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    try:
        return ipaddress.ip_address(value)
    except ValueError:
        return None


def _resolve_host_addresses(
    hostname: str,
    port: int,
) -> tuple[ipaddress.IPv4Address | ipaddress.IPv6Address, ...]:
    try:
        address_infos = socket.getaddrinfo(
            hostname,
            port,
            type=socket.SOCK_STREAM,
        )
    except OSError as exc:
        raise _BlockedFetchTargetError(
            f"Could not resolve '{hostname}' while checking safe fetch rules: {exc}."
        ) from exc

    addresses: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = []
    for _family, _socktype, _proto, _canonname, sockaddr in address_infos:
        host = sockaddr[0]
        address = ipaddress.ip_address(host)
        if address not in addresses:
            addresses.append(address)
    return tuple(addresses)


def _raise_if_blocked_ip(
    address: ipaddress.IPv4Address | ipaddress.IPv6Address,
    *,
    target: str,
) -> None:
    if not _is_blocked_ip(address):
        return
    raise _BlockedFetchTargetError(
        _build_blocked_fetch_message(
            target=target,
            reason=f"{address.compressed} is on a local or private network",
        )
    )


def _is_blocked_ip(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if address.compressed in _BLOCKED_FETCH_IPS:
        return True
    return any(
        (
            address.is_loopback,
            address.is_link_local,
            address.is_multicast,
            address.is_private,
            address.is_reserved,
            address.is_unspecified,
        )
    )


def _build_blocked_fetch_message(*, target: str, reason: str) -> str:
    return (
        f"Fetching '{target}' is blocked because {reason}. "
        "Only public HTTP and HTTPS targets are allowed by default. "
        "The local owner can relax `security.tools.fetch.allow_private_networks` "
        "in config/app.yaml if needed."
    )


def _is_text_content_type(content_type: str) -> bool:
    if content_type.startswith("text/"):
        return True
    return content_type in {
        "application/javascript",
        "application/json",
        "application/xml",
        "application/x-yaml",
    }


def _read_string_argument(arguments: Mapping[str, Any], key: str) -> str:
    value = arguments.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Argument '{key}' must be a non-empty string.")
    return value.strip()


def _read_positive_int_argument(
    arguments: Mapping[str, Any],
    key: str,
    *,
    default: int,
) -> int:
    value = arguments.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Argument '{key}' must be an integer.")
    if value < 1:
        raise ValueError(f"Argument '{key}' must be greater than zero.")
    return value


def _read_limited_positive_int_argument(
    arguments: Mapping[str, Any],
    key: str,
    *,
    default: int,
    maximum: int,
) -> int:
    value = _read_positive_int_argument(arguments, key, default=default)
    if value > maximum:
        raise ValueError(
            f"Argument '{key}' must be less than or equal to {maximum}."
        )
    return value


def _read_positive_number_argument(
    arguments: Mapping[str, Any],
    key: str,
    *,
    default: float,
) -> float:
    value = arguments.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"Argument '{key}' must be a number.")
    if value <= 0:
        raise ValueError(f"Argument '{key}' must be greater than zero.")
    return float(value)


__all__ = [
    "FETCH_URL_TEXT_DEFINITION",
    "SEARCH_WEB_DEFINITION",
    "fetch_url_text",
    "search_web",
    "register_web_tools",
]
