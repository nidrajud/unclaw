"""Shared dataclasses for the web tool pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from unclaw.tools.web.constants import (
    DEFAULT_MAX_CRAWL_DEPTH,
    DEFAULT_MAX_SEARCH_FETCHES,
    DEFAULT_MAX_SEARCH_RESULTS,
    MAX_CHILD_LINKS_PER_PAGE,
    MAX_KEPT_EVIDENCE_ITEMS,
)


@dataclass(frozen=True, slots=True)
class HTMLLink:
    """One normalized link extracted from a fetched HTML page."""

    url: str
    text: str


@dataclass(frozen=True, slots=True)
class RawFetchedDocument:
    """Decoded network response body and basic metadata."""

    requested_url: str
    resolved_url: str
    status_code: int | None
    content_type: str
    decoded_text: str


@dataclass(slots=True)
class FetchedTextDocument:
    """Compact extracted text payload for one fetched public URL."""

    requested_url: str
    resolved_url: str
    status_code: int | None
    content_type: str
    text_excerpt: str
    truncated: bool


@dataclass(frozen=True, slots=True)
class SearchQuery:
    """Normalized query tokens for generic retrieval scoring."""

    raw_query: str
    normalized_query: str
    keyword_tokens: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SearchCandidate:
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
class FetchedSearchPage:
    """Fetched page content enriched with extracted links for retrieval."""

    requested_url: str
    resolved_url: str
    status_code: int | None
    content_type: str
    title: str
    text: str
    truncated: bool
    links: tuple[HTMLLink, ...]


@dataclass(frozen=True, slots=True)
class EvidenceItem:
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
class EvidenceStatement:
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
class FactCluster:
    """One aggregated cluster of overlapping evidence statements."""

    merged_text: str
    evidence: tuple[EvidenceStatement, ...]
    supporting_urls: tuple[str, ...]
    source_titles: tuple[str, ...]
    score: float
    query_relevance: float
    evidence_quality: float
    novelty: float
    support_count: int


@dataclass(frozen=True, slots=True)
class SynthesizedFinding:
    """One user-facing finding built from an aggregated fact cluster."""

    text: str
    score: float
    support_count: int
    source_titles: tuple[str, ...]
    source_urls: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SynthesisOutcome:
    """Temporary synthesized knowledge built from extracted evidence."""

    statements: tuple[EvidenceStatement, ...]
    fact_clusters: tuple[FactCluster, ...]
    findings: tuple[SynthesizedFinding, ...]


@dataclass(slots=True)
class RetrievedSource:
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
class PageScores:
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
class RetrievalBudget:
    """Bounded retrieval budgets to keep /search deterministic and lightweight."""

    max_initial_results: int = DEFAULT_MAX_SEARCH_RESULTS
    max_total_fetches: int = DEFAULT_MAX_SEARCH_FETCHES
    max_depth: int = DEFAULT_MAX_CRAWL_DEPTH
    max_child_links_per_page: int = MAX_CHILD_LINKS_PER_PAGE
    max_kept_evidence_items: int = MAX_KEPT_EVIDENCE_ITEMS


@dataclass(frozen=True, slots=True)
class RetrievalOutcome:
    """Final retrieval state used to build tool output."""

    initial_result_count: int
    considered_candidate_count: int
    fetch_attempt_count: int
    fetch_success_count: int
    evidence_items: tuple[EvidenceItem, ...]
    sources: tuple[RetrievedSource, ...]


__all__ = [
    "EvidenceItem",
    "EvidenceStatement",
    "FactCluster",
    "FetchedSearchPage",
    "FetchedTextDocument",
    "HTMLLink",
    "PageScores",
    "RawFetchedDocument",
    "RetrievedSource",
    "RetrievalBudget",
    "RetrievalOutcome",
    "SearchCandidate",
    "SearchQuery",
    "SynthesisOutcome",
    "SynthesizedFinding",
]
