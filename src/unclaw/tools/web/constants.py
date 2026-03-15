"""Shared constants and public tool definitions for the web stack."""

from __future__ import annotations

from unclaw.tools.contracts import ToolDefinition, ToolPermissionLevel

DEFAULT_MAX_FETCH_CHARS = 8_000
MAX_FETCH_BYTES = 1_000_000
DEFAULT_TIMEOUT_SECONDS = 10.0
DEFAULT_MAX_SEARCH_RESULTS = 20
MAX_SEARCH_RESULTS = 20
DEFAULT_MAX_SEARCH_FETCHES = 30
DEFAULT_MAX_CRAWL_DEPTH = 2
MAX_CHILD_LINKS_PER_PAGE = 3
DEFAULT_SEARCH_FETCH_CHARS = 12_000
MAX_SUMMARY_POINTS = 5
MAX_SUMMARY_POINT_CHARS = 260
MAX_SOURCE_NOTE_CHARS = 220
MAX_PAGE_EVIDENCE_ITEMS = 3
MAX_KEPT_EVIDENCE_ITEMS = 8
MAX_OUTPUT_SOURCES = 8
DUCKDUCKGO_HTML_SEARCH_URL = "https://html.duckduckgo.com/html/"
SEARCH_PROVIDER_NAME = "DuckDuckGo HTML"

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


__all__ = [
    "DEFAULT_MAX_CRAWL_DEPTH",
    "DEFAULT_MAX_FETCH_CHARS",
    "DEFAULT_MAX_SEARCH_FETCHES",
    "DEFAULT_MAX_SEARCH_RESULTS",
    "DEFAULT_SEARCH_FETCH_CHARS",
    "DEFAULT_TIMEOUT_SECONDS",
    "DUCKDUCKGO_HTML_SEARCH_URL",
    "FETCH_URL_TEXT_DEFINITION",
    "MAX_CHILD_LINKS_PER_PAGE",
    "MAX_FETCH_BYTES",
    "MAX_KEPT_EVIDENCE_ITEMS",
    "MAX_OUTPUT_SOURCES",
    "MAX_PAGE_EVIDENCE_ITEMS",
    "MAX_SEARCH_RESULTS",
    "MAX_SOURCE_NOTE_CHARS",
    "MAX_SUMMARY_POINT_CHARS",
    "MAX_SUMMARY_POINTS",
    "SEARCH_PROVIDER_NAME",
    "SEARCH_WEB_DEFINITION",
]
