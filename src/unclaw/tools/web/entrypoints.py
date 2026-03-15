"""Public entrypoints for the web tools."""

from __future__ import annotations

from urllib.error import HTTPError, URLError

from unclaw.tools.contracts import ToolCall, ToolResult
from unclaw.tools.registry import ToolRegistry
from unclaw.tools.web.constants import (
    DEFAULT_MAX_FETCH_CHARS,
    DEFAULT_MAX_SEARCH_RESULTS,
    DEFAULT_TIMEOUT_SECONDS,
    FETCH_URL_TEXT_DEFINITION,
    MAX_SEARCH_RESULTS,
    SEARCH_PROVIDER_NAME,
    SEARCH_WEB_DEFINITION,
)
from unclaw.tools.web.fetch import (
    _fetch_text_document,
    _format_text_excerpt,
)
from unclaw.tools.web.retrieval import (
    RetrievalBudget,
    _build_search_query,
    _deduplicate_search_results,
    _rank_search_results,
    _run_iterative_retrieval,
)
from unclaw.tools.web.safety import BlockedFetchTargetError
from unclaw.tools.web.search_provider import (
    _parse_duckduckgo_html_results,
    _search_public_web,
)
from unclaw.tools.web.synthesis import (
    _format_search_results,
    _select_output_sources,
    _synthesize_search_knowledge,
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
            default=DEFAULT_MAX_FETCH_CHARS,
        )
        timeout_seconds = _read_positive_number_argument(
            call.arguments,
            "timeout_seconds",
            default=DEFAULT_TIMEOUT_SECONDS,
        )
    except ValueError as exc:
        return ToolResult.failure(tool_name=tool_name, error=str(exc))

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
    except BlockedFetchTargetError as exc:
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
            default=DEFAULT_MAX_SEARCH_RESULTS,
            maximum=MAX_SEARCH_RESULTS,
        )
        timeout_seconds = _read_positive_number_argument(
            call.arguments,
            "timeout_seconds",
            default=DEFAULT_TIMEOUT_SECONDS,
        )
    except ValueError as exc:
        return ToolResult.failure(tool_name=tool_name, error=str(exc))

    try:
        response_text = _search_public_web(
            query=query,
            timeout_seconds=timeout_seconds,
        )
    except BlockedFetchTargetError as exc:
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
        budget=RetrievalBudget(max_initial_results=max_results),
    )
    synthesis = _synthesize_search_knowledge(
        outcome.evidence_items,
        query=search_query,
    )
    summary_points = tuple(finding.text for finding in synthesis.findings)
    display_sources = _select_output_sources(
        sources=outcome.sources,
        synthesis=synthesis,
    )
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
            "provider": SEARCH_PROVIDER_NAME,
            "initial_result_count": outcome.initial_result_count,
            "considered_candidate_count": outcome.considered_candidate_count,
            "fetch_attempt_count": outcome.fetch_attempt_count,
            "fetch_success_count": outcome.fetch_success_count,
            "evidence_count": len(outcome.evidence_items),
            "statement_count": len(synthesis.statements),
            "fact_cluster_count": len(synthesis.fact_clusters),
            "finding_count": len(synthesis.findings),
            "summary_points": list(summary_points),
            "display_sources": [
                {
                    "title": source.title,
                    "url": source.url,
                }
                for source in display_sources
            ],
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


def _read_string_argument(arguments, key: str) -> str:
    value = arguments.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Argument '{key}' must be a non-empty string.")
    return value.strip()


def _read_positive_int_argument(
    arguments,
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
    arguments,
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
    arguments,
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
    "fetch_url_text",
    "register_web_tools",
    "search_web",
]
