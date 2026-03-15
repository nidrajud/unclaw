"""Search ranking and iterative retrieval helpers."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse

from unclaw.tools.web.constants import (
    DEFAULT_MAX_SEARCH_RESULTS,
    DEFAULT_SEARCH_FETCH_CHARS,
    MAX_KEPT_EVIDENCE_ITEMS,
    MAX_OUTPUT_SOURCES,
    MAX_PAGE_EVIDENCE_ITEMS,
    MAX_SOURCE_NOTE_CHARS,
    MAX_SUMMARY_POINTS,
)
from unclaw.tools.web.fetch import _fetch_search_page
from unclaw.tools.web.models import (
    EvidenceItem,
    FetchedSearchPage,
    PageScores,
    RetrievedSource,
    RetrievalBudget,
    RetrievalOutcome,
    SearchCandidate,
    SearchQuery,
)
from unclaw.tools.web.safety import BlockedFetchTargetError
from unclaw.tools.web.text import (
    canonicalize_url,
    clip_text,
    content_tokens,
    fold_for_match,
    is_informative_passage,
    is_supported_url,
    iter_passages,
    keyword_overlap_score,
    link_text_looks_generic,
    looks_boilerplate_text,
    looks_generic_result_title,
    looks_like_title_echo,
    looks_low_value_page,
    looks_promotional,
    looks_site_descriptive,
    merge_unique_strings,
    normalize_text,
    passage_has_noise_signals,
    passage_noise_score,
    registered_domain,
    text_tokens,
    truncate_sentences,
    url_looks_archive_like,
    url_looks_article_like,
    url_looks_homepage_like,
    url_looks_live_or_streaming,
    url_looks_low_value,
    url_path_segments,
    QUERY_STOPWORDS,
)


def _build_search_query(query: str) -> SearchQuery:
    keyword_tokens = tuple(
        token
        for token in text_tokens(query)
        if len(token) >= 3 and token not in QUERY_STOPWORDS
    )
    if not keyword_tokens:
        keyword_tokens = tuple(token for token in text_tokens(query) if len(token) >= 2)
    return SearchQuery(
        raw_query=query,
        normalized_query=fold_for_match(query),
        keyword_tokens=keyword_tokens,
    )


def _deduplicate_search_results(results: list[dict[str, str]]) -> list[dict[str, str]]:
    deduplicated: list[dict[str, str]] = []
    seen_urls: set[str] = set()

    for result in results:
        url = result["url"].strip()
        canonical_url = canonicalize_url(url)
        if not url or not canonical_url or canonical_url in seen_urls:
            continue
        deduplicated.append(dict(result))
        seen_urls.add(canonical_url)

    return deduplicated


def _rank_search_results(
    results: list[dict[str, str]],
    *,
    query: SearchQuery,
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
    query: SearchQuery,
) -> float:
    title = result.get("title", "")
    snippet = result.get("snippet", "")
    url = result.get("url", "")
    parsed_url = urlparse(url)
    folded_metadata = fold_for_match(f"{title} {snippet} {url}")
    path_segments = url_path_segments(url)
    score = 0.0

    score += keyword_overlap_score(folded_metadata, query.keyword_tokens) * 3.0
    score += min(len(snippet.split()) / 12.0, 2.0)
    score += min(len(path_segments), 3)

    if url_looks_article_like(url):
        score += 4.0
    if url_looks_archive_like(url):
        score += 1.0
    if url_looks_homepage_like(url):
        score -= 6.0
    if url_looks_live_or_streaming(url):
        score -= 5.0
    if looks_generic_result_title(title=title, hostname=parsed_url.hostname or ""):
        score -= 3.0
    if passage_has_noise_signals(snippet):
        score -= 2.0

    return score


def _run_iterative_retrieval(
    *,
    results: list[dict[str, str]],
    query: SearchQuery,
    timeout_seconds: float,
    budget: RetrievalBudget,
) -> RetrievalOutcome:
    queue: list[SearchCandidate] = []
    considered_candidate_urls: set[str] = set()
    sources_by_url: dict[str, RetrievedSource] = {}
    evidence_items: list[EvidenceItem] = []
    initial_results = results[: budget.max_initial_results]
    order = 0

    for result in initial_results:
        candidate = SearchCandidate(
            url=result["url"],
            title=result["title"],
            snippet=result.get("snippet", ""),
            depth=1,
            priority=_score_search_result(result, query=query),
            order=order,
        )
        order += 1
        canonical_url = canonicalize_url(candidate.url)
        if not canonical_url or canonical_url in considered_candidate_urls:
            continue
        queue.append(candidate)
        considered_candidate_urls.add(canonical_url)

    fetch_attempt_count = 0
    fetch_success_count = 0

    while queue and fetch_attempt_count < budget.max_total_fetches:
        queue.sort(key=lambda item: (-item.priority, item.depth, item.order))
        candidate = queue.pop(0)
        candidate_key = canonicalize_url(candidate.url)
        if not candidate_key:
            continue

        fetch_attempt_count += 1

        try:
            page = _fetch_search_page(
                candidate.url,
                max_chars=DEFAULT_SEARCH_FETCH_CHARS,
                timeout_seconds=timeout_seconds,
            )
        except (
            BlockedFetchTargetError,
            HTTPError,
            URLError,
            OSError,
            ValueError,
        ) as exc:
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

            sources_by_url[candidate_key] = RetrievedSource(
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
        page_key = canonicalize_url(page.resolved_url)
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
            best_takeaway = clip_text(
                evidence_candidates[0].text,
                limit=MAX_SOURCE_NOTE_CHARS,
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

        source = RetrievedSource(
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
                child_key = canonicalize_url(child_candidate.url)
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
        )[:MAX_OUTPUT_SOURCES]
    )
    return RetrievalOutcome(
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
    query: SearchQuery,
    existing_evidence: list[EvidenceItem],
) -> list[EvidenceItem]:
    candidates: list[EvidenceItem] = []

    for passage in iter_passages(text):
        evidence_text = truncate_sentences(
            passage,
            max_sentences=2,
            max_chars=MAX_SOURCE_NOTE_CHARS,
        )
        if not evidence_text:
            continue
        query_relevance = float(
            keyword_overlap_score(fold_for_match(evidence_text), query.keyword_tokens)
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
            EvidenceItem(
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
    return deduplicated[:MAX_PAGE_EVIDENCE_ITEMS]


def _build_snippet_evidence(
    *,
    snippet: str,
    url: str,
    title: str,
    depth: int,
    query: SearchQuery,
    existing_evidence: list[EvidenceItem],
) -> EvidenceItem | None:
    normalized_snippet = normalize_text(snippet)
    if not normalized_snippet:
        return None

    evidence_text = clip_text(normalized_snippet, limit=MAX_SOURCE_NOTE_CHARS)
    query_relevance = float(
        keyword_overlap_score(fold_for_match(evidence_text), query.keyword_tokens)
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
    return EvidenceItem(
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
    query: SearchQuery,
) -> float:
    if not is_informative_passage(text, title=title):
        return -1.0

    tokens = text_tokens(text)
    folded_text = fold_for_match(text)
    unique_ratio = len(set(tokens)) / max(len(tokens), 1)
    score = min(len(tokens) / 12.0, 4.0)
    keyword_overlap = keyword_overlap_score(folded_text, query.keyword_tokens)
    score += keyword_overlap * 2.5
    score += unique_ratio * 2.0
    if re.search(r"\b\d+(?:[.,]\d+)?%?\b", text):
        score += 0.5
    if looks_like_title_echo(text, title):
        score -= 2.5
    if looks_boilerplate_text(text):
        score -= 3.0
    score -= passage_noise_score(text)
    if looks_site_descriptive(text):
        score -= 2.0
    if looks_promotional(text):
        score -= 3.0
    if len(tokens) < 12 and keyword_overlap == 0:
        score -= 2.0
    return score


def _deduplicate_evidence_items(candidates: list[EvidenceItem]) -> list[EvidenceItem]:
    deduplicated: list[EvidenceItem] = []
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
    evidence_items: list[EvidenceItem],
    candidate: EvidenceItem,
    *,
    maximum: int,
) -> bool:
    for index, existing in enumerate(evidence_items):
        similarity = _evidence_similarity(
            candidate.signature_tokens,
            existing.signature_tokens,
        )
        if similarity >= 0.8:
            merged_urls = merge_unique_strings(
                existing.supporting_urls,
                candidate.supporting_urls,
            )
            merged_titles = merge_unique_strings(
                existing.supporting_titles,
                candidate.supporting_titles,
            )
            preferred = candidate if candidate.score > existing.score else existing
            evidence_items[index] = EvidenceItem(
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
    return tuple(text_tokens(text)[:24])


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
    evidence_items: list[EvidenceItem],
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
    page: FetchedSearchPage,
    title: str,
    query: SearchQuery,
    evidence_candidates: list[EvidenceItem],
    existing_evidence: list[EvidenceItem],
) -> PageScores:
    page_tokens = text_tokens(page.text)
    informative_passage_count = sum(
        1
        for passage in iter_passages(page.text)
        if is_informative_passage(passage, title=title)
    )
    internal_link_count = sum(
        1
        for link in page.links
        if _is_internal_link(
            parent_url=page.resolved_url,
            child_url=urljoin(page.resolved_url, link.url),
        )
    )
    folded_metadata = fold_for_match(f"{title} {page.text}")
    relevance = float(keyword_overlap_score(folded_metadata, query.keyword_tokens))
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
    if url_looks_homepage_like(page.resolved_url):
        hub_score += 2.5
    if url_looks_archive_like(page.resolved_url):
        hub_score += 1.5
    hub_score += min(internal_link_count / 5.0, 3.5)
    if informative_passage_count <= 1:
        hub_score += 1.5
    if len(page_tokens) < 120 and internal_link_count >= 4:
        hub_score += 1.5
    if internal_link_count >= 12 and len(page_tokens) < 200:
        hub_score += 2.0
    if url_looks_article_like(page.resolved_url):
        hub_score -= 0.8
    if url_looks_live_or_streaming(page.resolved_url):
        hub_score += 2.0

    terminal_score = 0.0
    if url_looks_article_like(page.resolved_url):
        terminal_score += 2.0
    terminal_score += min(informative_passage_count, 4) * 0.9
    if len(page_tokens) >= 120:
        terminal_score += 1.0
    if internal_link_count <= 3:
        terminal_score += 0.5

    if looks_low_value_page(
        url=page.resolved_url,
        text=page.text,
        title=title,
        link_count=internal_link_count,
    ):
        terminal_score -= 1.5
    if url_looks_live_or_streaming(page.resolved_url):
        terminal_score -= 2.0

    usefulness = relevance * 3.0 + density * 1.5 + novelty * 2.0 + terminal_score - hub_score
    return PageScores(
        relevance=relevance,
        density=density,
        novelty=novelty,
        usefulness=usefulness,
        hub_score=hub_score,
        terminal_score=terminal_score,
        informative_passage_count=informative_passage_count,
        internal_link_count=internal_link_count,
    )


def _should_expand_page(page_scores: PageScores) -> bool:
    return (
        page_scores.hub_score >= 2.0
        and page_scores.internal_link_count >= 2
        and page_scores.hub_score >= page_scores.terminal_score
    )


def _rank_child_candidates(
    *,
    page: FetchedSearchPage,
    parent_candidate: SearchCandidate,
    query: SearchQuery,
    start_order: int,
) -> list[SearchCandidate]:
    scored_candidates: list[tuple[float, int, SearchCandidate]] = []
    seen_urls: set[str] = set()

    for offset, link in enumerate(page.links):
        child_url = urljoin(page.resolved_url, link.url)
        canonical_url = canonicalize_url(child_url)
        if not canonical_url or canonical_url in seen_urls:
            continue
        if not is_supported_url(child_url):
            continue
        if not _is_internal_link(parent_url=page.resolved_url, child_url=child_url):
            continue
        if canonical_url == canonicalize_url(page.resolved_url):
            continue
        if url_looks_low_value(child_url):
            continue

        score = _score_child_link(
            child_url=child_url,
            anchor_text=link.text,
            parent_url=page.resolved_url,
            query=query,
        )
        candidate = SearchCandidate(
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
    query: SearchQuery,
) -> float:
    folded_metadata = fold_for_match(f"{anchor_text} {child_url}")
    score = keyword_overlap_score(folded_metadata, query.keyword_tokens) * 3.0
    score += min(len(url_path_segments(child_url)), 3)
    if len(text_tokens(anchor_text)) >= 3:
        score += 1.0
    if url_looks_article_like(child_url):
        score += 4.0
    if _child_looks_more_content_rich(child_url=child_url, parent_url=parent_url):
        score += 2.0
    if link_text_looks_generic(anchor_text):
        score -= 3.0
    if url_looks_archive_like(child_url):
        score -= 0.5
    return score


def _child_looks_more_content_rich(*, child_url: str, parent_url: str) -> bool:
    child_segments = url_path_segments(child_url)
    parent_segments = url_path_segments(parent_url)
    if url_looks_article_like(child_url) and not url_looks_article_like(parent_url):
        return True
    return len(child_segments) > len(parent_segments)


def _source_worth_showing(source: RetrievedSource) -> bool:
    if source.evidence_count > 0 and not source.used_snippet_fallback:
        return True
    if source.evidence_count > 0 and source.usefulness >= 3.0:
        return True
    if url_looks_live_or_streaming(source.url):
        return False
    if source.fetched and source.evidence_count == 0:
        return source.usefulness >= 8.0
    if not source.fetched and source.evidence_count == 0:
        return False
    return source.usefulness >= 2.0


def _should_stop_retrieval(
    *,
    evidence_items: list[EvidenceItem],
    sources: Iterable[RetrievedSource],
    fetch_attempt_count: int,
) -> bool:
    informative_source_count = sum(1 for source in sources if source.evidence_count > 0)
    strong_evidence_count = sum(1 for evidence in evidence_items if evidence.score >= 6.0)
    return (
        fetch_attempt_count >= 4
        and informative_source_count >= 2
        and strong_evidence_count >= MAX_SUMMARY_POINTS
    )


def _summarize_source_fetch_error(exc: Exception) -> str:
    if isinstance(exc, HTTPError):
        return f"HTTP {exc.code}: {exc.reason}"
    if isinstance(exc, URLError):
        return f"Network error: {exc.reason}"
    return str(exc)


def _select_source_title(page_title: str, search_title: str, anchor_text: str) -> str:
    for candidate in (search_title, page_title, anchor_text):
        normalized_candidate = normalize_text(candidate)
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
    normalized_snippet = normalize_text(snippet)
    if normalized_snippet:
        return clip_text(normalized_snippet, limit=MAX_SOURCE_NOTE_CHARS)
    if fetch_error:
        return clip_text(
            f"Could not read this page directly: {fetch_error}",
            limit=MAX_SOURCE_NOTE_CHARS,
        )
    if fetched_text and not looks_low_value_page(
        url=url,
        text=fetched_text,
        title=title,
        link_count=0,
    ):
        return truncate_sentences(
            fetched_text,
            max_sentences=2,
            max_chars=MAX_SOURCE_NOTE_CHARS,
        )
    return "Fetched page, but no strong evidence was kept."


def _is_internal_link(*, parent_url: str, child_url: str) -> bool:
    parent_hostname = urlparse(parent_url).hostname
    child_hostname = urlparse(child_url).hostname
    if parent_hostname is None or child_hostname is None:
        return False
    return registered_domain(parent_hostname) == registered_domain(child_hostname)


__all__ = [
    "RetrievedSource",
    "RetrievalBudget",
    "RetrievalOutcome",
    "SearchQuery",
    "_build_search_query",
    "_deduplicate_search_results",
    "_rank_search_results",
    "_run_iterative_retrieval",
    "_should_stop_retrieval",
]
