"""Evidence synthesis and output formatting for web search results."""

from __future__ import annotations

import re
from collections.abc import Mapping
from urllib.parse import urlparse

from unclaw.tools.web.constants import (
    MAX_OUTPUT_SOURCES,
    MAX_SUMMARY_POINT_CHARS,
    MAX_SUMMARY_POINTS,
)
from unclaw.tools.web.models import (
    EvidenceItem,
    EvidenceStatement,
    FactCluster,
    RetrievedSource,
    RetrievalOutcome,
    SearchQuery,
    SynthesisOutcome,
    SynthesizedFinding,
)
from unclaw.tools.web.retrieval import _score_evidence_text
from unclaw.tools.web.text import (
    COPULAR_TOKENS,
    QUERY_STOPWORDS,
    canonicalize_url,
    clip_summary_text,
    content_tokens,
    fold_for_match,
    join_summary_parts,
    keyword_overlap_score,
    looks_generic_result_title,
    looks_promotional,
    looks_site_descriptive,
    passage_has_noise_signals,
    strip_terminal_punctuation,
    text_tokens,
    url_looks_article_like,
    url_looks_homepage_like,
)

_MIN_SUMMARY_EVIDENCE_SCORE = 3.0


def _synthesize_search_knowledge(
    evidence_items: tuple[EvidenceItem, ...],
    *,
    query: SearchQuery,
) -> SynthesisOutcome:
    if not evidence_items:
        return SynthesisOutcome(
            statements=(),
            fact_clusters=(),
            findings=(),
        )

    statements = _build_evidence_statements(
        evidence_items,
        query=query,
    )
    if not statements:
        return SynthesisOutcome(
            statements=(),
            fact_clusters=(),
            findings=(),
        )

    fact_clusters = _build_fact_clusters(
        statements,
        query=query,
    )
    findings = _select_synthesized_findings(fact_clusters)
    return SynthesisOutcome(
        statements=statements,
        fact_clusters=fact_clusters,
        findings=findings,
    )


def _build_evidence_statements(
    evidence_items: tuple[EvidenceItem, ...],
    *,
    query: SearchQuery,
) -> tuple[EvidenceStatement, ...]:
    statements: list[EvidenceStatement] = []
    seen_keys: set[tuple[str, str]] = set()

    for evidence in sorted(
        evidence_items,
        key=lambda item: (-item.score, item.depth, item.url, item.text.casefold()),
    ):
        if evidence.score < _MIN_SUMMARY_EVIDENCE_SCORE and evidence.evidence_quality < 1.5:
            continue

        sentence_candidates = _split_sentences(evidence.text) or (evidence.text,)
        for sentence in sentence_candidates:
            statement_text = sentence.strip()
            if not statement_text:
                continue
            if looks_site_descriptive(statement_text):
                continue
            if looks_promotional(statement_text):
                continue
            if passage_has_noise_signals(statement_text):
                continue

            statement_quality = _score_evidence_text(
                statement_text,
                title=evidence.source_title,
                query=query,
            )
            if statement_quality < 2.0:
                continue

            query_relevance = float(
                keyword_overlap_score(fold_for_match(statement_text), query.keyword_tokens)
            )
            signature_tokens = tuple(text_tokens(statement_text)[:24])
            statement_content_tokens = content_tokens(statement_text)
            if not statement_content_tokens:
                continue

            support_pairs = tuple(
                zip(
                    evidence.supporting_urls,
                    evidence.supporting_titles,
                    strict=False,
                )
            ) or ((evidence.url, evidence.source_title),)
            for support_url, support_title in support_pairs:
                support_key = (canonicalize_url(support_url) or support_url, statement_text)
                if support_key in seen_keys:
                    continue
                statements.append(
                    EvidenceStatement(
                        text=statement_text,
                        url=support_url,
                        source_title=support_title,
                        depth=evidence.depth,
                        score=statement_quality + evidence.novelty,
                        query_relevance=query_relevance,
                        evidence_quality=max(statement_quality - query_relevance * 2.5, 0.0),
                        novelty=evidence.novelty,
                        signature_tokens=signature_tokens,
                        content_tokens=statement_content_tokens,
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
    statements: tuple[EvidenceStatement, ...],
    *,
    query: SearchQuery,
) -> tuple[FactCluster, ...]:
    del query
    clustered_statements: list[list[EvidenceStatement]] = []

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

    fact_clusters: list[FactCluster] = []
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
            FactCluster(
                merged_text=_merge_cluster_text(ordered_members),
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
    fact_clusters: tuple[FactCluster, ...],
) -> tuple[SynthesizedFinding, ...]:
    if not fact_clusters:
        return ()

    findings: list[SynthesizedFinding] = []
    selected_signatures: list[frozenset[str]] = []

    for cluster in fact_clusters:
        cluster_signature = frozenset(content_tokens(cluster.merged_text))
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
            SynthesizedFinding(
                text=clip_summary_text(cluster.merged_text, limit=MAX_SUMMARY_POINT_CHARS),
                score=cluster.score,
                support_count=cluster.support_count,
                source_titles=cluster.source_titles,
                source_urls=cluster.supporting_urls,
            )
        )
        selected_signatures.append(cluster_signature)

        if len(findings) >= MAX_SUMMARY_POINTS:
            break

    if findings:
        return tuple(findings)

    best_cluster = fact_clusters[0]
    return (
        SynthesizedFinding(
            text=clip_summary_text(best_cluster.merged_text, limit=MAX_SUMMARY_POINT_CHARS),
            score=best_cluster.score,
            support_count=best_cluster.support_count,
            source_titles=best_cluster.source_titles,
            source_urls=best_cluster.supporting_urls,
        ),
    )


def _format_search_results(
    *,
    query: str,
    outcome: RetrievalOutcome,
    summary_points: tuple[str, ...],
    synthesis: SynthesisOutcome,
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
    sources: tuple[RetrievedSource, ...],
    synthesis: SynthesisOutcome,
) -> tuple[RetrievedSource, ...]:
    if not sources:
        return ()

    finding_weight_by_url: dict[str, float] = {}
    for finding in synthesis.findings:
        base_weight = finding.score + finding.support_count * 2.0
        for position, url in enumerate(finding.source_urls):
            canonical_url = canonicalize_url(url) or url
            finding_weight_by_url[canonical_url] = (
                finding_weight_by_url.get(canonical_url, 0.0)
                + max(base_weight - position * 0.5, 0.5)
            )

    candidate_sources = tuple(
        source
        for source in sources
        if (canonicalize_url(source.url) or source.url) in finding_weight_by_url
    )
    if not candidate_sources:
        candidate_sources = sources

    ranked_sources = tuple(
        sorted(
            candidate_sources,
            key=lambda source: (
                -_output_source_priority(
                    source=source,
                    finding_weight_by_url=finding_weight_by_url,
                ),
                source.depth,
                source.title.casefold(),
                source.url,
            ),
        )
    )
    return ranked_sources[:MAX_OUTPUT_SOURCES]


def _output_source_priority(
    *,
    source: RetrievedSource,
    finding_weight_by_url: Mapping[str, float],
) -> float:
    canonical_url = canonicalize_url(source.url) or source.url
    priority = finding_weight_by_url.get(canonical_url, 0.0) * 4.0
    priority += source.usefulness
    priority += source.evidence_count * 2.0
    if source.fetched:
        priority += 0.75
    if not source.used_snippet_fallback:
        priority += 0.5
    if url_looks_article_like(source.url):
        priority += 0.5
    if url_looks_homepage_like(source.url):
        priority -= 3.0
    if looks_generic_result_title(
        title=source.title,
        hostname=urlparse(source.url).hostname or "",
    ):
        priority -= 2.0
    if _source_title_looks_weak(source.title):
        priority -= 1.25
    return priority


def _source_title_looks_weak(title: str) -> bool:
    folded_title = fold_for_match(title)
    return any(
        token in folded_title
        for token in ("podcast", "episode", "newsletter", "substack", "roundup")
    )


def _split_sentences(text: str) -> tuple[str, ...]:
    return tuple(
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text)
        if sentence.strip()
    )


def _extract_subject_tokens(text: str) -> tuple[str, ...]:
    tokens = text_tokens(text)
    for index, token in enumerate(tokens):
        if token not in COPULAR_TOKENS:
            continue
        subject = tuple(
            candidate
            for candidate in tokens[:index]
            if candidate not in QUERY_STOPWORDS
        )
        if len(subject) == 1 and subject[0] in {"elle", "he", "il", "it", "she", "they"}:
            return ()
        if 2 <= len(subject) <= 6:
            return subject
        return ()
    return ()


def _statement_similarity(
    left: EvidenceStatement,
    right: EvidenceStatement,
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
    left_folded = fold_for_match(left_text)
    right_folded = fold_for_match(right_text)
    if not left_folded or not right_folded:
        return 0.0
    if left_folded in right_folded or right_folded in left_folded:
        return 0.78
    return 0.0


def _merge_cluster_text(
    cluster_members: tuple[EvidenceStatement, ...],
) -> str:
    copular_merge = _merge_copular_cluster(cluster_members)
    if copular_merge:
        return clip_summary_text(copular_merge, limit=MAX_SUMMARY_POINT_CHARS)

    lead_statement = cluster_members[0].text.strip()
    lead_signature = frozenset(cluster_members[0].content_tokens)
    combined_parts = [strip_terminal_punctuation(lead_statement)]

    for statement in cluster_members[1:]:
        candidate_signature = frozenset(statement.content_tokens)
        if _overlap_coefficient(candidate_signature, lead_signature) >= 0.85:
            continue

        candidate_text = strip_terminal_punctuation(statement.text)
        proposed = join_summary_parts((*combined_parts, candidate_text))
        if len(proposed) <= MAX_SUMMARY_POINT_CHARS:
            combined_parts.append(candidate_text)
            break

    return clip_summary_text(
        join_summary_parts(tuple(combined_parts)),
        limit=MAX_SUMMARY_POINT_CHARS,
    )


def _merge_copular_cluster(
    cluster_members: tuple[EvidenceStatement, ...],
) -> str:
    parsed_statements: list[tuple[str, str, str, EvidenceStatement]] = []
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
    complement_parts = [strip_terminal_punctuation(best_complement)]
    complement_signatures = [frozenset(content_tokens(best_complement))]

    for subject_text, _verb_text, complement_text, statement in parsed_statements[1:]:
        if statement.subject_tokens != best_subject_tokens:
            continue
        if subject_text.casefold() != best_subject.casefold():
            continue
        complement_signature = frozenset(content_tokens(complement_text))
        if any(
            _overlap_coefficient(complement_signature, signature) >= 0.72
            for signature in complement_signatures
        ):
            continue
        candidate_part = strip_terminal_punctuation(complement_text)
        proposed = f"{best_subject} {best_verb} {'; '.join((*complement_parts, candidate_part))}."
        if len(proposed) > MAX_SUMMARY_POINT_CHARS:
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


def _unique_statement_urls(
    cluster_members: tuple[EvidenceStatement, ...],
) -> tuple[str, ...]:
    unique_urls: list[str] = []
    seen_urls: set[str] = set()
    for statement in cluster_members:
        canonical_url = canonicalize_url(statement.url) or statement.url
        if canonical_url in seen_urls:
            continue
        unique_urls.append(statement.url)
        seen_urls.add(canonical_url)
    return tuple(unique_urls)


def _unique_statement_titles(
    cluster_members: tuple[EvidenceStatement, ...],
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


def _registered_domain(hostname: str) -> str:
    normalized_hostname = hostname.rstrip(".").lower()
    if normalized_hostname.startswith("www."):
        normalized_hostname = normalized_hostname[4:]
    parts = normalized_hostname.split(".")
    if len(parts) >= 2:
        return ".".join(parts[-2:])
    return normalized_hostname


__all__ = [
    "SynthesisOutcome",
    "_format_search_results",
    "_select_output_sources",
    "_synthesize_search_knowledge",
]
