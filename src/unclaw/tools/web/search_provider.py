"""DuckDuckGo HTML search provider helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from html.parser import HTMLParser
from urllib.parse import parse_qs, unquote, urlencode, urljoin, urlparse
from urllib.request import Request

from unclaw.tools.web import safety as web_safety
from unclaw.tools.web.constants import (
    DUCKDUCKGO_HTML_SEARCH_URL,
    MAX_FETCH_BYTES,
)
from unclaw.tools.web.fetch import _decode_content
from unclaw.tools.web.text import normalize_text, sanitize_model_visible_text


@dataclass(slots=True)
class _SearchResultBuilder:
    """Collect one parsed DuckDuckGo HTML result block."""

    url: str
    title_parts: list[str] = field(default_factory=list)
    snippet_parts: list[str] = field(default_factory=list)


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

        title = sanitize_model_visible_text(
            normalize_text(" ".join(self._current_result.title_parts))
        ).replace("\n", " ")
        url = self._current_result.url.strip()
        snippet = sanitize_model_visible_text(
            normalize_text(" ".join(self._current_result.snippet_parts))
        ).replace("\n", " ")

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
    request_url = f"{DUCKDUCKGO_HTML_SEARCH_URL}?{urlencode({'q': query})}"
    web_safety._ensure_fetch_target_allowed(
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

    with web_safety._open_request(
        request,
        timeout_seconds=timeout_seconds,
        allow_private_networks=False,
    ) as response:
        content_type = response.headers.get_content_type()
        charset = response.headers.get_content_charset() or "utf-8"
        resolved_url = response.geturl()
        web_safety._ensure_fetch_target_allowed(
            resolved_url,
            allow_private_networks=False,
        )
        raw_content = response.read(MAX_FETCH_BYTES + 1)

    if len(raw_content) > MAX_FETCH_BYTES:
        raw_content = raw_content[:MAX_FETCH_BYTES]

    if content_type not in {"text/html", "application/xhtml+xml"}:
        raise ValueError(
            f"Search provider returned unsupported content type: {content_type}"
        )

    return _decode_content(raw_content, charset)


def _normalize_search_result_url(raw_url: str | None) -> str:
    if raw_url is None or not raw_url.strip():
        return ""

    resolved_url = urljoin(DUCKDUCKGO_HTML_SEARCH_URL, raw_url.strip())
    parsed = urlparse(resolved_url)
    if parsed.netloc.endswith("duckduckgo.com") and parsed.path.rstrip("/") == "/l":
        redirect_targets = parse_qs(parsed.query).get("uddg", ())
        if redirect_targets:
            target_url = unquote(redirect_targets[0]).strip()
            if target_url:
                return target_url

    return resolved_url


__all__ = [
    "_parse_duckduckgo_html_results",
    "_search_public_web",
]
