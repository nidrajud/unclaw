"""HTML parsing helpers for fetched web pages."""

from __future__ import annotations

from dataclasses import dataclass, field
from html.parser import HTMLParser

from unclaw.tools.web.models import HTMLLink
from unclaw.tools.web.text import normalize_text, sanitize_model_visible_text

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
        self._links: list[HTMLLink] = []

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
            text = sanitize_model_visible_text(
                normalize_text(" ".join(self._current_link.text_parts))
            ).replace("\n", " ")
            if href:
                self._links.append(HTMLLink(url=href, text=text))
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
        return sanitize_model_visible_text(normalize_text(" ".join(self._title_parts)))

    @property
    def text(self) -> str:
        return sanitize_model_visible_text(normalize_text("".join(self._parts)))

    @property
    def links(self) -> tuple[HTMLLink, ...]:
        return tuple(self._links)


def extract_html_content(content: str) -> tuple[str, str, tuple[HTMLLink, ...]]:
    parser = _HTMLPageExtractor()
    parser.feed(content)
    parser.close()
    return (parser.title, parser.text, parser.links)


__all__ = ["extract_html_content"]
