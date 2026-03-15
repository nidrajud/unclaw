"""HTTP fetching and text extraction helpers for the web tools."""

from __future__ import annotations

from urllib.error import HTTPError, URLError
from urllib.request import Request

from unclaw.tools.web import safety as web_safety
from unclaw.tools.web.constants import MAX_FETCH_BYTES
from unclaw.tools.web.html import extract_html_content
from unclaw.tools.web.models import FetchedSearchPage, FetchedTextDocument, RawFetchedDocument
from unclaw.tools.web.text import normalize_text, sanitize_model_visible_text

_HTML_CONTENT_TYPES = {"text/html", "application/xhtml+xml"}


def _decode_content(raw_content: bytes, charset: str) -> str:
    try:
        return raw_content.decode(charset)
    except (LookupError, UnicodeDecodeError):
        return raw_content.decode("utf-8", errors="replace")


def _fetch_raw_document(
    url: str,
    *,
    timeout_seconds: float,
    allow_private_networks: bool,
    accept_header: str,
) -> RawFetchedDocument:
    web_safety._ensure_fetch_target_allowed(
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

    with web_safety._open_request(
        request,
        timeout_seconds=timeout_seconds,
        allow_private_networks=allow_private_networks,
    ) as response:
        content_type = response.headers.get_content_type()
        charset = response.headers.get_content_charset() or "utf-8"
        status_code = getattr(response, "status", None)
        resolved_url = response.geturl()
        web_safety._ensure_fetch_target_allowed(
            resolved_url,
            allow_private_networks=allow_private_networks,
        )
        raw_content = response.read(MAX_FETCH_BYTES + 1)

    if len(raw_content) > MAX_FETCH_BYTES:
        raw_content = raw_content[:MAX_FETCH_BYTES]

    return RawFetchedDocument(
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
) -> FetchedTextDocument:
    raw_document = _fetch_raw_document(
        url,
        timeout_seconds=timeout_seconds,
        allow_private_networks=allow_private_networks,
        accept_header=accept_header,
    )
    extracted_text, _title, _links = _extract_text_content(
        raw_document.decoded_text,
        raw_document.content_type,
    )
    if not extracted_text:
        extracted_text = "[empty response body]"

    truncated = len(extracted_text) > max_chars
    text_excerpt = extracted_text[:max_chars].rstrip() if truncated else extracted_text
    return FetchedTextDocument(
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
) -> FetchedSearchPage:
    raw_document = _fetch_raw_document(
        url,
        timeout_seconds=timeout_seconds,
        allow_private_networks=False,
        accept_header="text/plain, text/html, application/json;q=0.9, */*;q=0.1",
    )
    extracted_text, page_title, links = _extract_text_content(
        raw_document.decoded_text,
        raw_document.content_type,
    )
    if not extracted_text:
        extracted_text = "[empty response body]"

    truncated = len(extracted_text) > max_chars
    page_text = extracted_text[:max_chars].rstrip() if truncated else extracted_text
    return FetchedSearchPage(
        requested_url=url,
        resolved_url=raw_document.resolved_url,
        status_code=raw_document.status_code,
        content_type=raw_document.content_type,
        title=page_title,
        text=page_text,
        truncated=truncated,
        links=links,
    )


def _extract_text_content(
    content: str,
    content_type: str,
):
    if content_type in _HTML_CONTENT_TYPES:
        title, extracted_text, links = extract_html_content(content)
        return (extracted_text, title, links)

    if _is_text_content_type(content_type):
        return (
            sanitize_model_visible_text(normalize_text(content)),
            "",
            (),
        )

    raise ValueError(
        f"Unsupported content type for text extraction: {content_type}"
    )


def _format_text_excerpt(text: str, *, truncated: bool) -> str:
    if not truncated:
        return text
    return f"{text}\n\n[truncated]"


def _is_text_content_type(content_type: str) -> bool:
    if content_type.startswith("text/"):
        return True
    return content_type in {
        "application/javascript",
        "application/json",
        "application/xml",
        "application/x-yaml",
    }


__all__ = [
    "HTTPError",
    "URLError",
    "_decode_content",
    "_fetch_raw_document",
    "_fetch_search_page",
    "_fetch_text_document",
    "_format_text_excerpt",
]
