"""Lightweight web tools for the early Unclaw runtime."""

from __future__ import annotations

from collections.abc import Mapping
from html.parser import HTMLParser
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

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


def register_web_tools(registry: ToolRegistry) -> None:
    """Register the built-in lightweight web tools."""
    registry.register(FETCH_URL_TEXT_DEFINITION, fetch_url_text)


def fetch_url_text(call: ToolCall) -> ToolResult:
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

    request = Request(
        url,
        headers={
            "User-Agent": "unclaw/0.1 (+https://local-first.invalid)",
            "Accept": "text/plain, text/html, application/json;q=0.9, */*;q=0.1",
        },
    )

    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            content_type = response.headers.get_content_type()
            charset = response.headers.get_content_charset() or "utf-8"
            status_code = getattr(response, "status", None)
            resolved_url = response.geturl()
            raw_content = response.read(_MAX_FETCH_BYTES + 1)
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

    if len(raw_content) > _MAX_FETCH_BYTES:
        raw_content = raw_content[:_MAX_FETCH_BYTES]

    decoded_text = _decode_content(raw_content, charset)
    extracted_text = _extract_text(decoded_text, content_type)
    if extracted_text is None:
        return ToolResult.failure(
            tool_name=tool_name,
            error=f"Unsupported content type for text extraction: {content_type}",
        )

    if not extracted_text:
        extracted_text = "[empty response body]"

    truncated = len(extracted_text) > max_chars
    display_text = extracted_text[:max_chars] if truncated else extracted_text
    if truncated:
        display_text = f"{display_text.rstrip()}\n\n[truncated]"

    output_text = "\n".join(
        [
            f"URL: {resolved_url}",
            f"Status: {status_code or 'unknown'}",
            f"Content-Type: {content_type}",
            "",
            display_text,
        ]
    )
    return ToolResult.ok(
        tool_name=tool_name,
        output_text=output_text,
        payload={
            "requested_url": url,
            "resolved_url": resolved_url,
            "status_code": status_code,
            "content_type": content_type,
            "truncated": truncated,
        },
    )


class _HTMLTextExtractor(HTMLParser):
    """Collect readable text from a basic HTML document."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._ignored_depth = 0
        self._parts: list[str] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        del attrs
        if tag in _IGNORED_TAGS:
            self._ignored_depth += 1
            return
        if tag in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _IGNORED_TAGS:
            if self._ignored_depth > 0:
                self._ignored_depth -= 1
            return
        if tag in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._ignored_depth > 0:
            return
        if data.strip():
            self._parts.append(data)

    def as_text(self) -> str:
        return _normalize_text("".join(self._parts))


def _decode_content(raw_content: bytes, charset: str) -> str:
    try:
        return raw_content.decode(charset)
    except (LookupError, UnicodeDecodeError):
        return raw_content.decode("utf-8", errors="replace")


def _extract_text(content: str, content_type: str) -> str | None:
    if content_type in {"text/html", "application/xhtml+xml"}:
        parser = _HTMLTextExtractor()
        parser.feed(content)
        parser.close()
        return parser.as_text()

    if _is_text_content_type(content_type):
        return _normalize_text(content)

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


def _is_supported_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


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
    "fetch_url_text",
    "register_web_tools",
]
