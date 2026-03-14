"""Lightweight web tools for the early Unclaw runtime."""

from __future__ import annotations

import ipaddress
import socket
from collections.abc import Mapping
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, unquote, urlencode, urljoin, urlparse
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
_DEFAULT_MAX_SEARCH_RESULTS = 5
_MAX_SEARCH_RESULTS = 10
_DUCKDUCKGO_HTML_SEARCH_URL = "https://html.duckduckgo.com/html/"
_SEARCH_PROVIDER_NAME = "DuckDuckGo HTML"
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
    description="Search the public web and return compact result summaries.",
    permission_level=ToolPermissionLevel.NETWORK,
    arguments={
        "query": "Plain-language search query.",
        "max_results": "Optional maximum number of results to return, between 1 and 10.",
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
        _ensure_fetch_target_allowed(
            url,
            allow_private_networks=allow_private_networks,
        )
    except _BlockedFetchTargetError as exc:
        return ToolResult.failure(tool_name=tool_name, error=str(exc))

    request = Request(
        url,
        headers={
            "User-Agent": "unclaw/0.1 (+https://local-first.invalid)",
            "Accept": "text/plain, text/html, application/json;q=0.9, */*;q=0.1",
        },
    )

    try:
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


def search_web(call: ToolCall) -> ToolResult:
    """Search the public web and return compact structured search results."""
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

    request_url = f"{_DUCKDUCKGO_HTML_SEARCH_URL}?{urlencode({'q': query})}"

    try:
        _ensure_fetch_target_allowed(
            request_url,
            allow_private_networks=False,
        )
    except _BlockedFetchTargetError as exc:
        return ToolResult.failure(tool_name=tool_name, error=str(exc))

    request = Request(
        request_url,
        headers={
            "User-Agent": "unclaw/0.1 (+https://local-first.invalid)",
            "Accept": "text/html, application/xhtml+xml;q=0.9, */*;q=0.1",
        },
    )

    try:
        with _open_request(
            request,
            timeout_seconds=timeout_seconds,
            allow_private_networks=False,
        ) as response:
            content_type = response.headers.get_content_type()
            charset = response.headers.get_content_charset() or "utf-8"
            raw_content = response.read(_MAX_FETCH_BYTES + 1)
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

    if len(raw_content) > _MAX_FETCH_BYTES:
        raw_content = raw_content[:_MAX_FETCH_BYTES]

    if content_type not in {"text/html", "application/xhtml+xml"}:
        return ToolResult.failure(
            tool_name=tool_name,
            error=f"Search provider returned unsupported content type: {content_type}",
        )

    response_text = _decode_content(raw_content, charset)
    results = _parse_duckduckgo_html_results(response_text, max_results=max_results)
    output_text = _format_search_results(query=query, results=results)

    return ToolResult.ok(
        tool_name=tool_name,
        output_text=output_text,
        payload={
            "query": query,
            "provider": _SEARCH_PROVIDER_NAME,
            "result_count": len(results),
            "results": [dict(result) for result in results],
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


def _parse_duckduckgo_html_results(
    response_text: str,
    *,
    max_results: int,
) -> list[dict[str, str]]:
    parser = _DuckDuckGoHTMLSearchParser()
    parser.feed(response_text)
    parser.close()
    return parser.results[:max_results]


def _format_search_results(
    *,
    query: str,
    results: list[dict[str, str]],
) -> str:
    lines = [
        f"Search query: {query}",
        f"Provider: {_SEARCH_PROVIDER_NAME}",
        f"Results: {len(results)}",
        "",
    ]
    if not results:
        lines.append("No public web results found.")
        return "\n".join(lines)

    for index, result in enumerate(results, start=1):
        lines.append(f"{index}. {result['title']}")
        lines.append(f"   URL: {result['url']}")
        snippet = result["snippet"] or "[no snippet]"
        lines.append(f"   Snippet: {snippet}")
        if index != len(results):
            lines.append("")

    return "\n".join(lines)


def _is_supported_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


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
