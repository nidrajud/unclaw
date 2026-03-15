"""SSRF and redirect safety helpers for the web tools."""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

from unclaw.tools.web.text import is_supported_url

_BLOCKED_FETCH_HOSTS = {
    "broadcasthost",
    "instance-data",
    "instance-data.ec2.internal",
    "ip6-localhost",
    "ip6-loopback",
    "local",
    "localdomain",
    "localhost",
    "localhost.localdomain",
    "localhost6",
    "localhost6.localdomain6",
    "metadata",
    "metadata.google.internal",
}
_BLOCKED_FETCH_IPS = {"100.100.100.200"}
_BLOCKED_HOST_SUFFIXES = (
    ".home.arpa",
    ".internal",
    ".local",
    ".localdomain",
    ".localhost",
)
_BLOCKED_HOST_LABELS = {"instance-data", "localhost", "metadata"}


class BlockedFetchTargetError(ValueError):
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
    if not is_supported_url(url):
        raise BlockedFetchTargetError(
            "Only HTTP and HTTPS fetch targets are supported."
        )
    if allow_private_networks:
        return

    parsed = urlparse(url)
    hostname = parsed.hostname
    if hostname is None:
        raise BlockedFetchTargetError("Could not determine which host to fetch.")

    normalized_host = hostname.rstrip(".").lower()
    if not normalized_host or "%" in normalized_host:
        raise BlockedFetchTargetError(
            _build_blocked_fetch_message(
                target=normalized_host or hostname,
                reason="hostnames with empty values or scoped addresses are blocked",
            )
        )

    if _is_blocked_hostname(normalized_host):
        raise BlockedFetchTargetError(
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
    if hostname in _BLOCKED_FETCH_HOSTS:
        return True
    if hostname.startswith("localhost"):
        return True
    if any(hostname.endswith(suffix) for suffix in _BLOCKED_HOST_SUFFIXES):
        return True
    host_labels = hostname.split(".")
    return any(label in _BLOCKED_HOST_LABELS for label in host_labels)


def _parse_ip_address(
    value: str,
) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
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
        raise BlockedFetchTargetError(
            f"Could not resolve '{hostname}' while checking safe fetch rules: {exc}."
        ) from exc

    addresses: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = []
    for _family, _socktype, _proto, _canonname, sockaddr in address_infos:
        host = sockaddr[0]
        try:
            address = ipaddress.ip_address(host)
        except ValueError as exc:
            raise BlockedFetchTargetError(
                f"Could not validate the resolved address '{host}' for '{hostname}'."
            ) from exc
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
    raise BlockedFetchTargetError(
        _build_blocked_fetch_message(
            target=target,
            reason=f"{_normalize_checked_ip(address).compressed} is on a local or private network",
        )
    )


def _normalize_checked_ip(
    address: ipaddress.IPv4Address | ipaddress.IPv6Address,
) -> ipaddress.IPv4Address | ipaddress.IPv6Address:
    mapped_ipv4 = getattr(address, "ipv4_mapped", None)
    if mapped_ipv4 is not None:
        return mapped_ipv4
    return address


def _is_blocked_ip(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    normalized_address = _normalize_checked_ip(address)
    if normalized_address.compressed in _BLOCKED_FETCH_IPS:
        return True
    if getattr(normalized_address, "is_site_local", False):
        return True
    return not normalized_address.is_global


def _build_blocked_fetch_message(*, target: str, reason: str) -> str:
    return (
        f"Fetching '{target}' is blocked because {reason}. "
        "Only public HTTP and HTTPS targets are allowed by default. "
        "The local owner can relax `security.tools.fetch.allow_private_networks` "
        "in config/app.yaml if needed."
    )


__all__ = [
    "BlockedFetchTargetError",
    "_SafeRedirectHandler",
    "_ensure_fetch_target_allowed",
    "_is_blocked_ip",
    "_open_request",
]
