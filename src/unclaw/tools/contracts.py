"""Core contracts for Unclaw tool execution."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class ToolPermissionLevel(StrEnum):
    """Minimal permission categories for built-in tools."""

    LOCAL_READ = "local_read"
    NETWORK = "network"


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """Describe one callable tool exposed to the runtime."""

    name: str
    description: str
    permission_level: ToolPermissionLevel
    arguments: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class ToolCall:
    """One requested tool invocation."""

    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ToolResult:
    """Structured result returned by a tool handler."""

    tool_name: str
    success: bool
    output_text: str
    payload: dict[str, Any] | None = None
    error: str | None = None

    @classmethod
    def ok(
        cls,
        *,
        tool_name: str,
        output_text: str,
        payload: Mapping[str, Any] | None = None,
    ) -> ToolResult:
        return cls(
            tool_name=tool_name,
            success=True,
            output_text=output_text,
            payload=dict(payload) if payload is not None else None,
            error=None,
        )

    @classmethod
    def failure(
        cls,
        *,
        tool_name: str,
        error: str,
        output_text: str | None = None,
        payload: Mapping[str, Any] | None = None,
    ) -> ToolResult:
        resolved_output = output_text if output_text is not None else error
        return cls(
            tool_name=tool_name,
            success=False,
            output_text=resolved_output,
            payload=dict(payload) if payload is not None else None,
            error=error,
        )


type ToolHandler = Callable[[ToolCall], ToolResult]


__all__ = [
    "ToolCall",
    "ToolDefinition",
    "ToolHandler",
    "ToolPermissionLevel",
    "ToolResult",
]
