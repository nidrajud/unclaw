"""Resolve and execute registered tool calls."""

from __future__ import annotations

from dataclasses import dataclass

from unclaw.tools.contracts import ToolCall, ToolResult
from unclaw.tools.registry import ToolRegistry


@dataclass(slots=True)
class ToolDispatcher:
    """Execute one tool call against a registry."""

    registry: ToolRegistry

    def dispatch(self, call: ToolCall) -> ToolResult:
        registered_tool = self.registry.get(call.tool_name)
        if registered_tool is None:
            return ToolResult.failure(
                tool_name=call.tool_name,
                error=f"Unknown tool '{call.tool_name}'.",
            )

        try:
            result = registered_tool.handler(call)
        except Exception as exc:  # pragma: no cover - defensive path
            return ToolResult.failure(
                tool_name=call.tool_name,
                error=f"Tool '{call.tool_name}' failed: {exc}",
            )

        if not isinstance(result, ToolResult):
            return ToolResult.failure(
                tool_name=call.tool_name,
                error=f"Tool '{call.tool_name}' returned an invalid result object.",
            )

        if result.tool_name != call.tool_name:
            return ToolResult.failure(
                tool_name=call.tool_name,
                error=(
                    f"Tool '{call.tool_name}' returned a mismatched result "
                    f"for '{result.tool_name}'."
                ),
            )

        return result


__all__ = ["ToolDispatcher"]
