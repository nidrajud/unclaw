"""Small central registry for tool definitions and handlers."""

from __future__ import annotations

from dataclasses import dataclass

from unclaw.tools.contracts import ToolDefinition, ToolHandler


@dataclass(frozen=True, slots=True)
class RegisteredTool:
    """A tool definition paired with its execution handler."""

    definition: ToolDefinition
    handler: ToolHandler


class ToolRegistry:
    """Store and expose the runtime's available tools."""

    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}

    def register(self, definition: ToolDefinition, handler: ToolHandler) -> None:
        name = definition.name.strip()
        if not name:
            raise ValueError("Tool name must be a non-empty string.")
        if name in self._tools:
            raise ValueError(f"Tool '{name}' is already registered.")

        self._tools[name] = RegisteredTool(definition=definition, handler=handler)

    def get(self, tool_name: str) -> RegisteredTool | None:
        return self._tools.get(tool_name)

    def list_tools(self) -> list[ToolDefinition]:
        return [registered_tool.definition for registered_tool in self._tools.values()]


__all__ = ["RegisteredTool", "ToolRegistry"]
