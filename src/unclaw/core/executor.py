"""Minimal tool executor wiring for the current runtime phase."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType

from unclaw.tools.contracts import ToolCall, ToolDefinition, ToolResult
from unclaw.tools.dispatcher import ToolDispatcher
from unclaw.tools.file_tools import (
    LIST_DIRECTORY_DEFINITION,
    READ_TEXT_FILE_DEFINITION,
    register_file_tools,
)
from unclaw.tools.registry import ToolRegistry
from unclaw.tools.web_tools import FETCH_URL_TEXT_DEFINITION, register_web_tools

BUILTIN_TOOL_COMMANDS = MappingProxyType(
    {
        "read": READ_TEXT_FILE_DEFINITION.name,
        "ls": LIST_DIRECTORY_DEFINITION.name,
        "fetch": FETCH_URL_TEXT_DEFINITION.name,
    }
)


def register_default_tools(registry: ToolRegistry) -> ToolRegistry:
    """Register the built-in local tools on the provided registry."""
    register_file_tools(registry)
    register_web_tools(registry)
    return registry


def create_default_tool_registry() -> ToolRegistry:
    """Create a registry populated with the initial built-in tools."""
    return register_default_tools(ToolRegistry())


def resolve_builtin_tool_command(command_name: str) -> str | None:
    """Resolve one CLI tool command name to a registered built-in tool name."""
    return BUILTIN_TOOL_COMMANDS.get(command_name)


@dataclass(slots=True)
class ToolExecutor:
    """Thin runtime-facing wrapper around the tool dispatcher."""

    registry: ToolRegistry
    dispatcher: ToolDispatcher = field(init=False)

    def __post_init__(self) -> None:
        self.dispatcher = ToolDispatcher(self.registry)

    @classmethod
    def with_default_tools(cls) -> ToolExecutor:
        return cls(registry=create_default_tool_registry())

    def list_tools(self) -> list[ToolDefinition]:
        return self.registry.list_tools()

    def execute(self, call: ToolCall) -> ToolResult:
        return self.dispatcher.dispatch(call)


def execute_tool_call(
    call: ToolCall,
    *,
    registry: ToolRegistry | None = None,
) -> ToolResult:
    """Execute one tool call using the provided or default registry."""
    active_registry = registry if registry is not None else create_default_tool_registry()
    dispatcher = ToolDispatcher(active_registry)
    return dispatcher.dispatch(call)


__all__ = [
    "BUILTIN_TOOL_COMMANDS",
    "ToolExecutor",
    "create_default_tool_registry",
    "execute_tool_call",
    "register_default_tools",
    "resolve_builtin_tool_command",
]
