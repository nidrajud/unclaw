"""Runtime capability summaries for prompt honesty and future CLI UX."""

from __future__ import annotations

from dataclasses import dataclass

from unclaw.core.runtime_modes import RuntimeMode, format_runtime_mode
from unclaw.tools.file_tools import LIST_DIRECTORY_DEFINITION, READ_TEXT_FILE_DEFINITION
from unclaw.tools.registry import ToolRegistry
from unclaw.tools.web_tools import FETCH_URL_TEXT_DEFINITION, SEARCH_WEB_DEFINITION


@dataclass(frozen=True, slots=True)
class RuntimeCapabilitySummary:
    """Compact summary of the runtime's real user-facing capabilities."""

    runtime_mode: RuntimeMode
    available_builtin_tool_names: tuple[str, ...]
    local_file_read_available: bool
    local_directory_listing_available: bool
    url_fetch_available: bool
    web_search_available: bool
    memory_summary_available: bool

    @property
    def enabled_builtin_tool_count(self) -> int:
        return len(self.available_builtin_tool_names)

    @property
    def has_builtin_tools(self) -> bool:
        return self.enabled_builtin_tool_count > 0


def build_runtime_capability_summary(
    *,
    tool_registry: ToolRegistry,
    memory_summary_available: bool,
    runtime_mode: RuntimeMode,
) -> RuntimeCapabilitySummary:
    """Summarize the currently enabled built-in tools and related runtime features."""
    available_tool_names = tuple(
        tool_definition.name for tool_definition in tool_registry.list_tools()
    )
    available_tool_name_set = frozenset(available_tool_names)

    return RuntimeCapabilitySummary(
        runtime_mode=runtime_mode,
        available_builtin_tool_names=available_tool_names,
        local_file_read_available=READ_TEXT_FILE_DEFINITION.name in available_tool_name_set,
        local_directory_listing_available=(
            LIST_DIRECTORY_DEFINITION.name in available_tool_name_set
        ),
        url_fetch_available=FETCH_URL_TEXT_DEFINITION.name in available_tool_name_set,
        web_search_available=SEARCH_WEB_DEFINITION.name in available_tool_name_set,
        memory_summary_available=memory_summary_available,
    )


def build_runtime_capability_context(summary: RuntimeCapabilitySummary) -> str:
    """Build the system-context note that keeps the model honest about tools."""
    available_tool_lines = _build_available_tool_lines(summary)
    unavailable_lines = _build_unavailable_lines(summary)

    lines = [
        "Runtime capability status:",
        f"Runtime mode: {format_runtime_mode(summary.runtime_mode)}",
        f"Enabled built-in tools: {summary.enabled_builtin_tool_count}",
        "Available built-in tools:",
    ]

    if available_tool_lines:
        lines.extend(f"- {line}" for line in available_tool_lines)
    else:
        lines.append("- None.")

    if summary.memory_summary_available:
        lines.append("Other available runtime capabilities:")
        lines.append("- Session memory and summary access.")

    lines.append("Mode rules:")
    if summary.runtime_mode is RuntimeMode.AGENT:
        lines.extend(
            (
                "- Agent mode enables capability routing for plain-language turns.",
                (
                    "- The runtime may route some turns into web research before "
                    "this model call when grounded web context is needed."
                ),
                (
                    "- Local file and automation requests may still require a "
                    "follow-up question or manual slash command."
                ),
            )
        )
    else:
        lines.extend(
            (
                (
                    "- Chatbot mode = simple conversation, no autonomous web "
                    "research, no automation."
                ),
                "- Manual slash commands remain available.",
            )
        )

    lines.append("Unavailable capabilities:")
    lines.extend(f"- {line}" for line in unavailable_lines)
    lines.extend(
        (
            "Behavior rules:",
            (
                "- Do not claim you have no tool access when one or more built-in "
                "tools are available."
            ),
            (
                "- You may say you can use Unclaw built-in tools that are listed "
                "as available."
            ),
            (
                "- If tool output is already present in the conversation, treat it "
                "as retrieved context that you may summarize, compare, extract, "
                "or analyze. Do not say you cannot access it, and do not ask the "
                "user to paste it again."
            ),
            (
                "- If a capability is unavailable, say so clearly instead of "
                "implying it exists."
            ),
        )
    )
    return "\n".join(lines)


def _build_available_tool_lines(summary: RuntimeCapabilitySummary) -> tuple[str, ...]:
    lines: list[str] = []
    if summary.local_file_read_available:
        lines.append("/read <path>: read local UTF-8 files inside allowed roots.")
    if summary.local_directory_listing_available:
        lines.append("/ls [path]: list local directories inside allowed roots.")
    if summary.url_fetch_available:
        lines.append("/fetch <url>: fetch one public URL and extract text.")
    if summary.web_search_available:
        lines.append(
            "/search <query>: search the public web, read a few relevant pages, "
            "and answer naturally from grounded web context with compact sources."
        )
    return tuple(lines)


def _build_unavailable_lines(summary: RuntimeCapabilitySummary) -> tuple[str, ...]:
    lines = [
        (
            "Autonomous model-side tool execution in this chat turn. "
            "Use the listed Unclaw built-in tools or routed runtime paths instead."
        ),
        "Shell command execution.",
        "Any capability that is not listed as available above.",
    ]

    if not summary.local_file_read_available:
        lines.insert(0, "Local file read via /read <path>.")
    if not summary.local_directory_listing_available:
        lines.insert(0, "Local directory listing via /ls [path].")
    if not summary.url_fetch_available:
        lines.insert(0, "Direct URL fetch via /fetch <url>.")
    if not summary.web_search_available:
        lines.insert(0, "Web search via /search <query>.")
    if not summary.memory_summary_available:
        lines.insert(0, "Session memory and summary access.")

    return tuple(lines)


__all__ = [
    "RuntimeCapabilitySummary",
    "build_runtime_capability_context",
    "build_runtime_capability_summary",
]
