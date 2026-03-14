"""Shared /search flow and tool-history helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from unclaw.core.runtime import (
    _EMPTY_RESPONSE_REPLY,
    _RUNTIME_ERROR_REPLY,
    run_user_turn,
)
from unclaw.schemas.chat import MessageRole
from unclaw.tools.contracts import ToolCall, ToolResult
from unclaw.tools.web_tools import SEARCH_WEB_DEFINITION

_NO_FINDINGS_REPLY = "No clear findings were extracted from the retrieved sources."


@dataclass(frozen=True, slots=True)
class ResearchTurnResult:
    """Completed /search turn metadata returned to interactive channels."""

    assistant_reply: str
    tool_result: ToolResult


def is_search_tool_call(call: ToolCall) -> bool:
    """Return whether one tool call should run through the research flow."""
    return call.tool_name == SEARCH_WEB_DEFINITION.name


def persist_tool_result(
    *,
    session_manager: Any,
    session_id: str,
    result: ToolResult,
    tool_call: ToolCall | None = None,
) -> None:
    """Persist one tool result into conversation history when possible."""
    add_message = getattr(session_manager, "add_message", None)
    if not callable(add_message):
        return

    content = build_tool_history_content(result, tool_call=tool_call)
    if not content.strip():
        return

    add_message(
        MessageRole.TOOL,
        content,
        session_id=session_id,
    )


def build_tool_history_content(
    result: ToolResult,
    *,
    tool_call: ToolCall | None = None,
) -> str:
    """Build compact history content for one tool result."""
    if result.tool_name == SEARCH_WEB_DEFINITION.name:
        return _build_search_tool_history_content(result, tool_call=tool_call)

    outcome = "success" if result.success else "error"
    body = result.output_text.strip()
    if not body:
        return ""

    return "\n".join(
        [
            f"Tool: {result.tool_name}",
            f"Outcome: {outcome}",
            "",
            body,
        ]
    )


def run_search_then_answer(
    *,
    session_manager: Any,
    command_handler: Any,
    tracer: Any,
    tool_executor: Any,
    tool_call: ToolCall,
) -> ResearchTurnResult:
    """Execute bounded web retrieval, persist it as tool context, then answer naturally."""
    query = _read_tool_call_query(tool_call)
    session = session_manager.ensure_current_session()
    session_manager.add_message(
        MessageRole.USER,
        query,
        session_id=session.id,
    )

    tracer.trace_tool_started(
        session_id=session.id,
        tool_name=tool_call.tool_name,
        arguments=tool_call.arguments,
    )
    tool_started_at = perf_counter()
    tool_result = tool_executor.execute(tool_call)
    tracer.trace_tool_finished(
        session_id=session.id,
        tool_name=tool_result.tool_name,
        success=tool_result.success,
        output_length=len(tool_result.output_text),
        error=tool_result.error,
        tool_duration_ms=_elapsed_ms(tool_started_at),
    )
    persist_tool_result(
        session_manager=session_manager,
        session_id=session.id,
        result=tool_result,
        tool_call=tool_call,
    )

    if not tool_result.success:
        assistant_reply = _build_search_failure_reply(
            query=query,
            result=tool_result,
        )
        session_manager.add_message(
            MessageRole.ASSISTANT,
            assistant_reply,
            session_id=session.id,
        )
        return ResearchTurnResult(
            assistant_reply=assistant_reply,
            tool_result=tool_result,
        )

    assistant_reply = run_user_turn(
        session_manager=session_manager,
        command_handler=command_handler,
        user_input=query,
        tracer=tracer,
        tool_registry=getattr(tool_executor, "registry", None),
        assistant_reply_transform=lambda reply: append_search_sources_section(
            reply,
            payload=tool_result.payload,
        ),
    )
    return ResearchTurnResult(
        assistant_reply=assistant_reply,
        tool_result=tool_result,
    )


def append_search_sources_section(
    reply_text: str,
    *,
    payload: Mapping[str, Any] | None,
) -> str:
    """Append a compact sources section to a natural-language reply."""
    if reply_text in {_RUNTIME_ERROR_REPLY, _EMPTY_RESPONSE_REPLY}:
        return reply_text

    sources = _extract_search_sources(payload)
    if not sources:
        return reply_text

    lines = [reply_text.rstrip(), "", "Sources:"]
    for title, url in sources:
        if title:
            lines.append(f"- {title}: {url}")
        else:
            lines.append(f"- {url}")
    return "\n".join(lines)


def _build_search_tool_history_content(
    result: ToolResult,
    *,
    tool_call: ToolCall | None,
) -> str:
    payload = result.payload if isinstance(result.payload, dict) else {}
    query = _read_search_query(payload, tool_call=tool_call)
    lines = [
        f"Tool: {result.tool_name}",
        f"Outcome: {'success' if result.success else 'error'}",
    ]

    body_lines: list[str] = []
    if query:
        body_lines.append(f"Search request: {query}")

    if result.success:
        summary_points = _read_string_list(payload.get("summary_points"))
        body_lines.extend(
            [
                "",
                "Findings:",
            ]
        )
        if summary_points:
            body_lines.extend(f"- {point}" for point in summary_points)
        else:
            body_lines.append(f"- {_NO_FINDINGS_REPLY}")

        sources = _extract_search_sources(payload)
        if sources:
            body_lines.extend(
                [
                    "",
                    "Sources:",
                ]
            )
            for title, url in sources:
                if title:
                    body_lines.append(f"- {title}: {url}")
                else:
                    body_lines.append(f"- {url}")
    else:
        error_text = result.error or result.output_text.strip()
        if error_text:
            if body_lines:
                body_lines.append("")
            body_lines.append(f"Error: {error_text}")

    if not body_lines:
        fallback_body = result.output_text.strip()
        if not fallback_body:
            return ""
        body_lines.append(fallback_body)

    return "\n".join([*lines, "", *body_lines])


def _extract_search_sources(
    payload: Mapping[str, Any] | None,
) -> tuple[tuple[str, str], ...]:
    if not isinstance(payload, Mapping):
        return ()

    raw_sources = payload.get("display_sources")
    if not isinstance(raw_sources, list):
        raw_sources = payload.get("results")
        if not isinstance(raw_sources, list):
            return ()

    sources: list[tuple[str, str]] = []
    seen_urls: set[str] = set()
    for entry in raw_sources:
        if not isinstance(entry, Mapping):
            continue
        title = entry.get("title")
        url = entry.get("url")
        if not isinstance(url, str) or not url.strip():
            continue
        normalized_url = url.strip()
        if normalized_url in seen_urls:
            continue
        seen_urls.add(normalized_url)
        resolved_title = title.strip() if isinstance(title, str) else ""
        sources.append((resolved_title, normalized_url))

    return tuple(sources)


def _read_search_query(
    payload: Mapping[str, Any],
    *,
    tool_call: ToolCall | None,
) -> str:
    query = payload.get("query")
    if isinstance(query, str) and query.strip():
        return query.strip()
    if tool_call is None:
        return ""
    return _read_tool_call_query(tool_call)


def _read_tool_call_query(tool_call: ToolCall) -> str:
    query = tool_call.arguments.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Search tool calls must include a non-empty query string.")
    return query.strip()


def _build_search_failure_reply(
    *,
    query: str,
    result: ToolResult,
) -> str:
    error_text = result.error or result.output_text.strip()
    if error_text:
        return f'I could not complete a web search for "{query}": {error_text}'
    return f'I could not complete a web search for "{query}".'


def _read_string_list(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(
        item.strip()
        for item in value
        if isinstance(item, str) and item.strip()
    )


def _elapsed_ms(started_at: float) -> int:
    return max(0, round((perf_counter() - started_at) * 1000))
