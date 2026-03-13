"""Local file tools for the early Unclaw runtime."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from unclaw.tools.contracts import (
    ToolCall,
    ToolDefinition,
    ToolPermissionLevel,
    ToolResult,
)
from unclaw.tools.registry import ToolRegistry

_DEFAULT_MAX_FILE_CHARS = 8_000
_DEFAULT_DIRECTORY_LIMIT = 200

READ_TEXT_FILE_DEFINITION = ToolDefinition(
    name="read_text_file",
    description="Read a local UTF-8 text file.",
    permission_level=ToolPermissionLevel.LOCAL_READ,
    arguments={
        "path": "Path to a local UTF-8 text file.",
        "max_chars": "Optional maximum number of characters to return.",
    },
)

LIST_DIRECTORY_DEFINITION = ToolDefinition(
    name="list_directory",
    description="List one local directory with depth 1 or 2.",
    permission_level=ToolPermissionLevel.LOCAL_READ,
    arguments={
        "path": "Path to a local directory.",
        "max_depth": "Optional depth, allowed values are 1 or 2.",
        "limit": "Optional maximum number of entries to include.",
    },
)


def register_file_tools(
    registry: ToolRegistry,
    *,
    project_root: Path | None = None,
    configured_roots: tuple[str, ...] = (),
) -> None:
    """Register the built-in local file tools."""
    allowed_roots = resolve_allowed_roots(
        project_root=project_root,
        configured_roots=configured_roots,
    )

    def read_handler(call: ToolCall) -> ToolResult:
        return read_text_file(call, allowed_roots=allowed_roots)

    def list_handler(call: ToolCall) -> ToolResult:
        return list_directory(call, allowed_roots=allowed_roots)

    registry.register(READ_TEXT_FILE_DEFINITION, read_handler)
    registry.register(LIST_DIRECTORY_DEFINITION, list_handler)


def read_text_file(
    call: ToolCall,
    *,
    allowed_roots: tuple[Path, ...] | None = None,
) -> ToolResult:
    """Read a local text file with a small, safe default output limit."""
    tool_name = READ_TEXT_FILE_DEFINITION.name

    try:
        path_value = _read_string_argument(call.arguments, "path")
        max_chars = _read_positive_int_argument(
            call.arguments,
            "max_chars",
            default=_DEFAULT_MAX_FILE_CHARS,
        )
    except ValueError as exc:
        return ToolResult.failure(tool_name=tool_name, error=str(exc))

    path = _resolve_path(path_value)
    access_error = _restrict_to_allowed_roots(
        tool_name=tool_name,
        path=path,
        allowed_roots=_normalize_allowed_roots(allowed_roots),
    )
    if access_error is not None:
        return access_error
    if not path.exists():
        return ToolResult.failure(
            tool_name=tool_name,
            error=f"File does not exist: {path}",
        )
    if not path.is_file():
        return ToolResult.failure(
            tool_name=tool_name,
            error=f"Path is not a file: {path}",
        )

    try:
        with path.open("r", encoding="utf-8") as handle:
            content = handle.read(max_chars + 1)
    except UnicodeDecodeError:
        return ToolResult.failure(
            tool_name=tool_name,
            error=f"File is not valid UTF-8 text: {path}",
        )
    except OSError as exc:
        return ToolResult.failure(
            tool_name=tool_name,
            error=f"Could not read file '{path}': {exc}",
        )

    truncated = len(content) > max_chars
    returned_text = content[:max_chars] if truncated else content
    display_text = returned_text or "[empty file]"
    if truncated:
        display_text = f"{returned_text.rstrip()}\n\n[truncated]"

    output_text = f"File: {path}\n\n{display_text}"
    return ToolResult.ok(
        tool_name=tool_name,
        output_text=output_text,
        payload={
            "path": str(path),
            "returned_characters": len(returned_text),
            "truncated": truncated,
        },
    )


def list_directory(
    call: ToolCall,
    *,
    allowed_roots: tuple[Path, ...] | None = None,
) -> ToolResult:
    """List one directory with a compact tree-like output."""
    tool_name = LIST_DIRECTORY_DEFINITION.name

    try:
        path_value = _read_string_argument(call.arguments, "path")
        max_depth = _read_positive_int_argument(call.arguments, "max_depth", default=1)
        limit = _read_positive_int_argument(
            call.arguments,
            "limit",
            default=_DEFAULT_DIRECTORY_LIMIT,
        )
    except ValueError as exc:
        return ToolResult.failure(tool_name=tool_name, error=str(exc))

    if max_depth not in (1, 2):
        return ToolResult.failure(
            tool_name=tool_name,
            error="Argument 'max_depth' must be 1 or 2.",
        )

    path = _resolve_path(path_value)
    access_error = _restrict_to_allowed_roots(
        tool_name=tool_name,
        path=path,
        allowed_roots=_normalize_allowed_roots(allowed_roots),
    )
    if access_error is not None:
        return access_error
    if not path.exists():
        return ToolResult.failure(
            tool_name=tool_name,
            error=f"Directory does not exist: {path}",
        )
    if not path.is_dir():
        return ToolResult.failure(
            tool_name=tool_name,
            error=f"Path is not a directory: {path}",
        )

    lines: list[str] = []
    truncated = _append_directory_lines(
        path=path,
        current_depth=1,
        max_depth=max_depth,
        lines=lines,
        limit=limit,
    )
    displayed_entries = len(lines)

    if not lines:
        lines.append("(empty)")
    if truncated:
        lines.append("... truncated ...")

    output_text = "\n".join([f"Directory: {path}", *lines])
    return ToolResult.ok(
        tool_name=tool_name,
        output_text=output_text,
        payload={
            "path": str(path),
            "max_depth": max_depth,
            "displayed_entries": displayed_entries,
            "truncated": truncated,
        },
    )


def _append_directory_lines(
    *,
    path: Path,
    current_depth: int,
    max_depth: int,
    lines: list[str],
    limit: int,
) -> bool:
    try:
        entries = sorted(path.iterdir(), key=_directory_sort_key)
    except PermissionError:
        indent = "  " * (current_depth - 1)
        lines.append(f"{indent}- [permission denied]")
        return False
    except OSError as exc:
        indent = "  " * (current_depth - 1)
        lines.append(f"{indent}- [error: {exc}]")
        return False

    for entry in entries:
        if len(lines) >= limit:
            return True

        indent = "  " * (current_depth - 1)
        is_directory = _is_directory(entry)
        lines.append(f"{indent}- {_entry_label(entry, is_directory)}")

        if is_directory and not entry.is_symlink() and current_depth < max_depth:
            truncated = _append_directory_lines(
                path=entry,
                current_depth=current_depth + 1,
                max_depth=max_depth,
                lines=lines,
                limit=limit,
            )
            if truncated:
                return True

    return False


def _directory_sort_key(path: Path) -> tuple[int, str]:
    return (0 if _is_directory(path) else 1, path.name.lower())


def _entry_label(path: Path, is_directory: bool) -> str:
    label = path.name
    if is_directory:
        label = f"{label}/"
    if path.is_symlink():
        label = f"{label} (symlink)"
    return label


def _is_directory(path: Path) -> bool:
    try:
        return path.is_dir()
    except OSError:
        return False


def _resolve_path(path_value: str) -> Path:
    return Path(path_value).expanduser().resolve()


def resolve_allowed_roots(
    *,
    project_root: Path | None,
    configured_roots: tuple[str, ...],
) -> tuple[Path, ...]:
    base_root = project_root.resolve() if project_root is not None else Path.cwd().resolve()
    roots = [base_root]

    for raw_root in configured_roots:
        candidate = Path(raw_root).expanduser()
        if not candidate.is_absolute():
            candidate = base_root / candidate
        roots.append(candidate.resolve())

    return _normalize_allowed_roots(tuple(roots))


def _normalize_allowed_roots(
    allowed_roots: tuple[Path, ...] | None,
) -> tuple[Path, ...]:
    if not allowed_roots:
        return (Path.cwd().resolve(),)
    return tuple(dict.fromkeys(root.resolve() for root in allowed_roots))


def _restrict_to_allowed_roots(
    *,
    tool_name: str,
    path: Path,
    allowed_roots: tuple[Path, ...],
) -> ToolResult | None:
    if _is_path_allowed(path, allowed_roots):
        return None

    allowed_roots_text = ", ".join(str(root) for root in allowed_roots)
    return ToolResult.failure(
        tool_name=tool_name,
        error=(
            f"Access to '{path}' is outside the allowed local roots. "
            f"Allowed roots: {allowed_roots_text}."
        ),
    )


def _is_path_allowed(path: Path, allowed_roots: tuple[Path, ...]) -> bool:
    for root in allowed_roots:
        try:
            path.relative_to(root)
        except ValueError:
            continue
        return True
    return False


def _read_string_argument(arguments: dict[str, Any], key: str) -> str:
    value = arguments.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Argument '{key}' must be a non-empty string.")
    return value.strip()


def _read_positive_int_argument(
    arguments: dict[str, Any],
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


__all__ = [
    "LIST_DIRECTORY_DEFINITION",
    "READ_TEXT_FILE_DEFINITION",
    "list_directory",
    "read_text_file",
    "register_file_tools",
]
