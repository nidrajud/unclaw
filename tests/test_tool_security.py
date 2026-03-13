from __future__ import annotations

import shutil
import socket
from pathlib import Path

import yaml

from unclaw.core.executor import ToolExecutor
from unclaw.settings import load_settings
from unclaw.tools.contracts import ToolCall


class _FakeHeaders:
    def get_content_type(self) -> str:
        return "text/plain"

    def get_content_charset(self) -> str:
        return "utf-8"


class _FakeResponse:
    def __init__(self, *, url: str, body: str) -> None:
        self._url = url
        self._body = body.encode("utf-8")
        self.headers = _FakeHeaders()
        self.status = 200

    def geturl(self) -> str:
        return self._url

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            return self._body
        return self._body[:size]

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:  # type: ignore[no-untyped-def]
        del exc_type, exc, traceback
        return False


def test_read_tool_rejects_paths_outside_allowed_roots(tmp_path: Path) -> None:
    project_root = _create_temp_project(tmp_path)
    settings = load_settings(project_root=project_root)
    outside_file = tmp_path / "outside.txt"
    outside_file.write_text("secret\n", encoding="utf-8")
    executor = ToolExecutor.with_default_tools(settings)

    result = executor.execute(
        ToolCall(
            tool_name="read_text_file",
            arguments={"path": str(outside_file)},
        )
    )

    assert result.success is False
    assert result.error is not None
    assert "outside the allowed local roots" in result.error


def test_list_tool_rejects_directories_outside_allowed_roots(tmp_path: Path) -> None:
    project_root = _create_temp_project(tmp_path)
    settings = load_settings(project_root=project_root)
    outside_directory = tmp_path / "outside-dir"
    outside_directory.mkdir()
    executor = ToolExecutor.with_default_tools(settings)

    result = executor.execute(
        ToolCall(
            tool_name="list_directory",
            arguments={"path": str(outside_directory)},
        )
    )

    assert result.success is False
    assert result.error is not None
    assert "outside the allowed local roots" in result.error


def test_fetch_tool_blocks_private_network_targets_by_default(
    monkeypatch,
    tmp_path: Path,
) -> None:
    project_root = _create_temp_project(tmp_path)
    settings = load_settings(project_root=project_root)
    executor = ToolExecutor.with_default_tools(settings)

    monkeypatch.setattr(
        "unclaw.tools.web_tools.socket.getaddrinfo",
        lambda host, port, type=socket.SOCK_STREAM: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", port))
        ],
    )

    result = executor.execute(
        ToolCall(
            tool_name="fetch_url_text",
            arguments={"url": "http://blocked.test"},
        )
    )

    assert result.success is False
    assert result.error is not None
    assert "blocked because" in result.error


def test_fetch_tool_allows_public_urls(monkeypatch, tmp_path: Path) -> None:
    project_root = _create_temp_project(tmp_path)
    settings = load_settings(project_root=project_root)
    executor = ToolExecutor.with_default_tools(settings)

    monkeypatch.setattr(
        "unclaw.tools.web_tools.socket.getaddrinfo",
        lambda host, port, type=socket.SOCK_STREAM: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port))
        ],
    )
    monkeypatch.setattr(
        "unclaw.tools.web_tools._open_request",
        lambda request, timeout_seconds, allow_private_networks: _FakeResponse(
            url=request.full_url,
            body="public content",
        ),
    )

    result = executor.execute(
        ToolCall(
            tool_name="fetch_url_text",
            arguments={"url": "https://example.com/docs"},
        )
    )

    assert result.success is True
    assert "public content" in result.output_text


def test_fetch_tool_can_opt_in_to_private_network_access(
    monkeypatch,
    tmp_path: Path,
) -> None:
    project_root = _create_temp_project(tmp_path)
    app_config_path = project_root / "config" / "app.yaml"
    payload = yaml.safe_load(app_config_path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    payload["security"]["tools"]["fetch"]["allow_private_networks"] = True
    app_config_path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    settings = load_settings(project_root=project_root)
    executor = ToolExecutor.with_default_tools(settings)

    monkeypatch.setattr(
        "unclaw.tools.web_tools._open_request",
        lambda request, timeout_seconds, allow_private_networks: _FakeResponse(
            url=request.full_url,
            body="private content",
        ),
    )

    result = executor.execute(
        ToolCall(
            tool_name="fetch_url_text",
            arguments={"url": "http://127.0.0.1:11434/api/tags"},
        )
    )

    assert result.success is True
    assert "private content" in result.output_text


def _create_temp_project(tmp_path: Path) -> Path:
    source_root = Path(__file__).resolve().parents[1]
    project_root = tmp_path / "project"
    shutil.copytree(source_root / "config", project_root / "config")
    return project_root
