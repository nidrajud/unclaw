from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

from unclaw.channels.cli import _TerminalAssistantStream
from unclaw.channels.cli import run_cli
from unclaw.core.command_handler import CommandHandler
from unclaw.core.session_manager import SessionManager
from unclaw.llm.base import LLMResponse
from unclaw.logs.event_bus import EventBus
from unclaw.logs.tracer import Tracer
from unclaw.settings import load_settings
from unclaw.tools.contracts import ToolResult
from unclaw.tools.registry import ToolRegistry


def test_terminal_assistant_stream_finishes_cleanly_when_stream_matches(capsys) -> None:
    stream = _TerminalAssistantStream()

    stream.write("Hel")
    stream.write("lo")
    stream.finish("Hello")

    assert capsys.readouterr().out == "Unclaw> Hello\n"


def test_terminal_assistant_stream_explains_interrupted_streams(capsys) -> None:
    stream = _TerminalAssistantStream()

    stream.write("Partial")
    stream.finish("Saved reply")

    assert capsys.readouterr().out == (
        "Unclaw> Partial\n"
        "[stream interrupted; showing the saved final reply]\n"
        "Unclaw> Saved reply\n"
    )


def test_cli_search_returns_a_natural_reply_with_compact_sources(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    project_root = _create_temp_project(tmp_path)
    settings = load_settings(project_root=project_root)
    session_manager = SessionManager.from_settings(settings)
    tracer = Tracer(
        event_bus=EventBus(),
        event_repository=session_manager.event_repository,
    )
    command_handler = CommandHandler(
        settings=settings,
        session_manager=session_manager,
        memory_manager=SimpleNamespace(
            build_or_refresh_session_summary=lambda _session_id: None
        ),
    )

    class FakeOllamaProvider:
        provider_name = "ollama"

        def __init__(
            self,
            *,
            base_url: str = "http://127.0.0.1:11434",
            default_timeout_seconds: float = 60.0,
        ) -> None:
            del base_url, default_timeout_seconds

        def chat(  # type: ignore[no-untyped-def]
            self,
            profile,
            messages,
            *,
            timeout_seconds=None,
            thinking_enabled=False,
            content_callback=None,
        ):
            del profile, messages, timeout_seconds, thinking_enabled, content_callback
            return LLMResponse(
                provider="ollama",
                model_name="qwen3.5:4b",
                content="Ollama shipped a new update with improved search grounding.",
                created_at="2026-03-13T12:00:00Z",
                finish_reason="stop",
            )

    monkeypatch.setattr("unclaw.core.orchestrator.OllamaProvider", FakeOllamaProvider)

    scripted_inputs = iter(["/search latest news about Ollama"])

    def fake_input(_prompt: str) -> str:
        try:
            return next(scripted_inputs)
        except StopIteration as exc:
            raise EOFError from exc

    monkeypatch.setattr("builtins.input", fake_input)

    try:
        exit_code = run_cli(
            session_manager=session_manager,
            command_handler=command_handler,
            memory_manager=SimpleNamespace(
                build_or_refresh_session_summary=lambda _session_id: None
            ),
            tracer=tracer,
            tool_executor=SimpleNamespace(
                list_tools=lambda: [],
                execute=lambda _tool_call: ToolResult.ok(
                    tool_name="search_web",
                    output_text=(
                        "Search query: latest news about Ollama\n"
                        "Sources fetched: 2 of 2 attempted\n"
                        "Evidence kept: 4\n"
                    ),
                    payload={
                        "query": "latest news about Ollama",
                        "summary_points": [
                            "Ollama shipped a new update with improved search grounding."
                        ],
                        "display_sources": [
                            {
                                "title": "Ollama Blog",
                                "url": "https://ollama.com/blog/search-update",
                            },
                            {
                                "title": "Release Notes",
                                "url": "https://example.com/releases/ollama-search",
                            },
                        ],
                    },
                ),
                registry=ToolRegistry(),
            ),
        )
    finally:
        session_manager.close()

    output = capsys.readouterr().out
    assert exit_code == 0
    assert (
        "Unclaw> Ollama shipped a new update with improved search grounding."
        in output
    )
    assert "Search query:" not in output
    assert "Sources fetched:" not in output
    assert "Evidence kept:" not in output
    assert "Sources:\n- Ollama Blog: https://ollama.com/blog/search-update" in output
    assert "- Release Notes: https://example.com/releases/ollama-search" in output


def _create_temp_project(tmp_path: Path) -> Path:
    source_root = Path(__file__).resolve().parents[1]
    project_root = tmp_path / "project"
    shutil.copytree(source_root / "config", project_root / "config")
    return project_root
