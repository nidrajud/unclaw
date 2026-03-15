from __future__ import annotations

import json

from unclaw.llm import ollama_provider
from unclaw.llm.base import LLMMessage, LLMRole, ModelCapabilities, ResolvedModelProfile


def test_chat_sends_boolean_think_flag_and_captures_reasoning(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_urlopen(request, timeout):  # type: ignore[no-untyped-def]
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return _FakeJsonResponse(
            json.dumps(
                {
                    "model": "qwen3.5:4b",
                    "created_at": "2026-03-13T12:00:00Z",
                    "done_reason": "stop",
                    "message": {
                        "content": "Hello",
                        "thinking": "Need a short answer.",
                    },
                }
            )
        )

    monkeypatch.setattr(ollama_provider, "urlopen", fake_urlopen)

    provider = ollama_provider.OllamaProvider()
    response = provider.chat(
        profile=_build_profile(thinking_supported=True),
        messages=[LLMMessage(role=LLMRole.USER, content="hi")],
        thinking_enabled=True,
    )

    assert captured["payload"] == {
        "model": "qwen3.5:4b",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": False,
        "options": {"temperature": 0.3},
        "think": True,
    }
    assert response.content == "Hello"
    assert response.reasoning == "Need a short answer."
    assert response.finish_reason == "stop"


def test_chat_forces_think_off_for_profiles_without_thinking_support(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_urlopen(request, timeout):  # type: ignore[no-untyped-def]
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return _FakeJsonResponse(
            json.dumps(
                {
                    "model": "llama3.2:3b",
                    "created_at": "2026-03-13T12:00:00Z",
                    "done_reason": "stop",
                    "message": {"content": "Fast reply"},
                }
            )
        )

    monkeypatch.setattr(ollama_provider, "urlopen", fake_urlopen)

    provider = ollama_provider.OllamaProvider()
    response = provider.chat(
        profile=_build_profile(
            model_name="llama3.2:3b",
            thinking_supported=False,
        ),
        messages=[LLMMessage(role=LLMRole.USER, content="hi")],
        thinking_enabled=True,
    )

    assert captured["payload"]["think"] is False
    assert response.reasoning is None


def test_chat_streams_real_incremental_content_when_callback_is_provided(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_urlopen(request, timeout):  # type: ignore[no-untyped-def]
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return _FakeStreamResponse(
            (
                {
                    "model": "qwen3.5:4b",
                    "created_at": "2026-03-13T12:00:00Z",
                    "message": {"thinking": "Need "},
                },
                {
                    "model": "qwen3.5:4b",
                    "created_at": "2026-03-13T12:00:01Z",
                    "message": {"thinking": "speed."},
                },
                {
                    "model": "qwen3.5:4b",
                    "created_at": "2026-03-13T12:00:01Z",
                    "message": {"content": "Hel"},
                },
                {
                    "model": "qwen3.5:4b",
                    "created_at": "2026-03-13T12:00:01Z",
                    "message": {"content": "lo"},
                },
                {
                    "model": "qwen3.5:4b",
                    "created_at": "2026-03-13T12:00:02Z",
                    "done_reason": "stop",
                },
            )
        )

    monkeypatch.setattr(ollama_provider, "urlopen", fake_urlopen)

    streamed_chunks: list[str] = []
    provider = ollama_provider.OllamaProvider()
    response = provider.chat(
        profile=_build_profile(thinking_supported=True),
        messages=[LLMMessage(role=LLMRole.USER, content="hi")],
        thinking_enabled=True,
        content_callback=streamed_chunks.append,
    )

    assert captured["payload"]["stream"] is True
    assert captured["payload"]["think"] is True
    assert streamed_chunks == ["Hel", "lo"]
    assert response.content == "Hello"
    assert response.reasoning == "Need speed."
    assert response.finish_reason == "stop"


def _build_profile(
    *,
    model_name: str = "qwen3.5:4b",
    thinking_supported: bool,
) -> ResolvedModelProfile:
    return ResolvedModelProfile(
        name="main",
        provider="ollama",
        model_name=model_name,
        temperature=0.3,
        capabilities=ModelCapabilities(
            thinking_supported=thinking_supported,
            tool_mode="json_plan",
            supports_tools=True,
            supports_native_tool_calling=False,
            supports_agent_mode=True,
        ),
    )


class _FakeJsonResponse:
    def __init__(self, body: str) -> None:
        self._body = body

    def __enter__(self) -> _FakeJsonResponse:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:  # type: ignore[no-untyped-def]
        return False

    def read(self) -> bytes:
        return self._body.encode("utf-8")


class _FakeStreamResponse:
    def __init__(self, payloads: tuple[dict[str, object], ...]) -> None:
        self._lines = [
            json.dumps(payload, sort_keys=True).encode("utf-8") + b"\n"
            for payload in payloads
        ]

    def __enter__(self) -> _FakeStreamResponse:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:  # type: ignore[no-untyped-def]
        return False

    def __iter__(self):
        return iter(self._lines)
