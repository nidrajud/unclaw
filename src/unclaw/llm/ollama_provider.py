"""Minimal Ollama provider for local chat completions."""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from unclaw.llm.base import (
    BaseLLMProvider,
    LLMContentCallback,
    LLMConnectionError,
    LLMMessage,
    LLMProviderError,
    LLMResponse,
    LLMResponseError,
    ResolvedModelProfile,
    utc_now_iso,
)


class OllamaProvider(BaseLLMProvider):
    """Synchronous provider backed by the local Ollama HTTP API."""

    provider_name = "ollama"

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:11434",
        default_timeout_seconds: float = 60.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._default_timeout_seconds = default_timeout_seconds

    def chat(
        self,
        profile: ResolvedModelProfile,
        messages: Sequence[LLMMessage],
        *,
        timeout_seconds: float | None = None,
        thinking_enabled: bool = False,
        content_callback: LLMContentCallback | None = None,
    ) -> LLMResponse:
        self.validate_profile(profile)

        if not messages:
            raise LLMProviderError("Ollama chat requires at least one message.")

        thinking_requested = (
            thinking_enabled and profile.capabilities.thinking_supported
        )
        payload = {
            "model": profile.model_name,
            "messages": [message.as_payload() for message in messages],
            "stream": content_callback is not None,
            "options": {"temperature": profile.temperature},
            # The current runtime models thinking as a strict on/off toggle.
            # Providers that need graded reasoning levels would require a wider config change.
            "think": thinking_requested,
        }
        if content_callback is not None:
            return self._stream_chat(
                profile=profile,
                payload=payload,
                timeout_seconds=timeout_seconds,
                content_callback=content_callback,
            )

        response_payload = self._request_json(
            method="POST",
            path="/api/chat",
            payload=payload,
            timeout_seconds=timeout_seconds,
        )

        model_name = response_payload.get("model")
        if not isinstance(model_name, str) or not model_name:
            model_name = profile.model_name

        created_at = response_payload.get("created_at")
        if not isinstance(created_at, str) or not created_at:
            created_at = utc_now_iso()

        finish_reason = response_payload.get("done_reason")
        if finish_reason is not None and not isinstance(finish_reason, str):
            finish_reason = str(finish_reason)

        return LLMResponse(
            provider=self.provider_name,
            model_name=model_name,
            content=_extract_content(response_payload),
            created_at=created_at,
            finish_reason=finish_reason,
            reasoning=_extract_reasoning(response_payload),
            raw_payload=response_payload,
        )

    def is_available(self, *, timeout_seconds: float | None = None) -> bool:
        try:
            self._request_json(
                method="GET",
                path="/api/tags",
                payload=None,
                timeout_seconds=timeout_seconds,
            )
        except LLMProviderError:
            return False

        return True

    def list_models(self, *, timeout_seconds: float | None = None) -> tuple[str, ...]:
        """Return locally available Ollama model names."""

        payload = self._request_json(
            method="GET",
            path="/api/tags",
            payload=None,
            timeout_seconds=timeout_seconds,
        )
        raw_models = payload.get("models")
        if not isinstance(raw_models, list):
            raise LLMResponseError("Ollama did not return a valid model list.")

        model_names: list[str] = []
        for raw_model in raw_models:
            if not isinstance(raw_model, dict):
                continue
            model_name = raw_model.get("name")
            if not isinstance(model_name, str) or not model_name.strip():
                continue
            model_names.append(model_name.strip())

        return tuple(model_names)

    def _stream_chat(
        self,
        *,
        profile: ResolvedModelProfile,
        payload: dict[str, Any],
        timeout_seconds: float | None,
        content_callback: LLMContentCallback,
    ) -> LLMResponse:
        model_name = profile.model_name
        created_at = utc_now_iso()
        finish_reason: str | None = None
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        streamed_chunks: list[dict[str, Any]] = []

        for chunk_payload in self._request_stream_json(
            method="POST",
            path="/api/chat",
            payload=payload,
            timeout_seconds=timeout_seconds,
        ):
            streamed_chunks.append(chunk_payload)

            raw_model_name = chunk_payload.get("model")
            if isinstance(raw_model_name, str) and raw_model_name:
                model_name = raw_model_name

            raw_created_at = chunk_payload.get("created_at")
            if isinstance(raw_created_at, str) and raw_created_at:
                created_at = raw_created_at

            raw_finish_reason = chunk_payload.get("done_reason")
            if raw_finish_reason is not None:
                finish_reason = (
                    raw_finish_reason
                    if isinstance(raw_finish_reason, str)
                    else str(raw_finish_reason)
                )

            message = chunk_payload.get("message")
            if not isinstance(message, dict):
                continue

            content_delta = _extract_message_text(message, key="content")
            if content_delta is not None:
                content_parts.append(content_delta)
                content_callback(content_delta)

            reasoning_delta = _extract_message_reasoning(message)
            if reasoning_delta is not None:
                reasoning_parts.append(reasoning_delta)

        if not streamed_chunks:
            raise LLMResponseError("Ollama streaming chat returned no chunks.")

        reasoning = "".join(reasoning_parts)
        return LLMResponse(
            provider=self.provider_name,
            model_name=model_name,
            content="".join(content_parts),
            created_at=created_at,
            finish_reason=finish_reason,
            reasoning=reasoning if reasoning.strip() else None,
            raw_payload={"stream_chunks": streamed_chunks},
        )

    def _request_json(
        self,
        *,
        method: str,
        path: str,
        payload: dict[str, Any] | None,
        timeout_seconds: float | None,
    ) -> dict[str, Any]:
        request_data: bytes | None = None
        headers = {"Accept": "application/json"}

        if payload is not None:
            request_data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"

        request = Request(
            url=f"{self._base_url}{path}",
            data=request_data,
            headers=headers,
            method=method,
        )

        try:
            with urlopen(
                request,
                timeout=self._resolve_timeout(timeout_seconds),
            ) as response:
                response_body = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = _read_error_body(exc)
            raise LLMProviderError(
                f"Ollama request failed with HTTP {exc.code}: {detail}"
            ) from exc
        except URLError as exc:
            raise LLMConnectionError(
                f"Could not reach Ollama at {self._base_url}."
            ) from exc
        except OSError as exc:
            raise LLMConnectionError(
                f"Could not reach Ollama at {self._base_url}."
            ) from exc

        try:
            decoded_payload = json.loads(response_body)
        except json.JSONDecodeError as exc:
            raise LLMResponseError("Ollama returned invalid JSON.") from exc

        if not isinstance(decoded_payload, dict):
            raise LLMResponseError("Ollama returned an unexpected JSON payload.")

        return decoded_payload

    def _request_stream_json(
        self,
        *,
        method: str,
        path: str,
        payload: dict[str, Any] | None,
        timeout_seconds: float | None,
    ) -> Iterator[dict[str, Any]]:
        request_data: bytes | None = None
        headers = {"Accept": "application/json"}

        if payload is not None:
            request_data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"

        request = Request(
            url=f"{self._base_url}{path}",
            data=request_data,
            headers=headers,
            method=method,
        )

        try:
            with urlopen(
                request,
                timeout=self._resolve_timeout(timeout_seconds),
            ) as response:
                for raw_line in response:
                    line = raw_line.decode("utf-8").strip()
                    if not line:
                        continue
                    try:
                        decoded_payload = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise LLMResponseError(
                            "Ollama returned invalid JSON while streaming."
                        ) from exc
                    if not isinstance(decoded_payload, dict):
                        raise LLMResponseError(
                            "Ollama returned an unexpected JSON payload while streaming."
                        )
                    yield decoded_payload
        except HTTPError as exc:
            detail = _read_error_body(exc)
            raise LLMProviderError(
                f"Ollama request failed with HTTP {exc.code}: {detail}"
            ) from exc
        except URLError as exc:
            raise LLMConnectionError(
                f"Could not reach Ollama at {self._base_url}."
            ) from exc
        except OSError as exc:
            raise LLMConnectionError(
                f"Could not reach Ollama at {self._base_url}."
            ) from exc

    def _resolve_timeout(self, timeout_seconds: float | None) -> float:
        if timeout_seconds is None:
            return self._default_timeout_seconds
        return timeout_seconds


def _extract_content(payload: dict[str, Any]) -> str:
    message = payload.get("message")
    if not isinstance(message, dict):
        raise LLMResponseError("Ollama response did not include a message object.")

    content = _extract_message_text(message, key="content")
    if content is None:
        raise LLMResponseError("Ollama response did not include message content.")

    return content


def _extract_reasoning(payload: dict[str, Any]) -> str | None:
    message = payload.get("message")
    if not isinstance(message, dict):
        return None

    return _extract_message_reasoning(message)


def _extract_message_reasoning(message: dict[str, Any]) -> str | None:
    for key in ("thinking", "reasoning"):
        reasoning = _extract_message_text(message, key=key)
        if reasoning is not None:
            return reasoning
    return None


def _extract_message_text(message: dict[str, Any], *, key: str) -> str | None:
    value = message.get(key)
    if not isinstance(value, str):
        return None
    return value


def _read_error_body(error: HTTPError) -> str:
    try:
        body = error.read().decode("utf-8").strip()
    except OSError:
        body = ""

    if body:
        return body
    return error.reason if isinstance(error.reason, str) else "request failed"
