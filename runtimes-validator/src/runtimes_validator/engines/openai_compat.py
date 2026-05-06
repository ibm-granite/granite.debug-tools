from __future__ import annotations

import json
from typing import Any, Iterator

import requests

from runtimes_validator.domain.models import EngineInfo, EngineTimeoutError
from runtimes_validator.engines.base import AbstractEngine, EngineConfig

_DEFAULT_HTTP_TIMEOUT = 120


class OpenAICompatibleEngine(AbstractEngine):
    """Base for engines that expose an OpenAI-compatible REST API.

    Subclasses must set ``_DEFAULT_PORT`` and implement ``engine_id()``,
    ``start()``, and ``stop()``.
    """

    _DEFAULT_PORT: int
    _HEALTH_ENDPOINT: str = "/health"

    def __init__(self, config: EngineConfig) -> None:
        self._config = config
        self._base_url = config.base_url or f"http://localhost:{self._DEFAULT_PORT}"
        self._headers: dict[str, str] = config.extra.get("headers", {})
        self._http_timeout: int = config.extra.get("request_timeout", _DEFAULT_HTTP_TIMEOUT)
        self._last_timeout: bool = False
        self._timeout_observed: bool = False

    def reset_timeout_observed(self) -> None:
        """Clear timeout state before running an independent validation test."""
        self._last_timeout = False
        self._timeout_observed = False

    def timed_out_since_last_check(self) -> bool:
        """Return whether any request timed out since the last reset."""
        return self._timeout_observed

    def _mark_timeout(self) -> None:
        self._last_timeout = True
        self._timeout_observed = True

    def _timeout_error(self) -> EngineTimeoutError:
        return EngineTimeoutError(
            f"Request timed out after {self._http_timeout}s. "
            f"The engine may need more resources or a longer timeout "
            f"(configure 'request_timeout' in engine extra). "
            f"For large models in managed mode, consider launching "
            f"the engine externally and running in external mode instead."
        )

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if self._config.model_id is not None:
            payload["model"] = self._config.model_id
        if tools is not None:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice

        self._last_timeout = False
        try:
            resp = requests.post(
                f"{self._base_url}/v1/chat/completions",
                json=payload,
                headers=self._headers or None,
                timeout=self._http_timeout,
            )
        except requests.Timeout as exc:
            self._mark_timeout()
            raise self._timeout_error() from exc
        resp.raise_for_status()
        body = resp.json()

        choice = body["choices"][0]
        msg = choice["message"]

        return {
            "role": msg.get("role", "assistant"),
            "content": msg.get("content"),
            "tool_calls": msg.get("tool_calls"),
            "finish_reason": choice.get("finish_reason"),
            "usage": body.get("usage"),
        }

    def chat_stream(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> Iterator[dict[str, Any]]:
        """Stream a chat completion request via SSE.

        Yields parsed JSON dicts for each ``data:`` line in the SSE stream.
        """
        payload: dict[str, Any] = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if self._config.model_id is not None:
            payload["model"] = self._config.model_id
        if tools is not None:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice

        self._last_timeout = False
        try:
            resp = requests.post(
                f"{self._base_url}/v1/chat/completions",
                json=payload,
                headers=self._headers or None,
                timeout=self._http_timeout,
                stream=True,
            )
        except requests.Timeout as exc:
            self._mark_timeout()
            raise self._timeout_error() from exc
        resp.raise_for_status()

        try:
            for line in resp.iter_lines(decode_unicode=True):
                if line and line.startswith("data: "):
                    data = line[len("data: ") :]
                    if data.strip() == "[DONE]":
                        break
                    yield json.loads(data)
        except requests.Timeout as exc:
            self._mark_timeout()
            raise self._timeout_error() from exc

    def health_check(self) -> bool:
        try:
            resp = requests.get(
                f"{self._base_url}{self._HEALTH_ENDPOINT}",
                headers=self._headers or None,
                timeout=10,
            )
            return bool(resp.status_code == 200)
        except requests.RequestException:
            return False

    def get_info(self) -> EngineInfo:
        return EngineInfo(
            engine_id=self.engine_id(),
            version="unknown",
            mode=self._config.mode,
            base_url=self._base_url,
        )
