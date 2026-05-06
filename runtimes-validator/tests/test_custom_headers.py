from __future__ import annotations

from unittest.mock import MagicMock, patch

from runtimes_validator.engines.base import EngineConfig
from runtimes_validator.engines.llamacpp import LlamaCppEngine
from runtimes_validator.engines.ollama import OllamaEngine
from runtimes_validator.engines.vllm import VllmEngine


# --- Helpers ---


def _fake_chat_response() -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Hello!",
                    "tool_calls": None,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    }
    return resp


def _fake_health_ok() -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    return resp


# --- chat() sends custom headers ---


@patch("runtimes_validator.engines.openai_compat.requests.post")
def test_chat_sends_custom_headers(mock_post: MagicMock):
    mock_post.return_value = _fake_chat_response()
    headers = {"RITS_API_KEY": "tok123", "X-Custom": "value"}
    engine = VllmEngine(EngineConfig(model_id="m", extra={"headers": headers}))

    engine.chat([{"role": "user", "content": "hi"}])

    call_kwargs = mock_post.call_args.kwargs
    assert call_kwargs["headers"] == headers


# --- health_check() sends custom headers ---


@patch("runtimes_validator.engines.openai_compat.requests.get")
def test_health_check_sends_custom_headers(mock_get: MagicMock):
    mock_get.return_value = _fake_health_ok()
    headers = {"RITS_API_KEY": "tok123"}
    engine = LlamaCppEngine(EngineConfig(extra={"headers": headers}))

    engine.health_check()

    call_kwargs = mock_get.call_args.kwargs
    assert call_kwargs["headers"] == headers


@patch("runtimes_validator.engines.openai_compat.requests.get")
def test_ollama_health_check_sends_custom_headers(mock_get: MagicMock):
    mock_get.return_value = _fake_health_ok()
    headers = {"RITS_API_KEY": "tok123"}
    engine = OllamaEngine(EngineConfig(extra={"headers": headers}))

    engine.health_check()

    call_kwargs = mock_get.call_args.kwargs
    assert call_kwargs["headers"] == headers
    # Ollama uses /api/version, not /health
    assert "/api/version" in mock_get.call_args.args[0]


# --- no headers when extra is empty (backward compat) ---


@patch("runtimes_validator.engines.openai_compat.requests.post")
def test_chat_no_headers_when_extra_empty(mock_post: MagicMock):
    mock_post.return_value = _fake_chat_response()
    engine = VllmEngine(EngineConfig(model_id="m"))

    engine.chat([{"role": "user", "content": "hi"}])

    call_kwargs = mock_post.call_args.kwargs
    assert call_kwargs["headers"] is None


@patch("runtimes_validator.engines.openai_compat.requests.get")
def test_health_check_no_headers_when_extra_empty(mock_get: MagicMock):
    mock_get.return_value = _fake_health_ok()
    engine = LlamaCppEngine(EngineConfig())

    engine.health_check()

    call_kwargs = mock_get.call_args.kwargs
    assert call_kwargs["headers"] is None
