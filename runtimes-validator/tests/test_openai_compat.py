from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from runtimes_validator.domain.models import EngineTimeoutError
from runtimes_validator.engines.base import EngineConfig
from runtimes_validator.engines.llamacpp import LlamaCppEngine
from runtimes_validator.engines.vllm import VllmEngine


# --- Helpers ---


def _fake_chat_response() -> MagicMock:
    """Return a mock requests.Response for /v1/chat/completions."""
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


def _fake_stream_response(lines: list[str]) -> MagicMock:
    """Return a mock streaming response whose iter_lines yields *lines*."""
    resp = MagicMock()
    resp.status_code = 200
    resp.iter_lines.return_value = iter(lines)
    return resp


def _fake_health_ok() -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    return resp


def _fake_health_down() -> MagicMock:
    resp = MagicMock()
    resp.status_code = 503
    return resp


# --- Default port / base_url ---


def test_llamacpp_default_port():
    engine = LlamaCppEngine(EngineConfig())
    assert engine._base_url == "http://localhost:8080"


def test_vllm_default_port():
    engine = VllmEngine(EngineConfig())
    assert engine._base_url == "http://localhost:8000"


def test_custom_base_url_overrides_default():
    engine = VllmEngine(EngineConfig(base_url="http://gpu-box:9000"))
    assert engine._base_url == "http://gpu-box:9000"


# --- engine_id / get_info ---


def test_engine_id():
    assert LlamaCppEngine(EngineConfig()).engine_id() == "llamacpp"
    assert VllmEngine(EngineConfig()).engine_id() == "vllm"


def test_get_info():
    config = EngineConfig(mode="external", base_url="http://myhost:8080")
    info = LlamaCppEngine(config).get_info()
    assert info.engine_id == "llamacpp"
    assert info.mode == "external"
    assert info.base_url == "http://myhost:8080"


# --- chat() ---


@patch("runtimes_validator.engines.openai_compat.requests.post")
def test_chat_basic(mock_post: MagicMock):
    mock_post.return_value = _fake_chat_response()
    engine = VllmEngine(EngineConfig(model_id="granite-3.3-8b"))
    messages = [{"role": "user", "content": "Hi"}]

    result = engine.chat(messages)

    assert result["role"] == "assistant"
    assert result["content"] == "Hello!"
    assert result["finish_reason"] == "stop"
    assert result["usage"]["total_tokens"] == 7

    # Verify the HTTP call
    mock_post.assert_called_once()
    call_kwargs = mock_post.call_args
    payload = call_kwargs.kwargs["json"]
    assert payload["messages"] == messages
    assert payload["model"] == "granite-3.3-8b"
    assert payload["temperature"] == 0.0
    assert payload["max_tokens"] == 512


@patch("runtimes_validator.engines.openai_compat.requests.post")
def test_chat_includes_tools_when_provided(mock_post: MagicMock):
    mock_post.return_value = _fake_chat_response()
    engine = LlamaCppEngine(EngineConfig(model_id="my-model"))
    tools = [{"type": "function", "function": {"name": "get_weather"}}]

    engine.chat([{"role": "user", "content": "weather?"}], tools=tools, tool_choice="auto")

    payload = mock_post.call_args.kwargs["json"]
    assert payload["tools"] == tools
    assert payload["tool_choice"] == "auto"


@patch("runtimes_validator.engines.openai_compat.requests.post")
def test_chat_omits_tools_when_none(mock_post: MagicMock):
    mock_post.return_value = _fake_chat_response()
    engine = VllmEngine(EngineConfig(model_id="m"))

    engine.chat([{"role": "user", "content": "hello"}])

    payload = mock_post.call_args.kwargs["json"]
    assert "tools" not in payload
    assert "tool_choice" not in payload


@patch("runtimes_validator.engines.openai_compat.requests.post")
def test_chat_omits_model_when_model_id_is_none(mock_post: MagicMock):
    mock_post.return_value = _fake_chat_response()
    engine = LlamaCppEngine(EngineConfig())  # no model_id

    engine.chat([{"role": "user", "content": "hi"}])

    payload = mock_post.call_args.kwargs["json"]
    assert "model" not in payload


@patch("runtimes_validator.engines.openai_compat.requests.post")
def test_chat_posts_to_correct_url(mock_post: MagicMock):
    mock_post.return_value = _fake_chat_response()
    engine = VllmEngine(EngineConfig(base_url="http://gpu:8000", model_id="m"))

    engine.chat([{"role": "user", "content": "hi"}])

    url = mock_post.call_args.args[0]
    assert url == "http://gpu:8000/v1/chat/completions"


# --- health_check() ---


@patch("runtimes_validator.engines.openai_compat.requests.get")
def test_health_check_returns_true_on_200(mock_get: MagicMock):
    mock_get.return_value = _fake_health_ok()
    engine = VllmEngine(EngineConfig())
    assert engine.health_check() is True


@patch("runtimes_validator.engines.openai_compat.requests.get")
def test_health_check_returns_false_on_non_200(mock_get: MagicMock):
    mock_get.return_value = _fake_health_down()
    engine = LlamaCppEngine(EngineConfig())
    assert engine.health_check() is False


@patch("runtimes_validator.engines.openai_compat.requests.get")
def test_health_check_returns_false_on_connection_error(mock_get: MagicMock):
    mock_get.side_effect = requests.ConnectionError("refused")
    engine = VllmEngine(EngineConfig())
    assert engine.health_check() is False


@patch("runtimes_validator.engines.openai_compat.requests.get")
def test_health_check_returns_false_on_timeout(mock_get: MagicMock):
    mock_get.side_effect = requests.Timeout("timed out")
    engine = VllmEngine(EngineConfig())
    assert engine.health_check() is False


@patch("runtimes_validator.engines.openai_compat.requests.get")
def test_health_check_hits_correct_url(mock_get: MagicMock):
    mock_get.return_value = _fake_health_ok()
    engine = LlamaCppEngine(EngineConfig(base_url="http://myhost:8080"))

    engine.health_check()

    url = mock_get.call_args_list[0].args[0]
    assert url == "http://myhost:8080/health"


def test_vllm_start_rejects_external_mode():
    engine = VllmEngine(EngineConfig(mode="external"))
    try:
        engine.start("model")
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_vllm_stop_noop_when_no_process():
    engine = VllmEngine(EngineConfig())
    engine.stop()  # Should not raise


# --- chat_stream() ---


@patch("runtimes_validator.engines.openai_compat.requests.post")
def test_chat_stream_yields_parsed_chunks(mock_post: MagicMock):
    sse_lines = [
        'data: {"choices":[{"delta":{"content":"Hi"}}]}',
        'data: {"choices":[{"delta":{"content":"!"},"finish_reason":"stop"}]}',
        "data: [DONE]",
    ]
    mock_post.return_value = _fake_stream_response(sse_lines)
    engine = VllmEngine(EngineConfig(model_id="m"))

    chunks = list(engine.chat_stream([{"role": "user", "content": "hello"}]))

    assert len(chunks) == 2
    assert chunks[0]["choices"][0]["delta"]["content"] == "Hi"
    assert chunks[1]["choices"][0]["finish_reason"] == "stop"


@patch("runtimes_validator.engines.openai_compat.requests.post")
def test_chat_stream_sends_correct_payload(mock_post: MagicMock):
    mock_post.return_value = _fake_stream_response(["data: [DONE]"])
    engine = VllmEngine(EngineConfig(model_id="granite-3.3-8b"))
    tools = [{"type": "function", "function": {"name": "f"}}]

    list(
        engine.chat_stream(
            [{"role": "user", "content": "hi"}],
            tools=tools,
            tool_choice="required",
            temperature=0.5,
            max_tokens=128,
        )
    )

    payload = mock_post.call_args.kwargs["json"]
    assert payload["stream"] is True
    assert payload["model"] == "granite-3.3-8b"
    assert payload["tools"] == tools
    assert payload["tool_choice"] == "required"
    assert payload["temperature"] == 0.5
    assert payload["max_tokens"] == 128


@patch("runtimes_validator.engines.openai_compat.requests.post")
def test_chat_stream_posts_to_correct_url(mock_post: MagicMock):
    mock_post.return_value = _fake_stream_response(["data: [DONE]"])
    engine = LlamaCppEngine(EngineConfig(base_url="http://gpu:8080", model_id="m"))

    list(engine.chat_stream([{"role": "user", "content": "hi"}]))

    url = mock_post.call_args.args[0]
    assert url == "http://gpu:8080/v1/chat/completions"
    assert mock_post.call_args.kwargs["stream"] is True


@patch("runtimes_validator.engines.openai_compat.requests.post")
def test_chat_stream_skips_empty_lines(mock_post: MagicMock):
    sse_lines = [
        "",
        'data: {"choices":[{"delta":{"content":"ok"}}]}',
        "",
        "data: [DONE]",
    ]
    mock_post.return_value = _fake_stream_response(sse_lines)
    engine = VllmEngine(EngineConfig(model_id="m"))

    chunks = list(engine.chat_stream([{"role": "user", "content": "hi"}]))

    assert len(chunks) == 1
    assert chunks[0]["choices"][0]["delta"]["content"] == "ok"


# --- LlamaCppEngine.tokenize() ---


@patch("runtimes_validator.engines.llamacpp.requests.post")
def test_tokenize_basic(mock_post: MagicMock):
    resp = MagicMock()
    resp.json.return_value = {"tokens": [1, 2, 3, 4]}
    mock_post.return_value = resp

    engine = LlamaCppEngine(EngineConfig())
    tokens = engine.tokenize("Hello world")

    assert tokens == [1, 2, 3, 4]
    url = mock_post.call_args.args[0]
    assert url == "http://localhost:8080/tokenize"
    payload = mock_post.call_args.kwargs["json"]
    assert payload["content"] == "Hello world"
    assert "add_special" not in payload
    assert "parse_special" not in payload


@patch("runtimes_validator.engines.llamacpp.requests.post")
def test_tokenize_with_special_flags(mock_post: MagicMock):
    resp = MagicMock()
    resp.json.return_value = {"tokens": [99]}
    mock_post.return_value = resp

    engine = LlamaCppEngine(EngineConfig())
    tokens = engine.tokenize("<|end_of_text|>", add_special=False, parse_special=True)

    assert tokens == [99]
    payload = mock_post.call_args.kwargs["json"]
    assert payload["add_special"] is False
    assert payload["parse_special"] is True


@patch("runtimes_validator.engines.llamacpp.requests.post")
def test_tokenize_sends_headers(mock_post: MagicMock):
    resp = MagicMock()
    resp.json.return_value = {"tokens": [1]}
    mock_post.return_value = resp

    headers = {"X-Key": "val"}
    engine = LlamaCppEngine(EngineConfig(extra={"headers": headers}))
    engine.tokenize("hi")

    assert mock_post.call_args.kwargs["headers"] == headers


# --- LlamaCppEngine.detokenize() ---


@patch("runtimes_validator.engines.llamacpp.requests.post")
def test_detokenize_basic(mock_post: MagicMock):
    resp = MagicMock()
    resp.json.return_value = {"content": "Hello world"}
    mock_post.return_value = resp

    engine = LlamaCppEngine(EngineConfig())
    text = engine.detokenize([1, 2, 3, 4])

    assert text == "Hello world"
    url = mock_post.call_args.args[0]
    assert url == "http://localhost:8080/detokenize"
    payload = mock_post.call_args.kwargs["json"]
    assert payload["tokens"] == [1, 2, 3, 4]


# --- LlamaCppEngine.props() ---


@patch("runtimes_validator.engines.llamacpp.requests.get")
def test_props_basic(mock_get: MagicMock):
    resp = MagicMock()
    resp.json.return_value = {
        "chat_template": "...",
        "default_generation_settings": {"temperature": 0.8},
    }
    mock_get.return_value = resp

    engine = LlamaCppEngine(EngineConfig(base_url="http://myhost:8080"))
    body = engine.props()

    assert "chat_template" in body
    assert "default_generation_settings" in body
    url = mock_get.call_args.args[0]
    assert url == "http://myhost:8080/props"


@patch("runtimes_validator.engines.llamacpp.requests.get")
def test_props_sends_headers(mock_get: MagicMock):
    resp = MagicMock()
    resp.json.return_value = {}
    mock_get.return_value = resp

    headers = {"Authorization": "Bearer tok"}
    engine = LlamaCppEngine(EngineConfig(extra={"headers": headers}))
    engine.props()

    assert mock_get.call_args.kwargs["headers"] == headers


# --- Timeout handling ---


@patch("runtimes_validator.engines.openai_compat.requests.post")
def test_chat_raises_engine_timeout_error_on_timeout(mock_post: MagicMock):
    mock_post.side_effect = requests.Timeout("Read timed out")
    engine = VllmEngine(EngineConfig(model_id="m"))

    with pytest.raises(EngineTimeoutError, match="timed out after 120s"):
        engine.chat([{"role": "user", "content": "hi"}])


@patch("runtimes_validator.engines.openai_compat.requests.post")
def test_chat_sets_last_timeout_flag_on_timeout(mock_post: MagicMock):
    mock_post.side_effect = requests.Timeout("Read timed out")
    engine = VllmEngine(EngineConfig(model_id="m"))
    assert engine._last_timeout is False

    with pytest.raises(EngineTimeoutError):
        engine.chat([{"role": "user", "content": "hi"}])

    assert engine._last_timeout is True


@patch("runtimes_validator.engines.openai_compat.requests.post")
def test_chat_resets_last_timeout_on_success(mock_post: MagicMock):
    engine = VllmEngine(EngineConfig(model_id="m"))
    engine._last_timeout = True

    mock_post.return_value = _fake_chat_response()
    engine.chat([{"role": "user", "content": "hi"}])

    assert engine._last_timeout is False


@patch("runtimes_validator.engines.openai_compat.requests.post")
def test_chat_timeout_observation_is_sticky_until_reset(mock_post: MagicMock):
    engine = VllmEngine(EngineConfig(model_id="m"))
    mock_post.side_effect = [
        requests.Timeout("Read timed out"),
        _fake_chat_response(),
    ]

    with pytest.raises(EngineTimeoutError):
        engine.chat([{"role": "user", "content": "first"}])
    engine.chat([{"role": "user", "content": "second"}])

    assert engine._last_timeout is False
    assert engine.timed_out_since_last_check() is True

    engine.reset_timeout_observed()
    assert engine.timed_out_since_last_check() is False


@patch("runtimes_validator.engines.openai_compat.requests.post")
def test_chat_stream_raises_engine_timeout_error_on_request_timeout(mock_post: MagicMock):
    mock_post.side_effect = requests.Timeout("Read timed out")
    engine = VllmEngine(EngineConfig(model_id="m"))

    with pytest.raises(EngineTimeoutError, match="timed out after 120s"):
        list(engine.chat_stream([{"role": "user", "content": "hi"}]))

    assert engine.timed_out_since_last_check() is True


@patch("runtimes_validator.engines.openai_compat.requests.post")
def test_chat_stream_raises_engine_timeout_error_on_iteration_timeout(mock_post: MagicMock):
    resp = MagicMock()
    resp.iter_lines.side_effect = requests.Timeout("Read timed out")
    mock_post.return_value = resp
    engine = VllmEngine(EngineConfig(model_id="m"))

    with pytest.raises(EngineTimeoutError, match="timed out after 120s"):
        list(engine.chat_stream([{"role": "user", "content": "hi"}]))

    assert engine.timed_out_since_last_check() is True


# --- inspection logger integration ---


@patch("runtimes_validator.engines.openai_compat.requests.post")
def test_chat_forwards_exchange_to_inspection_logger(mock_post: MagicMock):
    from runtimes_validator.reporting.inspection import InspectionLogger

    mock_post.return_value = _fake_chat_response()
    logger = MagicMock(spec=InspectionLogger)
    engine = VllmEngine(EngineConfig(model_id="m", extra={"inspection_logger": logger}))

    engine.chat([{"role": "user", "content": "hi"}])

    assert logger.log_exchange.call_count == 1
    args, kwargs = logger.log_exchange.call_args
    payload, response = args
    assert payload["messages"] == [{"role": "user", "content": "hi"}]
    assert payload["model"] == "m"
    assert response["choices"][0]["message"]["content"] == "Hello!"
    assert kwargs == {"streaming": False, "path": "/v1/chat/completions"}


@patch("runtimes_validator.engines.openai_compat.requests.post")
def test_chat_stream_forwards_payload_and_accumulated_chunks(mock_post: MagicMock):
    from runtimes_validator.reporting.inspection import InspectionLogger

    sse_lines = [
        'data: {"choices":[{"delta":{"content":"Hi"}}]}',
        'data: {"choices":[{"delta":{"content":"!"},"finish_reason":"stop"}]}',
        "data: [DONE]",
    ]
    mock_post.return_value = _fake_stream_response(sse_lines)
    logger = MagicMock(spec=InspectionLogger)
    engine = VllmEngine(EngineConfig(model_id="m", extra={"inspection_logger": logger}))

    chunks = list(engine.chat_stream([{"role": "user", "content": "hi"}]))

    assert logger.log_exchange.call_count == 1
    args, kwargs = logger.log_exchange.call_args
    payload, response = args
    assert payload["stream"] is True
    assert response == chunks
    assert kwargs == {"streaming": True, "path": "/v1/chat/completions"}


@patch("runtimes_validator.engines.openai_compat.requests.post")
def test_chat_without_inspection_logger_does_not_fail(mock_post: MagicMock):
    mock_post.return_value = _fake_chat_response()
    engine = VllmEngine(EngineConfig(model_id="m"))

    result = engine.chat([{"role": "user", "content": "hi"}])

    assert result["content"] == "Hello!"
    assert engine._inspection is None
