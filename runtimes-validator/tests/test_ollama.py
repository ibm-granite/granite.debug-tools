from __future__ import annotations

import io
import subprocess
from unittest.mock import MagicMock, patch

import pytest
import requests

from runtimes_validator.engines.base import EngineConfig
from runtimes_validator.engines.ollama import OllamaEngine


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
    resp.json.return_value = {"version": "0.6.2"}
    return resp


def _fake_pull_ok() -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status = MagicMock()
    return resp


def _fake_json_response(body: dict) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = body
    return resp


def _fake_stream_response(lines: list[str]) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.iter_lines.return_value = iter(lines)
    return resp


def _make_mock_process(*, poll_returns=None, pid=12345) -> MagicMock:
    proc = MagicMock()
    proc.pid = pid
    proc.poll.return_value = poll_returns
    proc.wait.return_value = 0
    proc.returncode = poll_returns
    return proc


def _make_stderr_file(content: bytes = b"") -> io.BytesIO:
    """Create a seekable in-memory stderr file pre-loaded with content."""
    buf = io.BytesIO(content)
    buf.seek(0, 2)  # move to end, as Popen would write at end
    return buf


# --- Fixtures ---


@pytest.fixture
def managed_start_env():
    """Common mock environment for managed start() tests.

    Patches: shutil.which, subprocess.Popen, health_check (via openai_compat),
    time.sleep, ollama-module requests.get, and ollama-module requests.post.

    Returns a dict with keys: popen, process, health_get, ollama_get, post, stderr_file.
    """
    with (
        patch("runtimes_validator.engines.ollama.shutil.which", return_value="/usr/bin/ollama"),
        patch("runtimes_validator.engines.ollama.subprocess.Popen") as mock_popen,
        patch("runtimes_validator.engines.openai_compat.requests.get") as mock_compat_get,
        patch("runtimes_validator.engines.ollama.time.sleep"),
        patch("runtimes_validator.engines.ollama.requests.get") as mock_ollama_get,
        patch("runtimes_validator.engines.ollama.requests.post") as mock_post,
        patch("runtimes_validator.engines.ollama.tempfile.TemporaryFile") as mock_tmp,
    ):
        proc = _make_mock_process()
        mock_popen.return_value = proc
        mock_tmp.return_value = _make_stderr_file()

        # Default: health OK, version OK
        mock_compat_get.return_value = _fake_health_ok()

        version_resp = MagicMock()
        version_resp.status_code = 200
        version_resp.json.return_value = {"version": "0.6.2"}
        mock_ollama_get.return_value = version_resp

        mock_post.return_value = _fake_pull_ok()

        yield {
            "popen": mock_popen,
            "process": proc,
            "health_get": mock_compat_get,
            "ollama_get": mock_ollama_get,
            "post": mock_post,
            "stderr_file": mock_tmp,
        }


# --- Default port / base_url ---


def test_ollama_default_port():
    engine = OllamaEngine(EngineConfig())
    assert engine._base_url == "http://localhost:11434"


def test_ollama_custom_base_url():
    engine = OllamaEngine(EngineConfig(base_url="http://remote-gpu:11434"))
    assert engine._base_url == "http://remote-gpu:11434"


# --- engine_id / get_info ---


def test_ollama_engine_id():
    assert OllamaEngine(EngineConfig()).engine_id() == "ollama"


def test_ollama_get_info():
    config = EngineConfig(mode="external", base_url="http://myhost:11434")
    info = OllamaEngine(config).get_info()
    assert info.engine_id == "ollama"
    assert info.mode == "external"
    assert info.base_url == "http://myhost:11434"
    assert info.version == "unknown"


# --- health_check() ---


@patch("runtimes_validator.engines.openai_compat.requests.get")
def test_ollama_health_check_hits_api_version(mock_get: MagicMock):
    mock_get.return_value = _fake_health_ok()
    engine = OllamaEngine(EngineConfig(base_url="http://myhost:11434"))

    engine.health_check()

    url = mock_get.call_args.args[0]
    assert url == "http://myhost:11434/api/version"


@patch("runtimes_validator.engines.openai_compat.requests.get")
def test_ollama_health_check_returns_true_on_200(mock_get: MagicMock):
    mock_get.return_value = _fake_health_ok()
    engine = OllamaEngine(EngineConfig())
    assert engine.health_check() is True


@patch("runtimes_validator.engines.openai_compat.requests.get")
def test_ollama_health_check_returns_false_on_connection_error(mock_get: MagicMock):
    mock_get.side_effect = requests.ConnectionError("refused")
    engine = OllamaEngine(EngineConfig())
    assert engine.health_check() is False


# --- chat() ---


@patch("runtimes_validator.engines.openai_compat.requests.post")
def test_ollama_chat_posts_to_correct_url(mock_post: MagicMock):
    mock_post.return_value = _fake_chat_response()
    engine = OllamaEngine(EngineConfig(base_url="http://gpu:11434", model_id="granite3.3:8b"))

    engine.chat([{"role": "user", "content": "hi"}])

    url = mock_post.call_args.args[0]
    assert url == "http://gpu:11434/v1/chat/completions"


@patch("runtimes_validator.engines.openai_compat.requests.post")
def test_ollama_chat_includes_model(mock_post: MagicMock):
    mock_post.return_value = _fake_chat_response()
    engine = OllamaEngine(EngineConfig(model_id="granite3.3:8b"))

    engine.chat([{"role": "user", "content": "hi"}])

    payload = mock_post.call_args.kwargs["json"]
    assert payload["model"] == "granite3.3:8b"


# --- start() tests ---


def test_start_rejects_external_mode():
    engine = OllamaEngine(EngineConfig(mode="external"))
    with pytest.raises(ValueError, match="mode is 'external'"):
        engine.start("granite3.3:8b")


@patch("runtimes_validator.engines.ollama.shutil.which", return_value=None)
def test_start_binary_not_found(mock_which: MagicMock):
    engine = OllamaEngine(EngineConfig(mode="managed"))
    with pytest.raises(RuntimeError, match="'ollama' binary not found"):
        engine.start("granite3.3:8b")


@patch("runtimes_validator.engines.ollama.tempfile.TemporaryFile")
@patch("runtimes_validator.engines.ollama.shutil.which", return_value=None)
def test_start_uses_custom_binary_path(mock_which: MagicMock, mock_tmp: MagicMock):
    """When ollama_bin is set in extra, it is used directly without calling shutil.which."""
    config = EngineConfig(
        mode="managed",
        extra={"ollama_bin": "/custom/ollama", "startup_timeout": 0.1},
    )
    engine = OllamaEngine(config)

    with patch("runtimes_validator.engines.ollama.subprocess.Popen") as mock_popen:
        proc = _make_mock_process(poll_returns=1)
        mock_popen.return_value = proc
        mock_tmp.return_value = _make_stderr_file(b"bind error")

        with pytest.raises(RuntimeError):
            engine.start("model")

        cmd = mock_popen.call_args.args[0]
        assert cmd[0] == "/custom/ollama"


@patch("runtimes_validator.engines.ollama.tempfile.TemporaryFile")
@patch("runtimes_validator.engines.ollama.time.sleep")
@patch("runtimes_validator.engines.ollama.requests.post")
@patch("runtimes_validator.engines.ollama.requests.get")
@patch("runtimes_validator.engines.ollama.subprocess.Popen")
@patch("runtimes_validator.engines.ollama.shutil.which", return_value="/usr/bin/ollama")
def test_start_process_exits_immediately(
    mock_which, mock_popen, mock_get, mock_post, mock_sleep, mock_tmp
):
    proc = _make_mock_process(poll_returns=1)
    mock_popen.return_value = proc
    mock_tmp.return_value = _make_stderr_file(b"address already in use")

    engine = OllamaEngine(EngineConfig(mode="managed"))
    with pytest.raises(RuntimeError, match="exited during startup with code 1") as exc_info:
        engine.start("model")
    assert "address already in use" in str(exc_info.value)


@patch("runtimes_validator.engines.ollama.tempfile.TemporaryFile")
@patch("runtimes_validator.engines.ollama.time.sleep")
@patch("runtimes_validator.engines.ollama.time.monotonic")
@patch("runtimes_validator.engines.openai_compat.requests.get")
@patch("runtimes_validator.engines.ollama.subprocess.Popen")
@patch("runtimes_validator.engines.ollama.shutil.which", return_value="/usr/bin/ollama")
def test_start_health_timeout(
    mock_which, mock_popen, mock_get, mock_monotonic, mock_sleep, mock_tmp
):
    proc = _make_mock_process()
    mock_popen.return_value = proc
    mock_tmp.return_value = _make_stderr_file()

    # Simulate time passing beyond the timeout.
    # Use a default so extra monotonic() calls don't raise StopIteration.
    _times = iter([0.0, 0.5, 1.0])
    mock_monotonic.side_effect = lambda: next(_times, 31.0)
    mock_get.side_effect = requests.ConnectionError("refused")

    config = EngineConfig(mode="managed", extra={"startup_timeout": 30})
    engine = OllamaEngine(config)

    with pytest.raises(RuntimeError, match="did not become ready within 30s"):
        engine.start("model")

    # Verify stop was called for cleanup
    proc.terminate.assert_called_once()


def test_start_successful(managed_start_env):
    env = managed_start_env
    engine = OllamaEngine(EngineConfig(mode="managed"))
    engine.start("granite3.3:8b")

    # Verify process was started with correct command
    cmd = env["popen"].call_args.args[0]
    assert cmd == ["/usr/bin/ollama", "serve"]

    # Verify OLLAMA_HOST was set
    popen_env = env["popen"].call_args.kwargs["env"]
    assert popen_env["OLLAMA_HOST"] == "localhost:11434"

    # Verify version was fetched
    assert engine._ollama_version == "0.6.2"
    assert engine.get_info().version == "0.6.2"

    # Verify model pull was called
    pull_url = env["post"].call_args.args[0]
    assert pull_url == "http://localhost:11434/api/pull"
    pull_payload = env["post"].call_args.kwargs["json"]
    assert pull_payload == {"name": "granite3.3:8b", "stream": False}


def test_start_with_server_args(managed_start_env):
    env = managed_start_env
    config = EngineConfig(
        mode="managed",
        extra={"server_args": ["--verbose", "--debug"]},
    )
    engine = OllamaEngine(config)
    engine.start("granite3.3:8b")

    cmd = env["popen"].call_args.args[0]
    assert cmd == ["/usr/bin/ollama", "serve", "--verbose", "--debug"]


def test_start_rejects_double_start(managed_start_env):
    engine = OllamaEngine(EngineConfig(mode="managed"))
    engine.start("granite3.3:8b")

    with pytest.raises(RuntimeError, match="already running"):
        engine.start("granite3.3:8b")


def test_start_model_pull_failure_cleans_up(managed_start_env):
    env = managed_start_env

    # Model pull fails
    env["post"].side_effect = requests.HTTPError("404 model not found")

    engine = OllamaEngine(EngineConfig(mode="managed"))
    with pytest.raises(RuntimeError, match="Failed to pull model"):
        engine.start("nonexistent-model")

    # Process should be cleaned up
    env["process"].terminate.assert_called_once()


# --- stop() tests ---


def test_stop_noop_when_no_process():
    engine = OllamaEngine(EngineConfig())
    engine.stop()  # Should not raise


def test_stop_graceful():
    engine = OllamaEngine(EngineConfig())
    proc = _make_mock_process()
    engine._process = proc

    engine.stop()

    proc.terminate.assert_called_once()
    proc.wait.assert_called_once()
    proc.kill.assert_not_called()
    assert engine._process is None


def test_stop_force_kill():
    engine = OllamaEngine(EngineConfig())
    proc = _make_mock_process()
    proc.wait.side_effect = [subprocess.TimeoutExpired("ollama", 10), 0]
    engine._process = proc

    engine.stop()

    proc.terminate.assert_called_once()
    proc.kill.assert_called_once()
    assert engine._process is None


def test_stop_closes_stderr_file():
    engine = OllamaEngine(EngineConfig())
    proc = _make_mock_process()
    stderr = _make_stderr_file(b"some output")
    engine._process = proc
    engine._stderr_file = stderr

    engine.stop()

    assert engine._stderr_file is None
    assert stderr.closed


def test_stop_uses_custom_timeout():
    engine = OllamaEngine(EngineConfig(extra={"stop_timeout": 42}))
    proc = _make_mock_process()
    engine._process = proc

    engine.stop()

    proc.wait.assert_called_once_with(timeout=42)


def test_stop_force_kill_wait_timeout():
    """After SIGKILL, if wait() also times out, TimeoutExpired propagates."""
    engine = OllamaEngine(EngineConfig())
    proc = _make_mock_process()
    proc.wait.side_effect = subprocess.TimeoutExpired("ollama", 10)
    engine._process = proc

    with pytest.raises(subprocess.TimeoutExpired):
        engine.stop()

    proc.terminate.assert_called_once()
    proc.kill.assert_called_once()
    # Process and stderr are still cleaned up via finally
    assert engine._process is None


# --- get_info with version ---


def test_get_info_returns_version_after_start():
    engine = OllamaEngine(EngineConfig(mode="managed"))
    engine._ollama_version = "0.6.2"

    info = engine.get_info()
    assert info.version == "0.6.2"
    assert info.mode == "managed"


# --- _fetch_version failure ---


@patch("runtimes_validator.engines.ollama.requests.get")
def test_fetch_version_handles_failure(mock_get: MagicMock):
    mock_get.side_effect = requests.ConnectionError("refused")
    engine = OllamaEngine(EngineConfig())

    engine._fetch_version()

    assert engine._ollama_version is None


@patch("runtimes_validator.engines.ollama.requests.get")
def test_fetch_version_handles_non_200(mock_get: MagicMock):
    resp = MagicMock()
    resp.status_code = 503
    mock_get.return_value = resp
    engine = OllamaEngine(EngineConfig())

    engine._fetch_version()

    assert engine._ollama_version is None


# --- OLLAMA_HOST derivation ---


def test_ollama_host_from_default_url():
    engine = OllamaEngine(EngineConfig())
    env = engine._build_env()
    assert env["OLLAMA_HOST"] == "localhost:11434"


def test_ollama_host_from_custom_url():
    engine = OllamaEngine(EngineConfig(base_url="http://192.168.1.100:9999"))
    env = engine._build_env()
    assert env["OLLAMA_HOST"] == "192.168.1.100:9999"


def test_ollama_host_ipv6():
    engine = OllamaEngine(EngineConfig(base_url="http://[::1]:11434"))
    env = engine._build_env()
    assert env["OLLAMA_HOST"] == "[::1]:11434"


# --- _read_stderr edge cases ---


def test_read_stderr_returns_empty_when_file_is_none():
    engine = OllamaEngine(EngineConfig())
    engine._stderr_file = None
    assert engine._read_stderr() == ""


def test_read_stderr_truncates_to_last_4096_bytes():
    engine = OllamaEngine(EngineConfig())
    content = b"X" * 5000 + b"TAIL_MARKER"
    engine._stderr_file = _make_stderr_file(content)

    result = engine._read_stderr()

    assert "TAIL_MARKER" in result
    assert len(result) <= 4096


# --- _ensure_model HTTP error ---


@patch("runtimes_validator.engines.ollama.requests.post")
def test_ensure_model_http_500(mock_post: MagicMock):
    resp = MagicMock()
    resp.status_code = 500
    resp.raise_for_status.side_effect = requests.HTTPError("500 Server Error")
    mock_post.return_value = resp

    engine = OllamaEngine(EngineConfig())
    with pytest.raises(RuntimeError, match="Failed to pull model 'bad-model'"):
        engine._ensure_model("bad-model")


# --- start() with custom base_url ---


def test_start_custom_base_url(managed_start_env):
    env = managed_start_env
    engine = OllamaEngine(EngineConfig(mode="managed", base_url="http://192.168.1.50:9999"))
    engine.start("granite3.3:8b")

    popen_env = env["popen"].call_args.kwargs["env"]
    assert popen_env["OLLAMA_HOST"] == "192.168.1.50:9999"

    pull_url = env["post"].call_args.args[0]
    assert pull_url == "http://192.168.1.50:9999/api/pull"


# --- _resolve_model() ---


def test_resolve_model_raises_when_no_model():
    engine = OllamaEngine(EngineConfig())
    with pytest.raises(ValueError, match="No model specified"):
        engine._resolve_model(None)


def test_resolve_model_uses_explicit_over_config():
    engine = OllamaEngine(EngineConfig(model_id="config-model"))
    assert engine._resolve_model("explicit") == "explicit"


def test_resolve_model_falls_back_to_config():
    engine = OllamaEngine(EngineConfig(model_id="config-model"))
    assert engine._resolve_model(None) == "config-model"


# --- generate() ---


@patch("runtimes_validator.engines.openai_compat.requests.post")
def test_generate_basic(mock_post: MagicMock):
    mock_post.return_value = _fake_json_response(
        {
            "response": "Paris",
            "done": True,
        }
    )
    engine = OllamaEngine(EngineConfig(model_id="granite3.3:8b"))

    body = engine.generate("Capital of France?")

    assert body["response"] == "Paris"
    url = mock_post.call_args.args[0]
    assert url == "http://localhost:11434/api/generate"
    payload = mock_post.call_args.kwargs["json"]
    assert payload["model"] == "granite3.3:8b"
    assert payload["prompt"] == "Capital of France?"
    assert payload["stream"] is False


@patch("runtimes_validator.engines.openai_compat.requests.post")
def test_generate_with_all_options(mock_post: MagicMock):
    mock_post.return_value = _fake_json_response({"response": "ok"})
    engine = OllamaEngine(EngineConfig(model_id="m"))

    engine.generate(
        "hi",
        model="override",
        system="Be helpful.",
        context=[1, 2, 3],
        options={"temperature": 0, "seed": 42},
        keep_alive=0,
        timeout=300,
    )

    payload = mock_post.call_args.kwargs["json"]
    assert payload["model"] == "override"
    assert payload["system"] == "Be helpful."
    assert payload["context"] == [1, 2, 3]
    assert payload["options"] == {"temperature": 0, "seed": 42}
    assert payload["keep_alive"] == 0
    assert mock_post.call_args.kwargs["timeout"] == 300


@patch("runtimes_validator.engines.openai_compat.requests.post")
def test_generate_sends_headers(mock_post: MagicMock):
    mock_post.return_value = _fake_json_response({"response": ""})
    headers = {"X-Key": "val"}
    engine = OllamaEngine(EngineConfig(model_id="m", extra={"headers": headers}))

    engine.generate("hi")

    assert mock_post.call_args.kwargs["headers"] == headers


# --- generate_stream() ---


@patch("runtimes_validator.engines.openai_compat.requests.post")
def test_generate_stream_yields_chunks(mock_post: MagicMock):
    ndjson_lines = [
        '{"response":"Hello","done":false}',
        '{"response":" world","done":false}',
        '{"response":"","done":true}',
    ]
    mock_post.return_value = _fake_stream_response(ndjson_lines)
    engine = OllamaEngine(EngineConfig(model_id="m"))

    chunks = list(engine.generate_stream("hi"))

    assert len(chunks) == 3
    assert chunks[0]["response"] == "Hello"
    assert chunks[1]["response"] == " world"
    assert chunks[2]["done"] is True


@patch("runtimes_validator.engines.openai_compat.requests.post")
def test_generate_stream_sends_correct_payload(mock_post: MagicMock):
    mock_post.return_value = _fake_stream_response([])
    engine = OllamaEngine(EngineConfig(model_id="m"))

    list(
        engine.generate_stream(
            "hi",
            model="override",
            options={"num_predict": 50},
            timeout=200,
        )
    )

    payload = mock_post.call_args.kwargs["json"]
    assert payload["stream"] is True
    assert payload["model"] == "override"
    assert payload["options"] == {"num_predict": 50}
    assert mock_post.call_args.kwargs["stream"] is True
    assert mock_post.call_args.kwargs["timeout"] == 200


@patch("runtimes_validator.engines.openai_compat.requests.post")
def test_generate_stream_skips_empty_lines(mock_post: MagicMock):
    ndjson_lines = [
        "",
        '{"response":"ok","done":true}',
        "",
    ]
    mock_post.return_value = _fake_stream_response(ndjson_lines)
    engine = OllamaEngine(EngineConfig(model_id="m"))

    chunks = list(engine.generate_stream("hi"))

    assert len(chunks) == 1


# --- native_chat() ---


@patch("runtimes_validator.engines.openai_compat.requests.post")
def test_native_chat_basic(mock_post: MagicMock):
    mock_post.return_value = _fake_json_response(
        {
            "message": {"role": "assistant", "content": "Hi!"},
            "done": True,
            "done_reason": "stop",
        }
    )
    engine = OllamaEngine(EngineConfig(model_id="granite3.3:8b"))
    messages = [{"role": "user", "content": "Hello"}]

    body = engine.native_chat(messages)

    assert body["message"]["content"] == "Hi!"
    url = mock_post.call_args.args[0]
    assert url == "http://localhost:11434/api/chat"
    payload = mock_post.call_args.kwargs["json"]
    assert payload["model"] == "granite3.3:8b"
    assert payload["messages"] == messages
    assert payload["stream"] is False


@patch("runtimes_validator.engines.openai_compat.requests.post")
def test_native_chat_with_tools_and_options(mock_post: MagicMock):
    mock_post.return_value = _fake_json_response({"message": {}, "done": True})
    engine = OllamaEngine(EngineConfig(model_id="m"))
    tools = [{"type": "function", "function": {"name": "get_weather"}}]

    engine.native_chat(
        [{"role": "user", "content": "weather?"}],
        tools=tools,
        options={"temperature": 0},
        timeout=60,
    )

    payload = mock_post.call_args.kwargs["json"]
    assert payload["tools"] == tools
    assert payload["options"] == {"temperature": 0}
    assert mock_post.call_args.kwargs["timeout"] == 60


# --- native_chat_stream() ---


@patch("runtimes_validator.engines.openai_compat.requests.post")
def test_native_chat_stream_yields_chunks(mock_post: MagicMock):
    ndjson_lines = [
        '{"message":{"content":"Hi"},"done":false}',
        '{"message":{"content":"!"},"done":true}',
    ]
    mock_post.return_value = _fake_stream_response(ndjson_lines)
    engine = OllamaEngine(EngineConfig(model_id="m"))

    chunks = list(
        engine.native_chat_stream(
            [{"role": "user", "content": "hello"}],
        )
    )

    assert len(chunks) == 2
    assert chunks[0]["message"]["content"] == "Hi"
    assert chunks[1]["done"] is True


@patch("runtimes_validator.engines.openai_compat.requests.post")
def test_native_chat_stream_sends_correct_payload(mock_post: MagicMock):
    mock_post.return_value = _fake_stream_response([])
    engine = OllamaEngine(EngineConfig(model_id="m"))
    tools = [{"type": "function", "function": {"name": "f"}}]

    list(
        engine.native_chat_stream(
            [{"role": "user", "content": "hi"}],
            model="override",
            tools=tools,
            options={"temperature": 0},
        )
    )

    payload = mock_post.call_args.kwargs["json"]
    assert payload["stream"] is True
    assert payload["model"] == "override"
    assert payload["tools"] == tools
    assert payload["options"] == {"temperature": 0}


# --- inspection logger integration for native endpoints ---


@patch("runtimes_validator.engines.openai_compat.requests.post")
def test_generate_forwards_exchange_to_inspection_logger(
    mock_post: MagicMock,
):
    from runtimes_validator.reporting.inspection import InspectionLogger

    mock_post.return_value = _fake_json_response({"response": "Paris", "done": True})
    logger = MagicMock(spec=InspectionLogger)
    engine = OllamaEngine(EngineConfig(model_id="m", extra={"inspection_logger": logger}))

    engine.generate("hi")

    assert logger.log_exchange.call_count == 1
    args, kwargs = logger.log_exchange.call_args
    payload, response = args
    assert payload["prompt"] == "hi"
    assert payload["stream"] is False
    assert response == {"response": "Paris", "done": True}
    assert kwargs == {"streaming": False, "path": "/api/generate"}


@patch("runtimes_validator.engines.openai_compat.requests.post")
def test_generate_stream_forwards_payload_and_accumulated_chunks_to_logger(
    mock_post: MagicMock,
):
    from runtimes_validator.reporting.inspection import InspectionLogger

    ndjson_lines = [
        '{"response":"Hi","done":false}',
        '{"response":"!","done":true}',
    ]
    mock_post.return_value = _fake_stream_response(ndjson_lines)
    logger = MagicMock(spec=InspectionLogger)
    engine = OllamaEngine(EngineConfig(model_id="m", extra={"inspection_logger": logger}))

    chunks = list(engine.generate_stream("hi"))

    assert logger.log_exchange.call_count == 1
    args, kwargs = logger.log_exchange.call_args
    payload, response = args
    assert payload["stream"] is True
    assert response == chunks
    assert kwargs == {"streaming": True, "path": "/api/generate"}


@patch("runtimes_validator.engines.openai_compat.requests.post")
def test_native_chat_forwards_exchange_to_inspection_logger(
    mock_post: MagicMock,
):
    from runtimes_validator.reporting.inspection import InspectionLogger

    mock_post.return_value = _fake_json_response(
        {
            "message": {"role": "assistant", "content": "Hi!"},
            "done": True,
        }
    )
    logger = MagicMock(spec=InspectionLogger)
    engine = OllamaEngine(EngineConfig(model_id="m", extra={"inspection_logger": logger}))

    engine.native_chat([{"role": "user", "content": "Hello"}])

    assert logger.log_exchange.call_count == 1
    args, kwargs = logger.log_exchange.call_args
    _, response = args
    assert response["message"]["content"] == "Hi!"
    assert kwargs == {"streaming": False, "path": "/api/chat"}


@patch("runtimes_validator.engines.openai_compat.requests.post")
def test_native_chat_stream_forwards_payload_and_accumulated_chunks_to_logger(
    mock_post: MagicMock,
):
    from runtimes_validator.reporting.inspection import InspectionLogger

    ndjson_lines = [
        '{"message":{"content":"Hi"},"done":false}',
        '{"message":{"content":"!"},"done":true}',
    ]
    mock_post.return_value = _fake_stream_response(ndjson_lines)
    logger = MagicMock(spec=InspectionLogger)
    engine = OllamaEngine(EngineConfig(model_id="m", extra={"inspection_logger": logger}))

    chunks = list(engine.native_chat_stream([{"role": "user", "content": "hi"}]))

    assert logger.log_exchange.call_count == 1
    args, kwargs = logger.log_exchange.call_args
    _, response = args
    assert response == chunks
    assert kwargs == {"streaming": True, "path": "/api/chat"}


@patch("runtimes_validator.engines.openai_compat.requests.post")
def test_generate_stream_logs_accumulated_chunks_on_mid_stream_timeout(
    mock_post: MagicMock,
):
    """A Timeout mid-iteration still logs the exchange with accumulated chunks."""
    from runtimes_validator.domain.models import EngineTimeoutError
    from runtimes_validator.reporting.inspection import InspectionLogger

    resp = MagicMock()
    resp.status_code = 200

    def _raise_timeout_after_two(*, decode_unicode: bool):
        yield '{"response":"a","done":false}'
        yield '{"response":"b","done":false}'
        raise requests.Timeout("Read timed out")

    resp.iter_lines.side_effect = _raise_timeout_after_two
    mock_post.return_value = resp

    logger = MagicMock(spec=InspectionLogger)
    engine = OllamaEngine(EngineConfig(model_id="m", extra={"inspection_logger": logger}))

    with pytest.raises(EngineTimeoutError):
        list(engine.generate_stream("hi"))

    assert engine.timed_out_since_last_check() is True
    assert logger.log_exchange.call_count == 1
    args, kwargs = logger.log_exchange.call_args
    _, response = args
    assert kwargs == {"streaming": True, "path": "/api/generate"}
    assert [c["response"] for c in response] == ["a", "b"]


@patch("runtimes_validator.engines.openai_compat.requests.post")
def test_generate_logs_null_response_on_initial_post_timeout(
    mock_post: MagicMock,
):
    """An initial-POST timeout logs one exchange with response=None."""
    from runtimes_validator.domain.models import EngineTimeoutError
    from runtimes_validator.reporting.inspection import InspectionLogger

    mock_post.side_effect = requests.Timeout("Connection timed out")
    logger = MagicMock(spec=InspectionLogger)
    engine = OllamaEngine(EngineConfig(model_id="m", extra={"inspection_logger": logger}))

    with pytest.raises(EngineTimeoutError):
        engine.generate("hi")

    assert logger.log_exchange.call_count == 1
    args, kwargs = logger.log_exchange.call_args
    payload, response = args
    assert payload["prompt"] == "hi"
    assert response is None
    assert kwargs == {"streaming": False, "path": "/api/generate"}


# --- show() ---


@patch("runtimes_validator.engines.ollama.requests.post")
def test_show_basic(mock_post: MagicMock):
    mock_post.return_value = _fake_json_response(
        {
            "modelfile": "FROM granite3.3:8b",
            "details": {"family": "granite", "parameter_size": "8B"},
            "template": "...",
            "model_info": {"key": "value"},
        }
    )
    engine = OllamaEngine(EngineConfig(model_id="granite3.3:8b"))

    body = engine.show()

    assert body["modelfile"] == "FROM granite3.3:8b"
    assert body["details"]["family"] == "granite"
    url = mock_post.call_args.args[0]
    assert url == "http://localhost:11434/api/show"
    payload = mock_post.call_args.kwargs["json"]
    assert payload["model"] == "granite3.3:8b"


@patch("runtimes_validator.engines.ollama.requests.post")
def test_show_with_explicit_model(mock_post: MagicMock):
    mock_post.return_value = _fake_json_response({"modelfile": "..."})
    engine = OllamaEngine(EngineConfig(model_id="default"))

    engine.show(model="override")

    payload = mock_post.call_args.kwargs["json"]
    assert payload["model"] == "override"


@patch("runtimes_validator.engines.ollama.requests.post")
def test_show_sends_headers(mock_post: MagicMock):
    mock_post.return_value = _fake_json_response({})
    headers = {"Authorization": "Bearer tok"}
    engine = OllamaEngine(EngineConfig(model_id="m", extra={"headers": headers}))

    engine.show()

    assert mock_post.call_args.kwargs["headers"] == headers
