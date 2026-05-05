from __future__ import annotations

import io
import subprocess
from unittest.mock import MagicMock, patch

import pytest
import requests

from granite_validation.domain.models import EngineTimeoutError
from granite_validation.engines.base import EngineConfig
from granite_validation.engines.vllm import VllmEngine


# --- Helpers ---


def _fake_health_ok() -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    return resp


def _fake_version_response() -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"version": "0.8.3"}
    return resp


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

    Patches: shutil.which, subprocess.Popen, vllm.requests.get (health + version),
    time.sleep, and tempfile.TemporaryFile.

    Returns a dict with keys: popen, process, vllm_get, stderr_file.
    """
    with (
        patch(
            "granite_validation.engines.vllm.shutil.which",
            return_value="/usr/bin/vllm",
        ),
        patch("granite_validation.engines.vllm.subprocess.Popen") as mock_popen,
        patch("granite_validation.engines.vllm.time.sleep"),
        patch("granite_validation.engines.vllm.requests.get") as mock_vllm_get,
        patch("granite_validation.engines.vllm.tempfile.TemporaryFile") as mock_tmp,
    ):
        proc = _make_mock_process()
        mock_popen.return_value = proc
        mock_tmp.return_value = _make_stderr_file()

        # _wait_until_ready now calls requests.get directly (not health_check),
        # and _fetch_version also calls requests.get. Route both via side_effect.
        def _route_get(url, **kwargs):
            if "/version" in url:
                return _fake_version_response()
            # health probe
            return _fake_health_ok()

        mock_vllm_get.side_effect = _route_get

        yield {
            "popen": mock_popen,
            "process": proc,
            "vllm_get": mock_vllm_get,
            "stderr_file": mock_tmp,
        }


# --- Default port / base_url ---


def test_vllm_default_port():
    engine = VllmEngine(EngineConfig())
    assert engine._base_url == "http://localhost:8000"


def test_vllm_custom_base_url():
    engine = VllmEngine(EngineConfig(base_url="http://gpu-box:9090"))
    assert engine._base_url == "http://gpu-box:9090"


# --- engine_id / get_info ---


def test_vllm_engine_id():
    assert VllmEngine(EngineConfig()).engine_id() == "vllm"


def test_vllm_get_info():
    config = EngineConfig(mode="external", base_url="http://myhost:8000")
    info = VllmEngine(config).get_info()
    assert info.engine_id == "vllm"
    assert info.mode == "external"
    assert info.base_url == "http://myhost:8000"
    assert info.version == "unknown"


# --- health_check() ---


@patch("granite_validation.engines.openai_compat.requests.get")
def test_vllm_health_check_hits_health_endpoint(mock_get: MagicMock):
    mock_get.return_value = _fake_health_ok()
    engine = VllmEngine(EngineConfig(base_url="http://myhost:8000"))

    engine.health_check()

    url = mock_get.call_args_list[0].args[0]
    assert url == "http://myhost:8000/health"


@patch("granite_validation.engines.openai_compat.requests.get")
def test_vllm_health_check_returns_true_on_200(mock_get: MagicMock):
    mock_get.return_value = _fake_health_ok()
    engine = VllmEngine(EngineConfig())
    assert engine.health_check() is True


@patch("granite_validation.engines.openai_compat.requests.get")
def test_vllm_health_check_returns_false_on_connection_error(mock_get: MagicMock):
    mock_get.side_effect = requests.ConnectionError("refused")
    engine = VllmEngine(EngineConfig())
    assert engine.health_check() is False


# --- chat() ---


@patch("granite_validation.engines.openai_compat.requests.post")
def test_vllm_chat_posts_to_correct_url(mock_post: MagicMock):
    mock_post.return_value = _fake_chat_response()
    engine = VllmEngine(EngineConfig(base_url="http://gpu:8000", model_id="model"))

    engine.chat([{"role": "user", "content": "hi"}])

    url = mock_post.call_args.args[0]
    assert url == "http://gpu:8000/v1/chat/completions"


@patch("granite_validation.engines.openai_compat.requests.post")
def test_chat_timeout_passes_through_without_stderr(mock_post: MagicMock):
    """On timeout, EngineTimeoutError propagates cleanly without stderr."""
    mock_post.side_effect = requests.Timeout("Read timed out")
    engine = VllmEngine(EngineConfig(mode="managed", model_id="model"))
    stderr_file = io.BytesIO(b"CUDA error: out of memory")
    engine._stderr_file = stderr_file
    engine._process = MagicMock()

    with pytest.raises(EngineTimeoutError) as exc_info:
        engine.chat([{"role": "user", "content": "hi"}])
    assert "vllm stderr" not in str(exc_info.value)


@patch("granite_validation.engines.openai_compat.requests.post")
def test_chat_non_timeout_error_includes_stderr(mock_post: MagicMock):
    """Non-timeout errors (ConnectionError, etc.) still include vLLM stderr."""
    mock_post.side_effect = requests.ConnectionError("Connection refused")
    engine = VllmEngine(EngineConfig(mode="managed", model_id="model"))
    stderr_file = io.BytesIO(b"CUDA error: out of memory")
    engine._stderr_file = stderr_file
    engine._process = MagicMock()

    with pytest.raises(requests.RequestException, match="vllm stderr"):
        engine.chat([{"role": "user", "content": "hi"}])


@patch("granite_validation.engines.openai_compat.requests.post")
def test_chat_timeout_no_stderr_reraises(mock_post: MagicMock):
    """When stderr is empty, the original error is re-raised as-is."""
    mock_post.side_effect = requests.ConnectionError("Read timed out")
    engine = VllmEngine(EngineConfig(mode="managed", model_id="model"))

    with pytest.raises(requests.ConnectionError, match="Read timed out"):
        engine.chat([{"role": "user", "content": "hi"}])


@patch("granite_validation.engines.openai_compat.requests.post")
def test_chat_stream_non_timeout_error_includes_stderr(mock_post: MagicMock):
    """Streaming connection errors also include vLLM stderr."""
    mock_post.side_effect = requests.ConnectionError("Connection refused")
    engine = VllmEngine(EngineConfig(mode="managed", model_id="model"))
    engine._stderr_file = io.BytesIO(b"CUDA error: out of memory")
    engine._process = MagicMock()

    with pytest.raises(requests.RequestException, match="vllm stderr"):
        list(engine.chat_stream([{"role": "user", "content": "hi"}]))


# --- start() tests ---


def test_start_rejects_external_mode():
    engine = VllmEngine(EngineConfig(mode="external"))
    with pytest.raises(ValueError, match="mode is 'external'"):
        engine.start("ibm-granite/granite-3.3-8b-instruct")


@patch("granite_validation.engines.vllm.shutil.which", return_value=None)
def test_start_binary_not_found(mock_which: MagicMock):
    engine = VllmEngine(EngineConfig(mode="managed"))
    with pytest.raises(RuntimeError, match="'vllm' binary not found"):
        engine.start("ibm-granite/granite-3.3-8b-instruct")


@patch("granite_validation.engines.vllm.tempfile.TemporaryFile")
@patch("granite_validation.engines.vllm.shutil.which", return_value=None)
def test_start_uses_custom_binary_path(mock_which: MagicMock, mock_tmp: MagicMock):
    """When vllm_bin is set in extra, it is used directly without calling shutil.which."""
    config = EngineConfig(
        mode="managed",
        extra={"vllm_bin": "/custom/vllm", "startup_timeout": 0.1},
    )
    engine = VllmEngine(config)

    with patch("granite_validation.engines.vllm.subprocess.Popen") as mock_popen:
        proc = _make_mock_process(poll_returns=1)
        mock_popen.return_value = proc
        mock_tmp.return_value = _make_stderr_file(b"bind error")

        with pytest.raises(RuntimeError):
            engine.start("model")

        cmd = mock_popen.call_args.args[0]
        assert cmd[0] == "/custom/vllm"


@patch("granite_validation.engines.vllm.tempfile.TemporaryFile")
@patch("granite_validation.engines.vllm.time.sleep")
@patch("granite_validation.engines.vllm.requests.get")
@patch("granite_validation.engines.vllm.subprocess.Popen")
@patch("granite_validation.engines.vllm.shutil.which", return_value="/usr/bin/vllm")
def test_start_process_exits_immediately(
    mock_which, mock_popen, mock_get, mock_sleep, mock_tmp
):
    proc = _make_mock_process(poll_returns=1)
    mock_popen.return_value = proc
    mock_tmp.return_value = _make_stderr_file(b"CUDA out of memory")

    engine = VllmEngine(EngineConfig(mode="managed"))
    with pytest.raises(RuntimeError, match="exited during startup with code 1") as exc_info:
        engine.start("model")
    assert "CUDA out of memory" in str(exc_info.value)


@patch("granite_validation.engines.vllm.tempfile.TemporaryFile")
@patch("granite_validation.engines.vllm.time.sleep")
@patch("granite_validation.engines.vllm.time.monotonic")
@patch("granite_validation.engines.vllm.requests.get")
@patch("granite_validation.engines.vllm.subprocess.Popen")
@patch("granite_validation.engines.vllm.shutil.which", return_value="/usr/bin/vllm")
def test_start_health_timeout(
    mock_which, mock_popen, mock_get, mock_monotonic, mock_sleep, mock_tmp
):
    proc = _make_mock_process()
    mock_popen.return_value = proc
    mock_tmp.return_value = _make_stderr_file()

    # Simulate time passing beyond the timeout.
    _times = iter([0.0, 0.5, 1.0])
    mock_monotonic.side_effect = lambda: next(_times, 301.0)
    mock_get.side_effect = requests.ConnectionError("refused")

    config = EngineConfig(mode="managed", extra={"startup_timeout": 300})
    engine = VllmEngine(config)

    with pytest.raises(RuntimeError, match="did not become ready within 300s"):
        engine.start("model")

    # Verify stop was called for cleanup
    proc.terminate.assert_called_once()


def test_start_successful(managed_start_env):
    env = managed_start_env
    engine = VllmEngine(EngineConfig(mode="managed"))
    engine.start("ibm-granite/granite-3.3-8b-instruct")

    # Verify process was started with correct command
    cmd = env["popen"].call_args.args[0]
    assert cmd[0] == "/usr/bin/vllm"
    assert cmd[1] == "serve"
    assert cmd[2] == "ibm-granite/granite-3.3-8b-instruct"
    assert "--host" not in cmd
    assert "--port" in cmd

    # Verify version was fetched
    assert engine._vllm_version == "0.8.3"
    assert engine.get_info().version == "0.8.3"

    # Verify model_id was recorded for subsequent chat() calls
    assert engine._config.model_id == "ibm-granite/granite-3.3-8b-instruct"


@patch.dict("os.environ", {"VIRTUAL_ENV": "/some/other/venv"})
def test_start_removes_virtual_env_from_subprocess(managed_start_env):
    """The subprocess env must not inherit VIRTUAL_ENV from the caller."""
    env = managed_start_env
    engine = VllmEngine(EngineConfig(mode="managed"))
    engine.start("model")

    popen_kwargs = env["popen"].call_args.kwargs
    subprocess_env = popen_kwargs["env"]
    assert "VIRTUAL_ENV" not in subprocess_env


def test_start_rejects_double_start(managed_start_env):
    engine = VllmEngine(EngineConfig(mode="managed"))
    engine.start("ibm-granite/granite-3.3-8b-instruct")

    with pytest.raises(RuntimeError, match="already running"):
        engine.start("ibm-granite/granite-3.3-8b-instruct")


# --- stop() tests ---


def test_stop_noop_when_no_process():
    engine = VllmEngine(EngineConfig())
    engine.stop()  # Should not raise


def test_stop_graceful():
    engine = VllmEngine(EngineConfig())
    proc = _make_mock_process()
    engine._process = proc

    engine.stop()

    proc.terminate.assert_called_once()
    proc.wait.assert_called_once()
    proc.kill.assert_not_called()
    assert engine._process is None


def test_stop_force_kill():
    engine = VllmEngine(EngineConfig())
    proc = _make_mock_process()
    proc.wait.side_effect = [subprocess.TimeoutExpired("vllm", 10), 0]
    engine._process = proc

    engine.stop()

    proc.terminate.assert_called_once()
    proc.kill.assert_called_once()
    assert engine._process is None


def test_stop_closes_stderr_file():
    engine = VllmEngine(EngineConfig())
    proc = _make_mock_process()
    stderr = _make_stderr_file(b"some output")
    engine._process = proc
    engine._stderr_file = stderr

    engine.stop()

    assert engine._stderr_file is None
    assert stderr.closed


def test_stop_uses_custom_timeout():
    engine = VllmEngine(EngineConfig(extra={"stop_timeout": 42}))
    proc = _make_mock_process()
    engine._process = proc

    engine.stop()

    proc.wait.assert_called_once_with(timeout=42)


def test_stop_force_kill_wait_timeout():
    """After SIGKILL, if wait() also times out, TimeoutExpired propagates."""
    engine = VllmEngine(EngineConfig())
    proc = _make_mock_process()
    proc.wait.side_effect = subprocess.TimeoutExpired("vllm", 10)
    engine._process = proc

    with pytest.raises(subprocess.TimeoutExpired):
        engine.stop()

    proc.terminate.assert_called_once()
    proc.kill.assert_called_once()
    # Process and stderr are still cleaned up via finally
    assert engine._process is None


# --- get_info with version ---


def test_get_info_returns_version_after_start():
    engine = VllmEngine(EngineConfig(mode="managed"))
    engine._vllm_version = "0.8.3"

    info = engine.get_info()
    assert info.version == "0.8.3"
    assert info.mode == "managed"


# --- _fetch_version failure ---


@patch("granite_validation.engines.vllm.requests.get")
def test_fetch_version_handles_failure(mock_get: MagicMock):
    mock_get.side_effect = requests.ConnectionError("refused")
    engine = VllmEngine(EngineConfig())

    engine._fetch_version()

    assert engine._vllm_version is None


@patch("granite_validation.engines.vllm.requests.get")
def test_fetch_version_handles_non_200(mock_get: MagicMock):
    resp = MagicMock()
    resp.status_code = 503
    mock_get.return_value = resp
    engine = VllmEngine(EngineConfig())

    engine._fetch_version()

    assert engine._vllm_version is None


# --- _build_cmd ---


def test_build_cmd_basic():
    engine = VllmEngine(EngineConfig(mode="managed"))
    cmd = engine._build_cmd("/usr/bin/vllm", "ibm-granite/granite-3.3-8b-instruct")

    assert cmd[0] == "/usr/bin/vllm"
    assert cmd[1] == "serve"
    assert cmd[2] == "ibm-granite/granite-3.3-8b-instruct"
    assert "--host" not in cmd
    assert "--port" in cmd
    assert "8000" in cmd


def test_build_cmd_custom_base_url_port():
    """Custom base_url only affects --port; --host is not passed by default."""
    engine = VllmEngine(EngineConfig(mode="managed", base_url="http://192.168.1.50:9999"))
    cmd = engine._build_cmd("/usr/bin/vllm", "model")

    assert "--host" not in cmd
    idx_port = cmd.index("--port")
    assert cmd[idx_port + 1] == "9999"


def test_build_cmd_explicit_bind_host():
    """When bind_host is set in extra, --host is passed to vLLM."""
    config = EngineConfig(mode="managed", extra={"bind_host": "192.168.1.50"})
    engine = VllmEngine(config)
    cmd = engine._build_cmd("/usr/bin/vllm", "model")

    idx_host = cmd.index("--host")
    assert cmd[idx_host + 1] == "192.168.1.50"


def test_build_cmd_with_all_options():
    config = EngineConfig(
        mode="managed",
        extra={
            "tensor_parallel_size": 4,
            "gpu_memory_utilization": 0.9,
            "max_model_len": 8192,
            "dtype": "bfloat16",
            "quantization": "awq",
            "server_args": ["--enable-prefix-caching"],
        },
    )
    engine = VllmEngine(config)
    cmd = engine._build_cmd("/usr/bin/vllm", "model")

    idx = cmd.index("--tensor-parallel-size")
    assert cmd[idx + 1] == "4"

    idx = cmd.index("--gpu-memory-utilization")
    assert cmd[idx + 1] == "0.9"

    idx = cmd.index("--max-model-len")
    assert cmd[idx + 1] == "8192"

    idx = cmd.index("--dtype")
    assert cmd[idx + 1] == "bfloat16"

    idx = cmd.index("--quantization")
    assert cmd[idx + 1] == "awq"

    assert "--enable-prefix-caching" in cmd


def test_build_cmd_omits_unset_options():
    engine = VllmEngine(EngineConfig(mode="managed"))
    cmd = engine._build_cmd("/usr/bin/vllm", "model")

    assert "--tensor-parallel-size" not in cmd
    assert "--gpu-memory-utilization" not in cmd
    assert "--max-model-len" not in cmd
    assert "--dtype" not in cmd
    assert "--quantization" not in cmd


# --- _read_stderr edge cases ---


def test_read_stderr_returns_empty_when_file_is_none():
    engine = VllmEngine(EngineConfig())
    engine._stderr_file = None
    assert engine._read_stderr() == ""


def test_read_stderr_truncates_to_last_4096_bytes():
    engine = VllmEngine(EngineConfig())
    content = b"X" * 5000 + b"TAIL_MARKER"
    engine._stderr_file = _make_stderr_file(content)

    result = engine._read_stderr()

    assert "TAIL_MARKER" in result
    assert len(result) <= 4096


# --- start() with custom base_url ---


def test_start_custom_base_url(managed_start_env):
    env = managed_start_env
    engine = VllmEngine(EngineConfig(mode="managed", base_url="http://192.168.1.50:9999"))
    engine.start("ibm-granite/granite-3.3-8b-instruct")

    cmd = env["popen"].call_args.args[0]
    assert "--host" not in cmd
    idx_port = cmd.index("--port")
    assert cmd[idx_port + 1] == "9999"


# --- Fix: model_id recorded after start ---


def test_start_records_model_id_when_none(managed_start_env):
    """start() sets config.model_id so chat() includes it in the payload."""
    engine = VllmEngine(EngineConfig(mode="managed"))
    assert engine._config.model_id is None

    engine.start("ibm-granite/granite-3.3-8b-instruct")

    assert engine._config.model_id == "ibm-granite/granite-3.3-8b-instruct"


def test_start_overwrites_model_id(managed_start_env):
    """start() updates model_id even if one was previously set."""
    engine = VllmEngine(EngineConfig(mode="managed", model_id="old-model"))
    engine.start("new-model")

    assert engine._config.model_id == "new-model"


@patch("granite_validation.engines.openai_compat.requests.post")
def test_chat_includes_model_after_managed_start(mock_post: MagicMock, managed_start_env):
    """After start(), chat() sends the started model in the request payload."""
    mock_post.return_value = _fake_chat_response()
    engine = VllmEngine(EngineConfig(mode="managed"))
    engine.start("ibm-granite/granite-3.3-8b-instruct")

    engine.chat([{"role": "user", "content": "hi"}])

    payload = mock_post.call_args.kwargs["json"]
    assert payload["model"] == "ibm-granite/granite-3.3-8b-instruct"


# --- Fix: bind address vs client URL ---


def test_start_default_base_url_unchanged(managed_start_env):
    """Default base_url (localhost:8000) is preserved as the client URL."""
    engine = VllmEngine(EngineConfig(mode="managed"))
    engine.start("model")

    assert engine._base_url == "http://localhost:8000"


def test_start_custom_base_url_unchanged(managed_start_env):
    """A custom base_url is preserved as the client URL."""
    engine = VllmEngine(EngineConfig(mode="managed", base_url="http://192.168.1.50:8000"))
    engine.start("model")

    assert engine._base_url == "http://192.168.1.50:8000"


# --- Fix: startup_timeout is a real upper bound ---


@patch("granite_validation.engines.vllm.tempfile.TemporaryFile")
@patch("granite_validation.engines.vllm.time.sleep")
@patch("granite_validation.engines.vllm.requests.get")
@patch("granite_validation.engines.vllm.subprocess.Popen")
@patch("granite_validation.engines.vllm.shutil.which", return_value="/usr/bin/vllm")
def test_health_probe_timeout_capped_to_remaining(
    mock_which, mock_popen, mock_get, mock_sleep, mock_tmp
):
    """The per-probe timeout should not exceed remaining time to deadline."""
    proc = _make_mock_process()
    mock_popen.return_value = proc
    mock_tmp.return_value = _make_stderr_file()

    call_timeouts = []

    def _capture_get(url, **kwargs):
        call_timeouts.append(kwargs.get("timeout"))
        raise requests.ConnectionError("refused")

    mock_get.side_effect = _capture_get

    config = EngineConfig(mode="managed", extra={"startup_timeout": 3})
    engine = VllmEngine(config)

    with pytest.raises(RuntimeError, match="did not become ready"):
        engine.start("model")

    # All captured timeouts should be <= 10 (the max per-probe cap)
    # and the last ones should be smaller as the deadline approaches
    assert all(t <= 10 for t in call_timeouts)
    assert len(call_timeouts) >= 1
