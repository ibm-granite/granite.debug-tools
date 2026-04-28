from __future__ import annotations

import io
import subprocess
from unittest.mock import MagicMock, patch

import pytest
import requests

from granite_validation.engines.base import EngineConfig
from granite_validation.engines.llamacpp import LlamaCppEngine


# --- Helpers ---


def _fake_health_ok() -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    return resp


def _fake_props_response() -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "build_info": {"version": "b5678", "build_number": 5678},
        "default_generation_settings": {},
    }
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
def managed_start_env(tmp_path):
    """Common mock environment for managed start() tests.

    Patches: shutil.which, subprocess.Popen, health_check (via openai_compat),
    time.sleep, tempfile.TemporaryFile, and llamacpp-module requests.get (for props).

    Returns a dict with keys: popen, process, health_get, props_get, stderr_file, model_path.
    """
    model_file = tmp_path / "model.gguf"
    model_file.write_bytes(b"GGUF_FAKE")

    with (
        patch(
            "granite_validation.engines.llamacpp.shutil.which",
            return_value="/usr/bin/llama-server",
        ),
        patch("granite_validation.engines.llamacpp.subprocess.Popen") as mock_popen,
        patch("granite_validation.engines.openai_compat.requests.get") as mock_compat_get,
        patch("granite_validation.engines.llamacpp.time.sleep"),
        patch("granite_validation.engines.llamacpp.requests.get") as mock_llamacpp_get,
        patch("granite_validation.engines.llamacpp.tempfile.TemporaryFile") as mock_tmp,
    ):
        proc = _make_mock_process()
        mock_popen.return_value = proc
        mock_tmp.return_value = _make_stderr_file()

        # Default: health OK
        mock_compat_get.return_value = _fake_health_ok()

        # Props response for version fetch
        mock_llamacpp_get.return_value = _fake_props_response()

        yield {
            "popen": mock_popen,
            "process": proc,
            "health_get": mock_compat_get,
            "props_get": mock_llamacpp_get,
            "stderr_file": mock_tmp,
            "model_path": str(model_file),
        }


# --- Default port / base_url ---


def test_llamacpp_default_port():
    engine = LlamaCppEngine(EngineConfig())
    assert engine._base_url == "http://localhost:8080"


def test_llamacpp_custom_base_url():
    engine = LlamaCppEngine(EngineConfig(base_url="http://gpu-box:9090"))
    assert engine._base_url == "http://gpu-box:9090"


# --- engine_id / get_info ---


def test_llamacpp_engine_id():
    assert LlamaCppEngine(EngineConfig()).engine_id() == "llamacpp"


def test_llamacpp_get_info():
    config = EngineConfig(mode="external", base_url="http://myhost:8080")
    info = LlamaCppEngine(config).get_info()
    assert info.engine_id == "llamacpp"
    assert info.mode == "external"
    assert info.base_url == "http://myhost:8080"
    assert info.version == "unknown"


# --- health_check() ---


@patch("granite_validation.engines.openai_compat.requests.get")
def test_llamacpp_health_check_hits_health_endpoint(mock_get: MagicMock):
    mock_get.return_value = _fake_health_ok()
    engine = LlamaCppEngine(EngineConfig(base_url="http://myhost:8080"))

    engine.health_check()

    url = mock_get.call_args.args[0]
    assert url == "http://myhost:8080/health"


@patch("granite_validation.engines.openai_compat.requests.get")
def test_llamacpp_health_check_returns_true_on_200(mock_get: MagicMock):
    mock_get.return_value = _fake_health_ok()
    engine = LlamaCppEngine(EngineConfig())
    assert engine.health_check() is True


@patch("granite_validation.engines.openai_compat.requests.get")
def test_llamacpp_health_check_returns_false_on_connection_error(mock_get: MagicMock):
    mock_get.side_effect = requests.ConnectionError("refused")
    engine = LlamaCppEngine(EngineConfig())
    assert engine.health_check() is False


# --- chat() ---


@patch("granite_validation.engines.openai_compat.requests.post")
def test_llamacpp_chat_posts_to_correct_url(mock_post: MagicMock):
    mock_post.return_value = _fake_chat_response()
    engine = LlamaCppEngine(EngineConfig(base_url="http://gpu:8080", model_id="model.gguf"))

    engine.chat([{"role": "user", "content": "hi"}])

    url = mock_post.call_args.args[0]
    assert url == "http://gpu:8080/v1/chat/completions"


# --- start() tests ---


def test_start_rejects_external_mode():
    engine = LlamaCppEngine(EngineConfig(mode="external"))
    with pytest.raises(ValueError, match="mode is 'external'"):
        engine.start("model.gguf")


@patch("granite_validation.engines.llamacpp.shutil.which", return_value=None)
def test_start_binary_not_found(mock_which: MagicMock, tmp_path):
    model_file = tmp_path / "model.gguf"
    model_file.write_bytes(b"GGUF")
    engine = LlamaCppEngine(EngineConfig(mode="managed"))
    with pytest.raises(RuntimeError, match="'llama-server' binary not found"):
        engine.start(str(model_file))


def test_start_model_file_not_found():
    engine = LlamaCppEngine(EngineConfig(mode="managed"))
    with pytest.raises(FileNotFoundError, match="Model file not found"):
        engine.start("/nonexistent/model.gguf")


@patch("granite_validation.engines.llamacpp.tempfile.TemporaryFile")
@patch("granite_validation.engines.llamacpp.shutil.which", return_value=None)
def test_start_uses_custom_binary_path(mock_which: MagicMock, mock_tmp: MagicMock, tmp_path):
    """When llamacpp_bin is set in extra, it is used directly without calling shutil.which."""
    model_file = tmp_path / "model.gguf"
    model_file.write_bytes(b"GGUF")
    config = EngineConfig(
        mode="managed",
        extra={"llamacpp_bin": "/custom/llama-server", "startup_timeout": 0.1},
    )
    engine = LlamaCppEngine(config)

    with patch("granite_validation.engines.llamacpp.subprocess.Popen") as mock_popen:
        proc = _make_mock_process(poll_returns=1)
        mock_popen.return_value = proc
        mock_tmp.return_value = _make_stderr_file(b"bind error")

        with pytest.raises(RuntimeError):
            engine.start(str(model_file))

        cmd = mock_popen.call_args.args[0]
        assert cmd[0] == "/custom/llama-server"


@patch("granite_validation.engines.llamacpp.tempfile.TemporaryFile")
@patch("granite_validation.engines.llamacpp.time.sleep")
@patch("granite_validation.engines.llamacpp.requests.get")
@patch("granite_validation.engines.llamacpp.subprocess.Popen")
@patch(
    "granite_validation.engines.llamacpp.shutil.which", return_value="/usr/bin/llama-server"
)
def test_start_process_exits_immediately(
    mock_which, mock_popen, mock_get, mock_sleep, mock_tmp, tmp_path
):
    model_file = tmp_path / "model.gguf"
    model_file.write_bytes(b"GGUF")

    proc = _make_mock_process(poll_returns=1)
    mock_popen.return_value = proc
    mock_tmp.return_value = _make_stderr_file(b"address already in use")

    engine = LlamaCppEngine(EngineConfig(mode="managed"))
    with pytest.raises(RuntimeError, match="exited during startup with code 1") as exc_info:
        engine.start(str(model_file))
    assert "address already in use" in str(exc_info.value)


@patch("granite_validation.engines.llamacpp.tempfile.TemporaryFile")
@patch("granite_validation.engines.llamacpp.time.sleep")
@patch("granite_validation.engines.llamacpp.time.monotonic")
@patch("granite_validation.engines.openai_compat.requests.get")
@patch("granite_validation.engines.llamacpp.subprocess.Popen")
@patch(
    "granite_validation.engines.llamacpp.shutil.which", return_value="/usr/bin/llama-server"
)
def test_start_health_timeout(
    mock_which, mock_popen, mock_get, mock_monotonic, mock_sleep, mock_tmp, tmp_path
):
    model_file = tmp_path / "model.gguf"
    model_file.write_bytes(b"GGUF")

    proc = _make_mock_process()
    mock_popen.return_value = proc
    mock_tmp.return_value = _make_stderr_file()

    # Simulate time passing beyond the timeout.
    _times = iter([0.0, 0.5, 1.0])
    mock_monotonic.side_effect = lambda: next(_times, 121.0)
    mock_get.side_effect = requests.ConnectionError("refused")

    config = EngineConfig(mode="managed", extra={"startup_timeout": 120})
    engine = LlamaCppEngine(config)

    with pytest.raises(RuntimeError, match="did not become ready within 120s"):
        engine.start(str(model_file))

    # Verify stop was called for cleanup
    proc.terminate.assert_called_once()


def test_start_successful(managed_start_env):
    env = managed_start_env
    engine = LlamaCppEngine(EngineConfig(mode="managed"))
    engine.start(env["model_path"])

    # Verify process was started with correct command
    cmd = env["popen"].call_args.args[0]
    assert cmd[0] == "/usr/bin/llama-server"
    assert "--model" in cmd
    assert env["model_path"] in cmd
    assert "--host" in cmd
    assert "--port" in cmd

    # Verify version was fetched
    assert engine._server_version == "b5678"
    assert engine.get_info().version == "b5678"


def test_start_rejects_double_start(managed_start_env):
    env = managed_start_env
    engine = LlamaCppEngine(EngineConfig(mode="managed"))
    engine.start(env["model_path"])

    with pytest.raises(RuntimeError, match="already running"):
        engine.start(env["model_path"])


def test_start_with_extra_options(managed_start_env):
    env = managed_start_env
    config = EngineConfig(
        mode="managed",
        extra={
            "n_gpu_layers": 99,
            "ctx_size": 4096,
            "jinja": True,
            "parallel": 2,
            "server_args": ["--temp", "0"],
        },
    )
    engine = LlamaCppEngine(config)
    engine.start(env["model_path"])

    cmd = env["popen"].call_args.args[0]
    assert "--n-gpu-layers" in cmd
    assert "99" in cmd
    assert "--ctx-size" in cmd
    assert "4096" in cmd
    assert "--jinja" in cmd
    assert "--parallel" in cmd
    assert "2" in cmd
    assert "--temp" in cmd
    assert "0" in cmd


# --- stop() tests ---


def test_stop_noop_when_no_process():
    engine = LlamaCppEngine(EngineConfig())
    engine.stop()  # Should not raise


def test_stop_graceful():
    engine = LlamaCppEngine(EngineConfig())
    proc = _make_mock_process()
    engine._process = proc

    engine.stop()

    proc.terminate.assert_called_once()
    proc.wait.assert_called_once()
    proc.kill.assert_not_called()
    assert engine._process is None


def test_stop_force_kill():
    engine = LlamaCppEngine(EngineConfig())
    proc = _make_mock_process()
    proc.wait.side_effect = [subprocess.TimeoutExpired("llama-server", 10), 0]
    engine._process = proc

    engine.stop()

    proc.terminate.assert_called_once()
    proc.kill.assert_called_once()
    assert engine._process is None


def test_stop_closes_stderr_file():
    engine = LlamaCppEngine(EngineConfig())
    proc = _make_mock_process()
    stderr = _make_stderr_file(b"some output")
    engine._process = proc
    engine._stderr_file = stderr

    engine.stop()

    assert engine._stderr_file is None
    assert stderr.closed


def test_stop_uses_custom_timeout():
    engine = LlamaCppEngine(EngineConfig(extra={"stop_timeout": 42}))
    proc = _make_mock_process()
    engine._process = proc

    engine.stop()

    proc.wait.assert_called_once_with(timeout=42)


def test_stop_force_kill_wait_timeout():
    """After SIGKILL, if wait() also times out, TimeoutExpired propagates."""
    engine = LlamaCppEngine(EngineConfig())
    proc = _make_mock_process()
    proc.wait.side_effect = subprocess.TimeoutExpired("llama-server", 10)
    engine._process = proc

    with pytest.raises(subprocess.TimeoutExpired):
        engine.stop()

    proc.terminate.assert_called_once()
    proc.kill.assert_called_once()
    # Process and stderr are still cleaned up via finally
    assert engine._process is None


# --- get_info with version ---


def test_get_info_returns_version_after_start():
    engine = LlamaCppEngine(EngineConfig(mode="managed"))
    engine._server_version = "b5678"

    info = engine.get_info()
    assert info.version == "b5678"
    assert info.mode == "managed"


# --- _fetch_version failure ---


@patch("granite_validation.engines.llamacpp.requests.get")
def test_fetch_version_handles_failure(mock_get: MagicMock):
    mock_get.side_effect = requests.ConnectionError("refused")
    engine = LlamaCppEngine(EngineConfig())

    engine._fetch_version()

    assert engine._server_version is None


# --- _build_cmd() ---


def test_build_cmd_default():
    engine = LlamaCppEngine(EngineConfig())
    cmd = engine._build_cmd("/usr/bin/llama-server", "/path/to/model.gguf")
    assert cmd == [
        "/usr/bin/llama-server",
        "--model", "/path/to/model.gguf",
        "--host", "localhost",
        "--port", "8080",
    ]


def test_build_cmd_custom_base_url():
    engine = LlamaCppEngine(EngineConfig(base_url="http://192.168.1.100:9999"))
    cmd = engine._build_cmd("/usr/bin/llama-server", "/m.gguf")
    assert "--host" in cmd
    idx = cmd.index("--host")
    assert cmd[idx + 1] == "192.168.1.100"
    idx = cmd.index("--port")
    assert cmd[idx + 1] == "9999"


def test_build_cmd_with_all_options():
    config = EngineConfig(
        extra={
            "n_gpu_layers": 99,
            "ctx_size": 8192,
            "jinja": True,
            "parallel": 4,
            "server_args": ["--flash-attn"],
        }
    )
    engine = LlamaCppEngine(config)
    cmd = engine._build_cmd("/bin/ls", "/m.gguf")

    assert "--n-gpu-layers" in cmd
    assert "99" in cmd
    assert "--ctx-size" in cmd
    assert "8192" in cmd
    assert "--jinja" in cmd
    assert "--parallel" in cmd
    assert "4" in cmd
    assert "--flash-attn" in cmd


def test_build_cmd_jinja_false_not_added():
    config = EngineConfig(extra={"jinja": False})
    engine = LlamaCppEngine(config)
    cmd = engine._build_cmd("/bin/ls", "/m.gguf")
    assert "--jinja" not in cmd


# --- _read_stderr edge cases ---


def test_read_stderr_returns_empty_when_file_is_none():
    engine = LlamaCppEngine(EngineConfig())
    engine._stderr_file = None
    assert engine._read_stderr() == ""


def test_read_stderr_truncates_to_last_4096_bytes():
    engine = LlamaCppEngine(EngineConfig())
    content = b"X" * 5000 + b"TAIL_MARKER"
    engine._stderr_file = _make_stderr_file(content)

    result = engine._read_stderr()

    assert "TAIL_MARKER" in result
    assert len(result) <= 4096


# --- start() with custom base_url ---


def test_start_custom_base_url(managed_start_env):
    env = managed_start_env
    engine = LlamaCppEngine(
        EngineConfig(mode="managed", base_url="http://192.168.1.50:9999")
    )
    engine.start(env["model_path"])

    cmd = env["popen"].call_args.args[0]
    assert "--host" in cmd
    idx = cmd.index("--host")
    assert cmd[idx + 1] == "192.168.1.50"
    idx = cmd.index("--port")
    assert cmd[idx + 1] == "9999"
