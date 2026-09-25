"""Tests for MCP tools."""

import asyncio
import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import perfbench.tools as tools_mod
from perfbench.tools import (
    _PRESETS,
    _RESULT_DIRS,
    _BenchmarkEntry,
    _benchmarks,
    _safe_result_dir,
    _save_stdout_result,
    _stream_reader,
    check_aiperf_benchmark_status,
    check_guidellm_benchmark_status,
    check_llama_bench_status,
    check_ollama_benchmark_status,
    check_vllm_benchmark_status,
    compare_results,
    list_benchmarks,
    list_results,
    ping,
    read_result,
    run_aiperf_benchmark,
    run_benchmark_preset,
    run_guidellm_benchmark,
    run_llama_bench,
    run_ollama_benchmark,
    run_streamlit_dashboard,
    run_vllm_benchmark,
    stop_aiperf_benchmark,
    stop_guidellm_benchmark,
    stop_llama_bench,
    stop_ollama_benchmark,
    stop_streamlit_dashboard,
    stop_vllm_benchmark,
)


@pytest.fixture(autouse=True)
def _clean_benchmark_state():
    """Ensure clean state before and after each test."""
    _benchmarks.clear()
    tools_mod._streamlit_proc = None
    yield
    _benchmarks.clear()
    tools_mod._streamlit_proc = None


def test_ping():
    """Test that the ping tool returns 'pong'."""
    assert ping() == "pong"


@pytest.mark.asyncio
async def test_run_vllm_benchmark_returns_id():
    """Test that run_vllm_benchmark launches and returns an ID."""
    mock_proc = MagicMock()
    mock_proc.returncode = None
    mock_proc.pid = 12345

    # Mock stdout/stderr with async readline
    stdout_data = [b"Starting benchmark...\n"]
    idx = {"out": 0}

    async def stdout_readline():
        if idx["out"] < len(stdout_data):
            line = stdout_data[idx["out"]]
            idx["out"] += 1
            return line
        return b""

    async def stderr_readline():
        return b""

    mock_proc.stdout = MagicMock()
    mock_proc.stdout.readline = stdout_readline
    mock_proc.stderr = MagicMock()
    mock_proc.stderr.readline = stderr_readline

    with (
        patch(
            "perfbench.tools.asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=mock_proc),
        ),
        patch(
            "perfbench.tools.asyncio.sleep",
            new=AsyncMock(),
        ),
    ):
        result = await run_vllm_benchmark(
            model="ibm-granite/granite-4.0-micro",
            base_url="https://vllm.example.com/",
            served_model_name="granite-4.0-micro",
        )

    assert "Benchmark started" in result
    assert "ID:" in result
    benchmark_id = result.split("ID: ")[1].split(")")[0]
    assert benchmark_id in _benchmarks


@pytest.mark.asyncio
async def test_run_vllm_benchmark_not_installed():
    """Test error message when vllm is not installed."""
    with patch(
        "perfbench.tools.asyncio.create_subprocess_exec",
        new=AsyncMock(side_effect=FileNotFoundError("No such file: 'vllm'")),
    ):
        result = await run_vllm_benchmark(
            model="my-model",
            base_url="http://localhost:8000",
            served_model_name="my-model",
        )

    assert "not installed" in result
    assert "pip install vllm" in result


@pytest.mark.asyncio
async def test_check_status_running_with_output():
    """Test status check returns new output for a running benchmark."""
    mock_proc = MagicMock()
    mock_proc.returncode = None
    mock_proc.pid = 99999

    entry = _BenchmarkEntry(proc=mock_proc)
    entry.output_lines = ["line1\n", "line2\n"]
    entry._read_cursor = 0  # nothing read yet

    _benchmarks["test123"] = entry
    result = await check_vllm_benchmark_status("test123")
    assert "still running" in result
    assert "line1" in result
    assert "line2" in result


@pytest.mark.asyncio
async def test_check_status_incremental_output():
    """Test that check_status returns only new output from _read_cursor."""
    mock_proc = MagicMock()
    mock_proc.returncode = None
    mock_proc.pid = 99998

    entry = _BenchmarkEntry(proc=mock_proc)
    entry.output_lines = ["line1\n", "line2\n", "line3\n"]
    entry._read_cursor = 1  # line1 was already returned

    _benchmarks["incr01"] = entry
    result = await check_vllm_benchmark_status("incr01")
    assert "still running" in result
    assert "line1" not in result
    assert "line2" in result
    assert "line3" in result
    # Cursor should advance to current length
    assert entry._read_cursor == 3


@pytest.mark.asyncio
async def test_check_status_completed():
    """Test status check for a completed benchmark."""
    mock_proc = MagicMock()
    mock_proc.returncode = 0

    entry = _BenchmarkEntry(proc=mock_proc)
    entry.output_lines = ["result line 1\n", "result line 2\n"]

    _benchmarks["done456"] = entry
    result = await check_vllm_benchmark_status("done456")

    assert "completed" in result
    assert "result line 1" in result
    assert "done456" not in _benchmarks


@pytest.mark.asyncio
async def test_check_status_failed():
    """Test status check for a failed benchmark."""
    mock_proc = MagicMock()
    mock_proc.returncode = 1

    entry = _BenchmarkEntry(proc=mock_proc)
    entry.output_lines = ["error: connection refused\n"]

    _benchmarks["fail789"] = entry
    result = await check_vllm_benchmark_status("fail789")

    assert "failed" in result
    assert "connection refused" in result
    assert "fail789" not in _benchmarks


@pytest.mark.asyncio
async def test_check_status_not_found():
    """Test status check for an unknown benchmark ID."""
    result = await check_vllm_benchmark_status("nonexistent")
    assert "not found" in result


@pytest.mark.asyncio
async def test_stop_benchmark():
    """Test stopping a running benchmark."""
    mock_proc = AsyncMock()
    mock_proc.returncode = None
    mock_proc.pid = 55555

    entry = _BenchmarkEntry(proc=mock_proc)
    _benchmarks["stop123"] = entry

    result = await stop_vllm_benchmark("stop123")

    assert "terminated" in result
    assert "55555" in result
    assert "stop123" not in _benchmarks
    mock_proc.terminate.assert_called_once()


# ── aiperf tool tests ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_aiperf_benchmark_returns_id():
    """Test that run_aiperf_benchmark launches and returns an ID."""
    mock_proc = MagicMock()
    mock_proc.returncode = None
    mock_proc.pid = 22222

    stdout_data = [b"AIPerf starting...\n"]
    idx = {"out": 0}

    async def stdout_readline():
        if idx["out"] < len(stdout_data):
            line = stdout_data[idx["out"]]
            idx["out"] += 1
            return line
        return b""

    async def stderr_readline():
        return b""

    mock_proc.stdout = MagicMock()
    mock_proc.stdout.readline = stdout_readline
    mock_proc.stderr = MagicMock()
    mock_proc.stderr.readline = stderr_readline

    with (
        patch(
            "perfbench.tools.asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=mock_proc),
        ),
        patch(
            "perfbench.tools.asyncio.sleep",
            new=AsyncMock(),
        ),
    ):
        result = await run_aiperf_benchmark(
            model="ibm-granite/granite-4.0-micro",
            tokenizer="ibm-granite/granite-4.0-micro",
            url="http://localhost:8000",
        )

    assert "Benchmark started" in result
    assert "ID:" in result
    benchmark_id = result.split("ID: ")[1].split(")")[0]
    assert benchmark_id in _benchmarks


@pytest.mark.asyncio
async def test_run_aiperf_benchmark_not_installed():
    """Test error message when aiperf is not installed."""
    with patch(
        "perfbench.tools.asyncio.create_subprocess_exec",
        new=AsyncMock(side_effect=FileNotFoundError("No such file: 'aiperf'")),
    ):
        result = await run_aiperf_benchmark(
            model="my-model",
            tokenizer="my-model",
            url="http://localhost:8000",
        )

    assert "not installed" in result
    assert "pip install aiperf" in result


@pytest.mark.asyncio
async def test_check_aiperf_status_running():
    """Test aiperf status check for a running benchmark."""
    mock_proc = MagicMock()
    mock_proc.returncode = None
    mock_proc.pid = 33333

    entry = _BenchmarkEntry(proc=mock_proc)
    entry.output_lines = ["Profiling in progress...\n"]

    _benchmarks["aiperf01"] = entry
    result = await check_aiperf_benchmark_status("aiperf01")
    assert "still running" in result
    assert "Profiling in progress" in result


@pytest.mark.asyncio
async def test_check_aiperf_status_completed():
    """Test aiperf status check for a completed benchmark."""
    mock_proc = MagicMock()
    mock_proc.returncode = 0

    entry = _BenchmarkEntry(proc=mock_proc)
    entry.output_lines = ["Benchmark complete\n"]

    _benchmarks["aiperf02"] = entry
    result = await check_aiperf_benchmark_status("aiperf02")

    assert "completed" in result
    assert "Benchmark complete" in result
    assert "aiperf02" not in _benchmarks


@pytest.mark.asyncio
async def test_stop_aiperf_benchmark():
    """Test stopping a running aiperf benchmark."""
    mock_proc = AsyncMock()
    mock_proc.returncode = None
    mock_proc.pid = 44444

    entry = _BenchmarkEntry(proc=mock_proc)
    _benchmarks["aiperf03"] = entry

    result = await stop_aiperf_benchmark("aiperf03")

    assert "terminated" in result
    assert "44444" in result
    assert "aiperf03" not in _benchmarks
    mock_proc.terminate.assert_called_once()


# ── guidellm tests ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_guidellm_benchmark_returns_id():
    """Test that run_guidellm_benchmark launches and returns an ID."""
    mock_proc = MagicMock()
    mock_proc.returncode = None
    mock_proc.pid = 55555

    stdout_data = [b"GuideLLM starting...\n"]
    idx = {"out": 0}

    async def stdout_readline():
        if idx["out"] < len(stdout_data):
            line = stdout_data[idx["out"]]
            idx["out"] += 1
            return line
        return b""

    async def stderr_readline():
        return b""

    mock_proc.stdout = MagicMock()
    mock_proc.stdout.readline = stdout_readline
    mock_proc.stderr = MagicMock()
    mock_proc.stderr.readline = stderr_readline

    with (
        patch(
            "perfbench.tools.asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=mock_proc),
        ),
        patch(
            "perfbench.tools.asyncio.sleep",
            new=AsyncMock(),
        ),
    ):
        result = await run_guidellm_benchmark(
            target="http://localhost:8000",
        )

    assert "Benchmark started" in result
    assert "ID:" in result
    benchmark_id = result.split("ID: ")[1].split(")")[0]
    assert benchmark_id in _benchmarks


@pytest.mark.asyncio
async def test_run_guidellm_benchmark_not_installed():
    """Test error message when guidellm is not installed."""
    with patch(
        "perfbench.tools.asyncio.create_subprocess_exec",
        new=AsyncMock(side_effect=FileNotFoundError("No such file: 'guidellm'")),
    ):
        result = await run_guidellm_benchmark(
            target="http://localhost:8000",
        )

    assert "not installed" in result
    assert "guidellm" in result


@pytest.mark.asyncio
async def test_check_guidellm_status_running():
    """Test guidellm status check for a running benchmark."""
    mock_proc = MagicMock()
    mock_proc.returncode = None
    mock_proc.pid = 66666

    entry = _BenchmarkEntry(proc=mock_proc)
    entry.output_lines = ["Benchmark sweep in progress...\n"]

    _benchmarks["guide01"] = entry
    result = await check_guidellm_benchmark_status("guide01")
    assert "still running" in result
    assert "Benchmark sweep in progress" in result


@pytest.mark.asyncio
async def test_check_guidellm_status_completed():
    """Test guidellm status check for a completed benchmark."""
    mock_proc = MagicMock()
    mock_proc.returncode = 0

    entry = _BenchmarkEntry(proc=mock_proc)
    entry.output_lines = ["Benchmark complete\n"]

    _benchmarks["guide02"] = entry
    result = await check_guidellm_benchmark_status("guide02")

    assert "completed" in result
    assert "Benchmark complete" in result
    assert "guide02" not in _benchmarks


@pytest.mark.asyncio
async def test_stop_guidellm_benchmark():
    """Test stopping a running guidellm benchmark."""
    mock_proc = AsyncMock()
    mock_proc.returncode = None
    mock_proc.pid = 77777

    entry = _BenchmarkEntry(proc=mock_proc)
    _benchmarks["guide03"] = entry

    result = await stop_guidellm_benchmark("guide03")

    assert "terminated" in result
    assert "77777" in result
    assert "guide03" not in _benchmarks
    mock_proc.terminate.assert_called_once()


# ── Streamlit dashboard tests ────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_streamlit_dashboard():
    """Test that run_streamlit_dashboard launches and returns the URL."""
    mock_proc = AsyncMock()
    mock_proc.returncode = None
    mock_proc.pid = 88888

    # Simulate stderr output (Streamlit writes its startup info to stderr)
    async def _readline():
        return b""

    mock_proc.stderr = MagicMock()
    mock_proc.stderr.readline = _readline

    with patch(
        "perfbench.tools.asyncio.create_subprocess_exec",
        return_value=mock_proc,
    ):
        result = await run_streamlit_dashboard(port=8502)

    assert "Dashboard started" in result
    assert "88888" in result
    assert "http://localhost:8502" in result


@pytest.mark.asyncio
async def test_run_streamlit_dashboard_not_installed():
    """Test the error message when streamlit is not installed."""
    with patch(
        "perfbench.tools.asyncio.create_subprocess_exec",
        side_effect=FileNotFoundError,
    ):
        result = await run_streamlit_dashboard()

    assert "not installed" in result


@pytest.mark.asyncio
async def test_stop_streamlit_dashboard():
    """Test stopping a running dashboard."""
    mock_proc = AsyncMock()
    mock_proc.returncode = None
    mock_proc.pid = 88888

    tools_mod._streamlit_proc = mock_proc

    result = await stop_streamlit_dashboard()

    assert "stopped" in result.lower() or "Dashboard stopped" in result
    assert "88888" in result
    mock_proc.terminate.assert_called_once()
    assert tools_mod._streamlit_proc is None


@pytest.mark.asyncio
async def test_stop_streamlit_dashboard_not_running():
    """Test stopping when no dashboard is running."""
    result = await stop_streamlit_dashboard()
    assert "No dashboard" in result


# ── _stream_reader tests ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_stream_reader_reads_chunks():
    """Test _stream_reader accumulates chunks into the output list."""
    chunks = [b"hello ", b"world\n"]
    call_count = {"n": 0}

    async def fake_read(n):
        if call_count["n"] < len(chunks):
            data = chunks[call_count["n"]]
            call_count["n"] += 1
            return data
        return b""

    stream = MagicMock()
    stream.read = fake_read

    output: list[str] = []
    await _stream_reader(stream, output)

    assert len(output) == 2
    assert output[0] == "hello "
    assert output[1] == "world\n"


@pytest.mark.asyncio
async def test_stream_reader_handles_empty_stream():
    """Test _stream_reader handles an immediately-closed stream."""

    async def fake_read(n):
        return b""

    stream = MagicMock()
    stream.read = fake_read

    output: list[str] = []
    await _stream_reader(stream, output)

    assert output == []


# ── _run_benchmark early failure test ──────────────────────────────


@pytest.mark.asyncio
async def test_run_benchmark_early_failure():
    """Test _run_benchmark returns error when process fails immediately."""
    mock_proc = MagicMock()
    mock_proc.returncode = 1
    mock_proc.pid = 11111

    async def fake_read(n):
        return b""

    mock_proc.stdout = MagicMock()
    mock_proc.stdout.read = fake_read
    mock_proc.stderr = MagicMock()
    mock_proc.stderr.read = fake_read

    with (
        patch(
            "perfbench.tools.asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=mock_proc),
        ),
        patch(
            "perfbench.tools.asyncio.sleep",
            new=AsyncMock(),
        ),
    ):
        result = await run_vllm_benchmark(
            model="test-model",
            base_url="http://localhost:8000",
            served_model_name="test-model",
        )

    assert "Benchmark failed" in result
    # The benchmark should NOT be added to _benchmarks
    assert len(_benchmarks) == 0


# ── _stop_benchmark timeout path ───────────────────────────────────


@pytest.mark.asyncio
async def test_stop_benchmark_kill_on_timeout():
    """Test that stop falls back to kill() when terminate() times out."""
    mock_proc = AsyncMock()
    mock_proc.returncode = None
    mock_proc.pid = 99999

    entry = _BenchmarkEntry(proc=mock_proc)
    _benchmarks["timeout01"] = entry

    with patch(
        "perfbench.tools.asyncio.wait_for",
        side_effect=asyncio.TimeoutError,
    ):
        result = await stop_vllm_benchmark("timeout01")

    assert "terminated" in result
    assert "timeout01" not in _benchmarks
    mock_proc.terminate.assert_called_once()
    mock_proc.kill.assert_called_once()


# ── _stop_benchmark for already-finished process ───────────────────


@pytest.mark.asyncio
async def test_stop_benchmark_already_finished():
    """Test stopping a benchmark that has already finished."""
    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_proc.pid = 44444

    entry = _BenchmarkEntry(proc=mock_proc)
    _benchmarks["finished01"] = entry

    result = await stop_vllm_benchmark("finished01")

    assert "already finished" in result
    assert "finished01" not in _benchmarks


# ── run_streamlit_dashboard already running ────────────────────────


@pytest.mark.asyncio
async def test_run_streamlit_dashboard_already_running():
    """Test that starting dashboard when already running returns warning."""
    mock_proc = MagicMock()
    mock_proc.returncode = None
    mock_proc.pid = 77777

    tools_mod._streamlit_proc = mock_proc

    result = await run_streamlit_dashboard()

    assert "already running" in result


# ── Concurrent benchmarks ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_concurrent_benchmarks_independent():
    """Test multiple benchmarks can run with independent state."""
    mock_proc_a = MagicMock()
    mock_proc_a.returncode = None
    mock_proc_a.pid = 10001

    mock_proc_b = MagicMock()
    mock_proc_b.returncode = None
    mock_proc_b.pid = 10002

    entry_a = _BenchmarkEntry(proc=mock_proc_a)
    entry_a.output_lines = ["output_a_line1\n", "output_a_line2\n"]
    entry_a._read_cursor = 0

    entry_b = _BenchmarkEntry(proc=mock_proc_b)
    entry_b.output_lines = ["output_b_line1\n"]
    entry_b._read_cursor = 0

    _benchmarks["bench_a"] = entry_a
    _benchmarks["bench_b"] = entry_b

    result_a = await check_vllm_benchmark_status("bench_a")
    result_b = await check_aiperf_benchmark_status("bench_b")

    # Each returns its own output
    assert "output_a_line1" in result_a
    assert "output_a_line2" in result_a
    assert "output_b_line1" in result_b

    # Output from one does not leak into the other
    assert "output_b" not in result_a
    assert "output_a" not in result_b

    # Cursors advanced independently
    assert entry_a._read_cursor == 2
    assert entry_b._read_cursor == 1


# ── list_benchmarks tests ────────────────────────────────────────


def test_list_benchmarks_empty():
    """Test list_benchmarks when no benchmarks are running."""
    result = list_benchmarks()
    assert result == "No benchmarks are currently running."


def test_list_benchmarks_with_entries():
    """Test list_benchmarks shows running benchmarks with runner info."""
    mock_proc_a = MagicMock()
    mock_proc_a.returncode = None
    mock_proc_a.pid = 10001

    mock_proc_b = MagicMock()
    mock_proc_b.returncode = None
    mock_proc_b.pid = 10002

    _benchmarks["abc123"] = _BenchmarkEntry(proc=mock_proc_a, runner="vllm")
    _benchmarks["def456"] = _BenchmarkEntry(proc=mock_proc_b, runner="aiperf")

    result = list_benchmarks()
    assert "abc123" in result
    assert "vllm" in result
    assert "10001" in result
    assert "running" in result
    assert "def456" in result
    assert "aiperf" in result
    assert "10002" in result


def test_list_benchmarks_finished_entry():
    """Test list_benchmarks shows finished status for completed processes."""
    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_proc.pid = 30003

    _benchmarks["fin789"] = _BenchmarkEntry(proc=mock_proc, runner="guidellm")

    result = list_benchmarks()
    assert "fin789" in result
    assert "guidellm" in result
    assert "finished" in result


# ── list_results / read_result tests ─────────────────────────────


@pytest.fixture()
def fake_results(tmp_path):
    """Create a fake results tree and patch _RESULT_DIRS to use it."""
    vllm_dir = tmp_path / "results_vllm_bench"
    aiperf_dir = tmp_path / "results_aiperf"
    guidellm_dir = tmp_path / "results_guidellm"

    # vLLM: model dir with two JSON files
    model_v = vllm_dir / "granite_model"
    model_v.mkdir(parents=True)
    (model_v / "20260301_VLLM_curr=1_input=10_output=100.json").write_text(
        json.dumps({"request_throughput": 12.3})
    )
    (model_v / "20260302_VLLM_curr=10_input=256_output=128.json").write_text(
        json.dumps({"request_throughput": 45.7})
    )

    # AIPerf: model dir with one timestamped run
    run_a = aiperf_dir / "granite_model" / "20260301120000"
    run_a.mkdir(parents=True)
    (run_a / "profile_export_aiperf.json").write_text(
        json.dumps({"request_throughput": {"avg": 5.0}})
    )

    # GuideLLM: model dir with one timestamped run (full metrics structure)
    run_g = guidellm_dir / "granite_model" / "20260301130000"
    run_g.mkdir(parents=True)
    (run_g / "benchmarks.json").write_text(
        json.dumps(
            {
                "benchmarks": [
                    {
                        "config": {"strategy": {"type_": "synchronous"}},
                        "metrics": {
                            "requests_per_second": {"successful": {"mean": 8.5}},
                            "output_tokens_per_second": {"successful": {"mean": 900.0}},
                            "time_to_first_token_ms": {"successful": {"mean": 55.0}},
                        },
                    }
                ],
            }
        )
    )

    # Empty run dir (no benchmarks.json) — should be skipped
    (guidellm_dir / "granite_model" / "20260301140000").mkdir()

    # llama-bench: model dir with one JSON array file
    llamabench_dir = tmp_path / "results_llama_bench"
    model_lb = llamabench_dir / "my-model.Q4_K_M"
    model_lb.mkdir(parents=True)
    (model_lb / "20260301120000.json").write_text(
        json.dumps(
            [
                {
                    "model_type": "llama 8B Q4_K_M",
                    "model_size": 4912345678,
                    "n_gpu_layers": 99,
                    "n_batch": 2048,
                    "n_threads": 8,
                    "backends": "CUDA",
                    "n_prompt": 512,
                    "n_gen": 0,
                    "avg_ts": 1234.56,
                },
                {
                    "model_type": "llama 8B Q4_K_M",
                    "model_size": 4912345678,
                    "n_gpu_layers": 99,
                    "n_batch": 2048,
                    "n_threads": 8,
                    "backends": "CUDA",
                    "n_prompt": 0,
                    "n_gen": 128,
                    "avg_ts": 89.12,
                },
            ]
        )
    )

    # Ollama: model dir with one JSON file
    ollama_dir = tmp_path / "results_ollama_bench"
    model_ol = ollama_dir / "llama3.1_8b"
    model_ol.mkdir(parents=True)
    (model_ol / "20260401120000.json").write_text(
        json.dumps(
            {
                "model": "llama3.1:8b",
                "ollama_url": "http://localhost:11434",
                "category": "general",
                "num_prompts": 5,
                "num_iterations": 3,
                "aggregated": {
                    "avg_eval_rate": 42.5,
                    "stddev_eval_rate": 2.1,
                    "avg_prompt_eval_rate": 680.2,
                    "stddev_prompt_eval_rate": 15.3,
                    "avg_total_duration_ms": 5043.5,
                    "avg_load_duration_ms": 5.0,
                    "total_tokens_generated": 4350,
                    "total_prompt_tokens": 390,
                },
                "per_prompt": [],
            }
        )
    )

    patched = {
        "vllm": vllm_dir,
        "aiperf": aiperf_dir,
        "guidellm": guidellm_dir,
        "llamabench": llamabench_dir,
        "ollama": ollama_dir,
    }
    with (
        patch.object(tools_mod, "_RESULT_DIRS", patched),
        patch.object(tools_mod, "_RESULTS_ROOT", tmp_path),
    ):
        yield tmp_path


def test_list_results_all_runners(fake_results):
    """Test list_results returns results from all runners."""
    result = list_results()
    assert "vLLM Bench" in result
    assert "granite_model" in result
    assert "20260302_VLLM_curr=10_input=256_output=128" in result
    assert "AIPerf" in result
    assert "20260301120000" in result
    assert "GuideLLM" in result
    assert "20260301130000" in result
    # Empty run dir should not appear
    assert "20260301140000" not in result


def test_list_results_single_runner(fake_results):
    """Test list_results filters by runner."""
    result = list_results(runner="aiperf")
    assert "AIPerf" in result
    assert "vLLM" not in result
    assert "GuideLLM" not in result


def test_list_results_invalid_runner(fake_results):
    """Test list_results returns error for unknown runner."""
    result = list_results(runner="unknown")
    assert "Error" in result
    assert "unknown" in result


def test_list_results_empty_dirs(tmp_path):
    """Test list_results with empty result directories."""
    vllm_dir = tmp_path / "results_vllm_bench"
    vllm_dir.mkdir()
    patched = {
        "vllm": vllm_dir,
        "aiperf": tmp_path / "results_aiperf",
        "guidellm": tmp_path / "results_guidellm",
    }
    with (
        patch.object(tools_mod, "_RESULT_DIRS", patched),
        patch.object(tools_mod, "_RESULTS_ROOT", tmp_path),
    ):
        result = list_results()
    assert "No results found" in result


def test_read_result_vllm(fake_results):
    """Test read_result reads a vLLM result file."""
    result = read_result(
        runner="vllm",
        model="granite_model",
        run="20260301_VLLM_curr=1_input=10_output=100",
    )
    data = json.loads(result)
    assert data["request_throughput"] == 12.3


def test_read_result_aiperf(fake_results):
    """Test read_result reads an AIPerf result file."""
    result = read_result(runner="aiperf", model="granite_model", run="20260301120000")
    data = json.loads(result)
    assert data["request_throughput"]["avg"] == 5.0


def test_read_result_guidellm(fake_results):
    """Test read_result reads a GuideLLM result file."""
    result = read_result(runner="guidellm", model="granite_model", run="20260301130000")
    data = json.loads(result)
    assert "benchmarks" in data


def test_read_result_invalid_runner(fake_results):
    """Test read_result returns error for unknown runner."""
    result = read_result(runner="unknown", model="m", run="r")
    assert "Error" in result
    assert "unknown" in result


def test_read_result_missing_file(fake_results):
    """Test read_result returns error when file does not exist."""
    result = read_result(runner="vllm", model="granite_model", run="nonexistent")
    assert "Error" in result
    assert "not found" in result


def test_read_result_missing_model(fake_results):
    """Test read_result returns error when model dir does not exist."""
    result = read_result(runner="vllm", model="no_such_model", run="some_run")
    assert "Error" in result
    assert "not found" in result


# ── compare_results tests ────────────────────────────────────────


def test_compare_two_vllm_results(fake_results):
    """Test comparing two vLLM results shows throughput values."""
    result = compare_results(
        results=[
            {
                "runner": "vllm",
                "model": "granite_model",
                "run": "20260301_VLLM_curr=1_input=10_output=100",
            },
            {
                "runner": "vllm",
                "model": "granite_model",
                "run": "20260302_VLLM_curr=10_input=256_output=128",
            },
        ]
    )
    assert "Request throughput" in result
    assert "12.30" in result
    assert "45.70" in result


def test_compare_cross_runner(fake_results):
    """Test comparing results across vLLM and AIPerf runners."""
    result = compare_results(
        results=[
            {
                "runner": "vllm",
                "model": "granite_model",
                "run": "20260301_VLLM_curr=1_input=10_output=100",
            },
            {
                "runner": "aiperf",
                "model": "granite_model",
                "run": "20260301120000",
            },
        ]
    )
    assert "Request throughput" in result
    assert "12.30" in result
    assert "5.00" in result


def test_compare_with_guidellm(fake_results):
    """Test comparing GuideLLM result extracts metrics correctly."""
    result = compare_results(
        results=[
            {
                "runner": "guidellm",
                "model": "granite_model",
                "run": "20260301130000",
            },
        ]
    )
    assert "8.50" in result
    assert "900.00" in result
    assert "55.00" in result


def test_compare_custom_metrics(fake_results):
    """Test compare_results with a custom metrics filter."""
    result = compare_results(
        results=[
            {
                "runner": "vllm",
                "model": "granite_model",
                "run": "20260301_VLLM_curr=1_input=10_output=100",
            },
        ],
        metrics=["Request throughput (req/s)"],
    )
    assert "Request throughput" in result
    # Other metrics should not appear
    assert "Mean TTFT" not in result
    assert "Completed" not in result


def test_compare_invalid_metric_filter(fake_results):
    """Test compare_results with unrecognized metric names."""
    result = compare_results(
        results=[
            {
                "runner": "vllm",
                "model": "granite_model",
                "run": "20260301_VLLM_curr=1_input=10_output=100",
            },
        ],
        metrics=["nonexistent_metric"],
    )
    assert "Error" in result
    assert "not recognized" in result or "recognized" in result.lower()


def test_compare_empty_results():
    """Test compare_results with empty results list."""
    result = compare_results(results=[])
    assert "Error" in result


def test_compare_invalid_reference(fake_results):
    """Test compare_results with a bad result reference."""
    result = compare_results(
        results=[
            {"runner": "vllm", "model": "no_model", "run": "no_run"},
        ]
    )
    assert "Error" in result
    assert "not found" in result


# ── Logging tests ────────────────────────────────────────────────────


def test_tools_logger_name():
    """Test that the tools module logger has the expected name."""
    assert tools_mod.logger.name == "perfbench.tools"


def test_log_level_from_env():
    """Test that MCP_LOG_LEVEL env var configures the root logger level."""
    import importlib

    import perfbench.server as server_mod

    original_level = logging.getLogger().level
    try:
        with patch.dict("os.environ", {"MCP_LOG_LEVEL": "DEBUG"}):
            importlib.reload(server_mod)
            assert logging.getLogger().level == logging.DEBUG
    finally:
        # Restore original level
        logging.getLogger().setLevel(original_level)
        importlib.reload(server_mod)


@pytest.mark.asyncio
async def test_run_benchmark_logs_lifecycle(caplog):
    """Test that _run_benchmark emits log messages for launch and start."""
    mock_proc = MagicMock()
    mock_proc.returncode = None
    mock_proc.pid = 12345

    async def fake_read(n):
        return b""

    mock_proc.stdout = MagicMock()
    mock_proc.stdout.read = fake_read
    mock_proc.stderr = MagicMock()
    mock_proc.stderr.read = fake_read

    with (
        patch(
            "perfbench.tools.asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=mock_proc),
        ),
        patch(
            "perfbench.tools.asyncio.sleep",
            new=AsyncMock(),
        ),
        caplog.at_level(logging.INFO, logger="perfbench.tools"),
    ):
        await tools_mod._run_benchmark(
            cmd=["vllm", "bench", "serve"],
            install_hint="Install vllm",
            check_tool_name="check_vllm_benchmark_status",
            runner="vllm",
        )

    assert any("Launching vllm benchmark" in r.message for r in caplog.records)
    assert any("started (PID:" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_stop_benchmark_logs_termination(caplog):
    """Test that _stop_benchmark emits an INFO log on termination."""
    mock_proc = AsyncMock()
    mock_proc.returncode = None
    mock_proc.pid = 55555

    entry = _BenchmarkEntry(proc=mock_proc)
    _benchmarks["logtest01"] = entry

    with caplog.at_level(logging.INFO, logger="perfbench.tools"):
        await tools_mod._stop_benchmark("logtest01")

    assert any("Terminating benchmark logtest01" in r.message for r in caplog.records)


# ═══════════════════════════════════════════════════════════════════
#  Benchmark Presets
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_run_benchmark_preset_invalid():
    """Test that an invalid preset name returns an error listing valid presets."""
    result = await run_benchmark_preset(
        preset="nonexistent",
        model="my-model",
        base_url="http://localhost:8000",
    )
    assert "Unknown preset" in result
    assert "nonexistent" in result
    for name in _PRESETS:
        assert name in result


@pytest.mark.asyncio
async def test_run_benchmark_preset_quick():
    """Test that the 'quick' preset delegates to run_vllm_benchmark."""
    with patch(
        "perfbench.tools.run_vllm_benchmark",
        new=AsyncMock(return_value="Benchmark started (ID: abc12345)"),
    ) as mock_vllm:
        result = await run_benchmark_preset(
            preset="quick",
            model="ibm-granite/granite-4.0-micro",
            base_url="http://localhost:8000",
            served_model_name="granite-4.0-micro",
            api_token="tok-123",
        )

    assert "Benchmark started" in result
    mock_vllm.assert_called_once_with(
        model="ibm-granite/granite-4.0-micro",
        base_url="http://localhost:8000",
        served_model_name="granite-4.0-micro",
        api_token="tok-123",
        auth_header_name=None,
        num_prompts=10,
        max_concurrency=1,
        random_input_len=10,
        random_output_len=100,
    )


@pytest.mark.asyncio
async def test_run_benchmark_preset_throughput():
    """Test that the 'throughput' preset delegates to run_vllm_benchmark."""
    with patch(
        "perfbench.tools.run_vllm_benchmark",
        new=AsyncMock(return_value="Benchmark started (ID: def45678)"),
    ) as mock_vllm:
        result = await run_benchmark_preset(
            preset="throughput",
            model="my-model",
            base_url="http://localhost:8000",
        )

    assert "Benchmark started" in result
    mock_vllm.assert_called_once_with(
        model="my-model",
        base_url="http://localhost:8000",
        served_model_name="my-model",
        api_token=None,
        auth_header_name=None,
        num_prompts=100,
        max_concurrency=10,
        random_input_len=128,
        random_output_len=256,
    )


@pytest.mark.asyncio
async def test_run_benchmark_preset_latency():
    """Test that the 'latency' preset delegates to run_aiperf_benchmark."""
    with patch(
        "perfbench.tools.run_aiperf_benchmark",
        new=AsyncMock(return_value="Benchmark started (ID: lat12345)"),
    ) as mock_aiperf:
        result = await run_benchmark_preset(
            preset="latency",
            model="my-model",
            base_url="http://localhost:8000",
            api_token="key-abc",
        )

    assert "Benchmark started" in result
    mock_aiperf.assert_called_once_with(
        model="my-model",
        tokenizer="my-model",
        url="http://localhost:8000",
        api_key="key-abc",
        auth_header_name=None,
        streaming=True,
        concurrency=1,
        request_count=50,
    )


@pytest.mark.asyncio
async def test_run_benchmark_preset_stress():
    """Test that the 'stress' preset delegates to run_aiperf_benchmark."""
    with patch(
        "perfbench.tools.run_aiperf_benchmark",
        new=AsyncMock(return_value="Benchmark started (ID: str12345)"),
    ) as mock_aiperf:
        result = await run_benchmark_preset(
            preset="stress",
            model="my-model",
            base_url="http://localhost:8000",
        )

    assert "Benchmark started" in result
    mock_aiperf.assert_called_once_with(
        model="my-model",
        tokenizer="my-model",
        url="http://localhost:8000",
        api_key=None,
        auth_header_name=None,
        streaming=True,
        concurrency=50,
        request_count=500,
    )


@pytest.mark.asyncio
async def test_run_benchmark_preset_sweep():
    """Test that the 'sweep' preset delegates to run_guidellm_benchmark."""
    with patch(
        "perfbench.tools.run_guidellm_benchmark",
        new=AsyncMock(return_value="Benchmark started (ID: swp12345)"),
    ) as mock_guidellm:
        result = await run_benchmark_preset(
            preset="sweep",
            model="my-model",
            base_url="http://localhost:8000",
            api_token="key-xyz",
        )

    assert "Benchmark started" in result
    mock_guidellm.assert_called_once_with(
        target="http://localhost:8000",
        model="my-model",
        api_key="key-xyz",
        profile="sweep",
        max_requests=100,
        prompt_tokens=256,
        output_tokens=128,
    )


@pytest.mark.asyncio
async def test_run_benchmark_preset_full():
    """Test that the 'full' preset runs all three runners."""
    with (
        patch(
            "perfbench.tools.run_vllm_benchmark",
            new=AsyncMock(return_value="Benchmark started (ID: vllm0001)"),
        ) as mock_vllm,
        patch(
            "perfbench.tools.run_aiperf_benchmark",
            new=AsyncMock(return_value="Benchmark started (ID: aperf001)"),
        ) as mock_aiperf,
        patch(
            "perfbench.tools.run_guidellm_benchmark",
            new=AsyncMock(return_value="Benchmark started (ID: guide001)"),
        ) as mock_guidellm,
    ):
        result = await run_benchmark_preset(
            preset="full",
            model="ibm-granite/granite-4.0-micro",
            base_url="http://localhost:8000",
            served_model_name="granite-4.0-micro",
            api_token="tok-full",
        )

    assert "[vLLM" in result
    assert "[AIPerf" in result
    assert "[GuideLLM" in result
    assert "vllm0001" in result
    assert "aperf001" in result
    assert "guide001" in result

    mock_vllm.assert_called_once()
    mock_aiperf.assert_called_once()
    mock_guidellm.assert_called_once()


@pytest.mark.asyncio
async def test_run_benchmark_preset_served_model_defaults_to_model():
    """Test that served_model_name defaults to model when not provided."""
    with patch(
        "perfbench.tools.run_vllm_benchmark",
        new=AsyncMock(return_value="Benchmark started (ID: def00001)"),
    ) as mock_vllm:
        await run_benchmark_preset(
            preset="quick",
            model="ibm-granite/granite-4.0-micro",
            base_url="http://localhost:8000",
        )

    call_kwargs = mock_vllm.call_args.kwargs
    assert call_kwargs["served_model_name"] == "ibm-granite/granite-4.0-micro"
    assert call_kwargs["auth_header_name"] is None


@pytest.mark.asyncio
async def test_run_benchmark_preset_passes_auth_header_name():
    """Test that auth_header_name is forwarded to vLLM and AIPerf presets."""
    with patch(
        "perfbench.tools.run_vllm_benchmark",
        new=AsyncMock(return_value="Benchmark started (ID: hdr12345)"),
    ) as mock_vllm:
        await run_benchmark_preset(
            preset="quick",
            model="my-model",
            base_url="http://localhost:8000",
            api_token="tok-123",
            auth_header_name="CUSTOM_API_KEY_NAME",
        )

    assert mock_vllm.call_args.kwargs["auth_header_name"] == "CUSTOM_API_KEY_NAME"

    with patch(
        "perfbench.tools.run_aiperf_benchmark",
        new=AsyncMock(return_value="Benchmark started (ID: hdr12346)"),
    ) as mock_aiperf:
        await run_benchmark_preset(
            preset="latency",
            model="my-model",
            base_url="http://localhost:8000",
            api_token="tok-456",
            auth_header_name="X-Custom-Key",
        )

    assert mock_aiperf.call_args.kwargs["auth_header_name"] == "X-Custom-Key"


# ── Auth header command construction tests ───────────────────────────


@pytest.mark.asyncio
async def test_vllm_standard_auth_header():
    """Test vLLM uses Authorization Bearer when no auth_header_name."""
    mock_proc = MagicMock()
    mock_proc.returncode = None
    mock_proc.pid = 12345

    async def fake_read(n):
        return b""

    mock_proc.stdout = MagicMock()
    mock_proc.stdout.read = fake_read
    mock_proc.stderr = MagicMock()
    mock_proc.stderr.read = fake_read

    with (
        patch(
            "perfbench.tools.asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=mock_proc),
        ) as mock_exec,
        patch(
            "perfbench.tools.asyncio.sleep",
            new=AsyncMock(),
        ),
    ):
        await run_vllm_benchmark(
            model="test-model",
            base_url="http://localhost:8000",
            served_model_name="test-model",
            api_token="my-secret",
        )

    cmd = mock_exec.call_args.args
    assert "--header" in cmd
    header_idx = cmd.index("--header")
    assert cmd[header_idx + 1] == "Authorization=Bearer my-secret"


@pytest.mark.asyncio
async def test_vllm_custom_auth_header():
    """Test vLLM uses custom header when auth_header_name is provided."""
    mock_proc = MagicMock()
    mock_proc.returncode = None
    mock_proc.pid = 12345

    async def fake_read(n):
        return b""

    mock_proc.stdout = MagicMock()
    mock_proc.stdout.read = fake_read
    mock_proc.stderr = MagicMock()
    mock_proc.stderr.read = fake_read

    with (
        patch(
            "perfbench.tools.asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=mock_proc),
        ) as mock_exec,
        patch(
            "perfbench.tools.asyncio.sleep",
            new=AsyncMock(),
        ),
    ):
        await run_vllm_benchmark(
            model="test-model",
            base_url="http://localhost:8000",
            served_model_name="test-model",
            api_token="my-secret",
            auth_header_name="CUSTOM_API_KEY_NAME",
        )

    cmd = mock_exec.call_args.args
    assert "--header" in cmd
    header_idx = cmd.index("--header")
    assert cmd[header_idx + 1] == "CUSTOM_API_KEY_NAME=my-secret"


@pytest.mark.asyncio
async def test_aiperf_standard_auth():
    """Test AIPerf uses --api-key when no auth_header_name."""
    mock_proc = MagicMock()
    mock_proc.returncode = None
    mock_proc.pid = 22222

    async def fake_read(n):
        return b""

    mock_proc.stdout = MagicMock()
    mock_proc.stdout.read = fake_read
    mock_proc.stderr = MagicMock()
    mock_proc.stderr.read = fake_read

    with (
        patch(
            "perfbench.tools.asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=mock_proc),
        ) as mock_exec,
        patch(
            "perfbench.tools.asyncio.sleep",
            new=AsyncMock(),
        ),
    ):
        await run_aiperf_benchmark(
            model="test-model",
            tokenizer="test-model",
            url="http://localhost:8000",
            api_key="my-secret",
        )

    cmd = mock_exec.call_args.args
    assert "--api-key" in cmd
    assert "my-secret" in cmd
    assert "--header" not in cmd


@pytest.mark.asyncio
async def test_aiperf_custom_auth_header():
    """Test AIPerf uses custom header when auth_header_name is provided."""
    mock_proc = MagicMock()
    mock_proc.returncode = None
    mock_proc.pid = 22222

    async def fake_read(n):
        return b""

    mock_proc.stdout = MagicMock()
    mock_proc.stdout.read = fake_read
    mock_proc.stderr = MagicMock()
    mock_proc.stderr.read = fake_read

    with (
        patch(
            "perfbench.tools.asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=mock_proc),
        ) as mock_exec,
        patch(
            "perfbench.tools.asyncio.sleep",
            new=AsyncMock(),
        ),
    ):
        await run_aiperf_benchmark(
            model="test-model",
            tokenizer="test-model",
            url="http://localhost:8000",
            api_key="my-secret",
            auth_header_name="CUSTOM_API_KEY_NAME",
        )

    cmd = mock_exec.call_args.args
    assert "--header" in cmd
    header_idx = cmd.index("--header")
    assert cmd[header_idx + 1] == "CUSTOM_API_KEY_NAME:my-secret"
    assert "--api-key" not in cmd


# ── llama-bench tests ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_llama_bench_returns_id():
    """Test that run_llama_bench launches and returns an ID."""
    mock_proc = MagicMock()
    mock_proc.returncode = None
    mock_proc.pid = 55500

    async def fake_read(n):
        return b""

    mock_proc.stdout = MagicMock()
    mock_proc.stdout.read = fake_read
    mock_proc.stderr = MagicMock()
    mock_proc.stderr.read = fake_read

    with (
        patch(
            "perfbench.tools.asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=mock_proc),
        ),
        patch(
            "perfbench.tools.asyncio.sleep",
            new=AsyncMock(),
        ),
        patch(
            "perfbench.tools.pathlib.Path.is_file",
            return_value=True,
        ),
    ):
        result = await run_llama_bench(model_path="/models/my-model.Q4_K_M.gguf")

    assert "Benchmark started" in result
    assert "ID:" in result
    benchmark_id = result.split("ID: ")[1].split(")")[0]
    assert benchmark_id in _benchmarks


@pytest.mark.asyncio
async def test_run_llama_bench_model_not_found():
    """Test error message when model file does not exist."""
    result = await run_llama_bench(model_path="/nonexistent/model.gguf")
    assert "not found" in result
    assert "/nonexistent/model.gguf" in result


@pytest.mark.asyncio
async def test_run_llama_bench_not_installed():
    """Test error message when llama-bench is not installed."""
    with (
        patch(
            "perfbench.tools.asyncio.create_subprocess_exec",
            new=AsyncMock(side_effect=FileNotFoundError("No such file: 'llama-bench'")),
        ),
        patch(
            "perfbench.tools.pathlib.Path.is_file",
            return_value=True,
        ),
    ):
        result = await run_llama_bench(model_path="/models/model.gguf")

    assert "not installed" in result
    assert "llama.cpp" in result


@pytest.mark.asyncio
async def test_check_llama_bench_status_running():
    """Test llama-bench status check for a running benchmark."""
    mock_proc = MagicMock()
    mock_proc.returncode = None
    mock_proc.pid = 55501

    entry = _BenchmarkEntry(proc=mock_proc)
    entry.output_lines = ["| pp 512 | ...\n"]

    _benchmarks["llama01"] = entry
    result = await check_llama_bench_status("llama01")
    assert "still running" in result
    assert "pp 512" in result


@pytest.mark.asyncio
async def test_check_llama_bench_status_completed_saves_result(tmp_path):
    """Test that completed llama-bench saves stdout JSON to disk."""
    mock_proc = MagicMock()
    mock_proc.returncode = 0

    sample_data = [
        {"n_prompt": 512, "n_gen": 0, "avg_ts": 1000.0},
        {"n_prompt": 0, "n_gen": 128, "avg_ts": 80.0},
    ]

    entry = _BenchmarkEntry(proc=mock_proc)
    entry.output_lines = ["progress...\n"]
    entry.stdout_lines = [json.dumps(sample_data)]
    entry.result_dir = str(tmp_path / "results")
    entry.model_name = "test-model"

    _benchmarks["llama02"] = entry
    result = await check_llama_bench_status("llama02")

    assert "completed" in result
    assert "Results saved to:" in result
    assert "llama02" not in _benchmarks

    saved_files = list((tmp_path / "results").glob("*.json"))
    assert len(saved_files) == 1
    saved = json.loads(saved_files[0].read_text())
    assert len(saved) == 2
    assert saved[0]["avg_ts"] == 1000.0


@pytest.mark.asyncio
async def test_stop_llama_bench():
    """Test stopping a running llama-bench benchmark."""
    mock_proc = AsyncMock()
    mock_proc.returncode = None
    mock_proc.pid = 55502

    entry = _BenchmarkEntry(proc=mock_proc)
    _benchmarks["llama03"] = entry

    result = await stop_llama_bench("llama03")

    assert "terminated" in result
    assert "55502" in result
    assert "llama03" not in _benchmarks
    mock_proc.terminate.assert_called_once()


# ── _stream_reader extra parameter tests ─────────────────────────


@pytest.mark.asyncio
async def test_stream_reader_with_extra():
    """Test _stream_reader populates extra list when provided."""
    chunks = [b"hello ", b"world\n"]
    call_count = {"n": 0}

    async def fake_read(n):
        if call_count["n"] < len(chunks):
            data = chunks[call_count["n"]]
            call_count["n"] += 1
            return data
        return b""

    stream = MagicMock()
    stream.read = fake_read

    output: list[str] = []
    extra: list[str] = []
    await _stream_reader(stream, output, extra=extra)

    assert len(output) == 2
    assert len(extra) == 2
    assert extra[0] == "hello "
    assert extra[1] == "world\n"


@pytest.mark.asyncio
async def test_stream_reader_without_extra():
    """Test _stream_reader works without extra (backward compat)."""
    chunks = [b"data\n"]
    call_count = {"n": 0}

    async def fake_read(n):
        if call_count["n"] < len(chunks):
            data = chunks[call_count["n"]]
            call_count["n"] += 1
            return data
        return b""

    stream = MagicMock()
    stream.read = fake_read

    output: list[str] = []
    await _stream_reader(stream, output)

    assert len(output) == 1
    assert output[0] == "data\n"


# ── _save_stdout_result tests ────────────────────────────────────


def test_save_stdout_result_valid_json(tmp_path):
    """Test _save_stdout_result parses and saves valid JSON."""
    mock_proc = MagicMock()
    entry = _BenchmarkEntry(proc=mock_proc, runner="llamabench")
    entry.stdout_lines = ['[{"avg_ts": 100.0}]']
    entry.result_dir = str(tmp_path / "out")

    path = _save_stdout_result(entry)

    assert path is not None
    saved = json.loads(open(path).read())
    assert saved[0]["avg_ts"] == 100.0


def test_save_stdout_result_invalid_json(tmp_path):
    """Test _save_stdout_result returns None for invalid JSON."""
    mock_proc = MagicMock()
    entry = _BenchmarkEntry(proc=mock_proc, runner="llamabench")
    entry.stdout_lines = ["not valid json {{{"]
    entry.result_dir = str(tmp_path / "out")

    path = _save_stdout_result(entry)
    assert path is None


def test_save_stdout_result_empty_stdout(tmp_path):
    """Test _save_stdout_result returns None when stdout is empty."""
    mock_proc = MagicMock()
    entry = _BenchmarkEntry(proc=mock_proc, runner="llamabench")
    entry.stdout_lines = []
    entry.result_dir = str(tmp_path / "out")

    path = _save_stdout_result(entry)
    assert path is None


# ── llama-bench preset tests ────────────────────────────────────


@pytest.mark.asyncio
async def test_run_benchmark_preset_inference():
    """Test that the 'inference' preset delegates to run_llama_bench."""
    with patch(
        "perfbench.tools.run_llama_bench",
        new=AsyncMock(return_value="Benchmark started (ID: llb12345)"),
    ) as mock_llama:
        result = await run_benchmark_preset(
            preset="inference",
            model_path="/models/model.gguf",
        )

    assert "Benchmark started" in result
    mock_llama.assert_called_once_with(
        model_path="/models/model.gguf",
        n_prompt=512,
        n_gen=128,
        repetitions=5,
    )


@pytest.mark.asyncio
async def test_run_benchmark_preset_inference_missing_model_path():
    """Test that 'inference' preset requires model_path."""
    result = await run_benchmark_preset(
        preset="inference",
        model="my-model",
        base_url="http://localhost:8000",
    )
    assert "model_path" in result


# ── llama-bench list/read/compare tests ─────────────────────────


def test_list_results_llamabench(fake_results):
    """Test list_results includes llama-bench results."""
    result = list_results()
    assert "llama-bench" in result
    assert "my-model.Q4_K_M" in result
    assert "20260301120000" in result


def test_list_results_filter_llamabench(fake_results):
    """Test list_results filters to llamabench only."""
    result = list_results(runner="llamabench")
    assert "llama-bench" in result
    assert "vLLM" not in result
    assert "AIPerf" not in result


def test_read_result_llamabench(fake_results):
    """Test read_result reads a llama-bench result file."""
    result = read_result(
        runner="llamabench",
        model="my-model.Q4_K_M",
        run="20260301120000",
    )
    data = json.loads(result)
    assert isinstance(data, list)
    assert data[0]["avg_ts"] == 1234.56


def test_compare_with_llamabench(fake_results):
    """Test comparing llama-bench results extracts metrics correctly."""
    result = compare_results(
        results=[
            {
                "runner": "llamabench",
                "model": "my-model.Q4_K_M",
                "run": "20260301120000",
            },
        ]
    )
    assert "Prompt eval" in result
    assert "1,234.56" in result
    assert "Generation" in result
    assert "89.12" in result


# ── llama-bench in full preset with model_path ──────────────────


@pytest.mark.asyncio
async def test_run_benchmark_preset_full_with_model_path():
    """Test that 'full' preset includes llama-bench when model_path given."""
    with (
        patch(
            "perfbench.tools.run_vllm_benchmark",
            new=AsyncMock(return_value="Benchmark started (ID: vllm0001)"),
        ),
        patch(
            "perfbench.tools.run_aiperf_benchmark",
            new=AsyncMock(return_value="Benchmark started (ID: aperf001)"),
        ),
        patch(
            "perfbench.tools.run_guidellm_benchmark",
            new=AsyncMock(return_value="Benchmark started (ID: guide001)"),
        ),
        patch(
            "perfbench.tools.run_llama_bench",
            new=AsyncMock(return_value="Benchmark started (ID: llb00001)"),
        ) as mock_llama,
    ):
        result = await run_benchmark_preset(
            preset="full",
            model="my-model",
            base_url="http://localhost:8000",
            model_path="/models/model.gguf",
        )

    assert "[llama-bench" in result
    assert "llb00001" in result
    mock_llama.assert_called_once()


@pytest.mark.asyncio
async def test_run_benchmark_preset_serving_requires_model_base_url():
    """Test that serving presets require model and base_url."""
    result = await run_benchmark_preset(
        preset="quick",
        model_path="/models/model.gguf",
    )
    assert "model" in result.lower() and "base_url" in result.lower()


# ── ollama-bench tests ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_ollama_benchmark_returns_id():
    """Test that run_ollama_benchmark launches and returns an ID."""
    mock_proc = MagicMock()
    mock_proc.returncode = None
    mock_proc.pid = 66600

    async def fake_read(n):
        return b""

    mock_proc.stdout = MagicMock()
    mock_proc.stdout.read = fake_read
    mock_proc.stderr = MagicMock()
    mock_proc.stderr.read = fake_read

    with (
        patch(
            "perfbench.tools.asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=mock_proc),
        ),
        patch(
            "perfbench.tools.asyncio.sleep",
            new=AsyncMock(),
        ),
    ):
        result = await run_ollama_benchmark(model="llama3.1:8b")

    assert "Benchmark started" in result
    assert "ID:" in result
    benchmark_id = result.split("ID: ")[1].split(")")[0]
    assert benchmark_id in _benchmarks


@pytest.mark.asyncio
async def test_run_ollama_benchmark_not_installed():
    """Test error message when Python subprocess fails to start."""
    with patch(
        "perfbench.tools.asyncio.create_subprocess_exec",
        new=AsyncMock(
            side_effect=FileNotFoundError("No such file: 'python'")
        ),
    ):
        result = await run_ollama_benchmark(model="llama3.1:8b")

    assert "not installed" in result
    assert "ollama.com" in result


@pytest.mark.asyncio
async def test_check_ollama_benchmark_status_running():
    """Test ollama status check for a running benchmark."""
    mock_proc = MagicMock()
    mock_proc.returncode = None
    mock_proc.pid = 66601

    entry = _BenchmarkEntry(proc=mock_proc)
    entry.output_lines = ["[1/5] iter 1/3: eval=42.5 t/s\n"]

    _benchmarks["ollama01"] = entry
    result = await check_ollama_benchmark_status("ollama01")
    assert "still running" in result
    assert "eval=42.5" in result


@pytest.mark.asyncio
async def test_check_ollama_benchmark_status_completed_saves_result(tmp_path):
    """Test that completed ollama benchmark saves stdout JSON to disk."""
    mock_proc = MagicMock()
    mock_proc.returncode = 0

    sample_data = {
        "model": "llama3.1:8b",
        "aggregated": {"avg_eval_rate": 42.5, "avg_prompt_eval_rate": 680.2},
        "per_prompt": [],
    }

    entry = _BenchmarkEntry(proc=mock_proc)
    entry.output_lines = ["progress...\n"]
    entry.stdout_lines = [json.dumps(sample_data)]
    entry.result_dir = str(tmp_path / "results")
    entry.model_name = "llama3.1_8b"

    _benchmarks["ollama02"] = entry
    result = await check_ollama_benchmark_status("ollama02")

    assert "completed" in result
    assert "Results saved to:" in result
    assert "ollama02" not in _benchmarks

    saved_files = list((tmp_path / "results").glob("*.json"))
    assert len(saved_files) == 1
    saved = json.loads(saved_files[0].read_text())
    assert saved["aggregated"]["avg_eval_rate"] == 42.5


@pytest.mark.asyncio
async def test_stop_ollama_benchmark():
    """Test stopping a running ollama benchmark."""
    mock_proc = AsyncMock()
    mock_proc.returncode = None
    mock_proc.pid = 66602

    entry = _BenchmarkEntry(proc=mock_proc)
    _benchmarks["ollama03"] = entry

    result = await stop_ollama_benchmark("ollama03")

    assert "terminated" in result
    assert "66602" in result
    assert "ollama03" not in _benchmarks
    mock_proc.terminate.assert_called_once()


# ── ollama list/read/compare tests ─────────────────────────────────


def test_list_results_ollama(fake_results):
    """Test list_results includes ollama results."""
    result = list_results()
    assert "Ollama Bench" in result
    assert "llama3.1_8b" in result
    assert "20260401120000" in result


def test_list_results_filter_ollama(fake_results):
    """Test list_results filters to ollama only."""
    result = list_results(runner="ollama")
    assert "Ollama Bench" in result
    assert "vLLM" not in result
    assert "AIPerf" not in result


def test_read_result_ollama(fake_results):
    """Test read_result reads an ollama result file."""
    result = read_result(
        runner="ollama",
        model="llama3.1_8b",
        run="20260401120000",
    )
    data = json.loads(result)
    assert data["aggregated"]["avg_eval_rate"] == 42.5


def test_compare_with_ollama(fake_results):
    """Test comparing ollama results extracts metrics correctly."""
    result = compare_results(
        results=[
            {
                "runner": "ollama",
                "model": "llama3.1_8b",
                "run": "20260401120000",
            },
        ]
    )
    assert "Prompt eval" in result
    assert "680.20" in result
    assert "Generation" in result
    assert "42.50" in result


# ── ollama preset tests ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_benchmark_preset_ollama_quick():
    """Test that the 'ollama-quick' preset delegates to run_ollama_benchmark."""
    with patch(
        "perfbench.tools.run_ollama_benchmark",
        new=AsyncMock(return_value="Benchmark started (ID: oll12345)"),
    ) as mock_ollama:
        result = await run_benchmark_preset(
            preset="ollama-quick",
            model="llama3.1:8b",
        )

    assert "Benchmark started" in result
    mock_ollama.assert_called_once_with(
        model="llama3.1:8b",
        base_url="http://localhost:11434",
        num_iterations=3,
    )


@pytest.mark.asyncio
async def test_run_benchmark_preset_ollama_requires_model():
    """Test that 'ollama-quick' preset requires model parameter."""
    result = await run_benchmark_preset(
        preset="ollama-quick",
    )
    assert "model" in result.lower()


@pytest.mark.asyncio
async def test_run_benchmark_preset_full_with_ollama():
    """Test that 'full' preset includes ollama when ollama_model given."""
    with (
        patch(
            "perfbench.tools.run_vllm_benchmark",
            new=AsyncMock(return_value="Benchmark started (ID: vllm0001)"),
        ),
        patch(
            "perfbench.tools.run_aiperf_benchmark",
            new=AsyncMock(return_value="Benchmark started (ID: aperf001)"),
        ),
        patch(
            "perfbench.tools.run_guidellm_benchmark",
            new=AsyncMock(return_value="Benchmark started (ID: guide001)"),
        ),
        patch(
            "perfbench.tools.run_ollama_benchmark",
            new=AsyncMock(return_value="Benchmark started (ID: oll00001)"),
        ) as mock_ollama,
    ):
        result = await run_benchmark_preset(
            preset="full",
            model="my-model",
            base_url="http://localhost:8000",
            ollama_model="llama3.1:8b",
        )

    assert "[Ollama" in result
    assert "oll00001" in result
    mock_ollama.assert_called_once()


# ── result directory confinement tests ──────────────────────────


@pytest.mark.parametrize(
    "raw",
    [
        "../../../../tmp/pwned",
        "/tmp/pwned",
        "results_vllm_bench/../../../../tmp/pwned",
    ],
)
def test_safe_result_dir_rejects_escape(raw):
    """Traversal and absolute paths outside the root are rejected."""
    result = _safe_result_dir(raw, _RESULT_DIRS["vllm"], "model")
    assert isinstance(result, str)
    assert "invalid result directory" in result


def test_safe_result_dir_rejects_escape_in_parts():
    """A traversal hidden in a path component is rejected too."""
    result = _safe_result_dir("", _RESULT_DIRS["vllm"], "../../../../tmp")
    assert isinstance(result, str)
    assert "invalid result directory" in result


def test_safe_result_dir_accepts_relative_and_default():
    """Relative names stay under the root; empty falls back to default."""
    assert _safe_result_dir("results_vllm_bench", _RESULT_DIRS["vllm"], "m") == (
        _RESULT_DIRS["vllm"].resolve() / "m"
    )
    assert _safe_result_dir("", _RESULT_DIRS["ollama"], "m") == (
        _RESULT_DIRS["ollama"].resolve() / "m"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool", "kwargs", "dir_arg"),
    [
        (
            run_vllm_benchmark,
            {"model": "m", "base_url": "http://x", "served_model_name": "m"},
            "result_dir",
        ),
        (
            run_aiperf_benchmark,
            {"model": "m", "tokenizer": "t", "url": "http://x"},
            "artifact_dir",
        ),
        (run_guidellm_benchmark, {"target": "http://x"}, "output_dir"),
        (run_ollama_benchmark, {"model": "m"}, "result_dir"),
    ],
)
async def test_run_tools_reject_out_of_root_dir(tool, kwargs, dir_arg, tmp_path):
    """No subprocess is launched and nothing is created outside the root."""
    escape = tmp_path / "pwned"
    with patch(
        "perfbench.tools.asyncio.create_subprocess_exec", new=AsyncMock()
    ) as mock_exec:
        result = await tool(**kwargs, **{dir_arg: str(escape)})

    assert "invalid result directory" in result
    mock_exec.assert_not_called()
    assert not escape.exists()


@pytest.mark.asyncio
async def test_run_llama_bench_rejects_out_of_root_dir(tmp_path):
    """llama-bench rejects an out-of-root result_dir before launching."""
    model_file = tmp_path / "model.gguf"
    model_file.write_bytes(b"")
    escape = tmp_path / "pwned"

    with patch(
        "perfbench.tools.asyncio.create_subprocess_exec", new=AsyncMock()
    ) as mock_exec:
        result = await run_llama_bench(
            model_path=str(model_file), result_dir=str(escape)
        )

    assert "invalid result directory" in result
    mock_exec.assert_not_called()
    assert not escape.exists()
