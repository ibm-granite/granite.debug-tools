"""MCP tools for the perfbench server.

Register tools using the ``@mcp.tool()`` decorator. Tools let LLMs take
actions through the server — they are expected to perform computation and
may have side effects.
"""

import asyncio
import json as json_mod
import logging
import os
import pathlib
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime

from perfbench.dashboard_helpers import (
    fmt as _fmt,
)
from perfbench.dashboard_helpers import (
    guidellm_stat as _guidellm_stat,
)
from perfbench.dashboard_helpers import (
    metric_val as _metric_val,
)
from perfbench.dashboard_helpers import (
    split_pp_tg as _split_pp_tg,
)
from perfbench.server import mcp

logger = logging.getLogger(__name__)


@dataclass
class _BenchmarkEntry:
    """Tracks a running benchmark subprocess and its accumulated output."""

    proc: asyncio.subprocess.Process
    runner: str = ""  # vllm, aiperf, guidellm, llamabench, ollama
    output_lines: list[str] = field(default_factory=list)
    stdout_lines: list[str] = field(default_factory=list)
    _read_cursor: int = 0  # tracks what check_status already returned
    result_dir: str | None = None
    model_name: str | None = None


# Active benchmarks keyed by benchmark ID.
_benchmarks: dict[str, _BenchmarkEntry] = {}

# Result directories per runner.
_RESULTS_ROOT = pathlib.Path.cwd()
_RESULT_DIRS: dict[str, pathlib.Path] = {
    "vllm": _RESULTS_ROOT / "results_vllm_bench",
    "aiperf": _RESULTS_ROOT / "results_aiperf",
    "guidellm": _RESULTS_ROOT / "results_guidellm",
    "llamabench": _RESULTS_ROOT / "results_llama_bench",
    "ollama": _RESULTS_ROOT / "results_ollama_bench",
}


async def _stream_reader(
    stream: asyncio.StreamReader,
    output: list[str],
    extra: list[str] | None = None,
) -> None:
    """Continuously read chunks from *stream* into *output*.

    When *extra* is provided, decoded chunks are appended there too.
    This is used to capture stdout separately for runners that emit
    structured JSON on stdout (e.g. llama-bench).
    """
    while True:
        chunk = await stream.read(1024)
        if not chunk:
            break
        decoded = chunk.decode(errors="replace")
        output.append(decoded)
        if extra is not None:
            extra.append(decoded)


# ── Generic benchmark lifecycle helpers ─────────────────────────────


async def _run_benchmark(
    cmd: list[str],
    install_hint: str,
    check_tool_name: str,
    runner: str = "",
    env: dict[str, str] | None = None,
    *,
    result_dir: str | None = None,
    model_name: str | None = None,
) -> str:
    """Launch a benchmark subprocess and return an ID with early output.

    This is the shared implementation behind every ``run_*`` tool.
    """
    cmd_str = " ".join(cmd)
    logger.info("Launching %s benchmark: %s", runner, cmd_str)
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
    except FileNotFoundError:
        logger.warning("Command not found: %s", cmd[0])
        return (
            f"Error: '{cmd[0]}' is not installed or not found in PATH. {install_hint}"
        )

    entry = _BenchmarkEntry(
        proc=proc, runner=runner, result_dir=result_dir, model_name=model_name
    )

    # Start background readers that continuously accumulate output.
    # When result_dir is set, also capture stdout separately for JSON parsing.
    stdout_extra = entry.stdout_lines if result_dir else None
    asyncio.create_task(
        _stream_reader(proc.stdout, entry.output_lines, extra=stdout_extra)
    )
    asyncio.create_task(_stream_reader(proc.stderr, entry.output_lines))

    # Wait briefly for early output / fast failures.
    for _ in range(10):
        await asyncio.sleep(0.5)
        if proc.returncode is not None:
            break

    benchmark_id = uuid.uuid4().hex[:8]
    early = "".join(entry.output_lines).strip()

    if proc.returncode is not None and proc.returncode != 0:
        logger.warning(
            "Benchmark failed early (exit code %d): %s", proc.returncode, cmd_str
        )
        return (
            f"Benchmark failed (exit code {proc.returncode}).\n\n"
            f"Command: {cmd_str}\n\n"
            f"Output:\n{early}"
        )

    _benchmarks[benchmark_id] = entry
    logger.info("Benchmark %s started (PID: %d)", benchmark_id, proc.pid)
    entry._read_cursor = len(entry.output_lines)

    msg = (
        f"Benchmark started (ID: {benchmark_id}). "
        f"Use {check_tool_name} to poll for results.\n\n"
        f"Command: {cmd_str}"
    )
    if early:
        msg += f"\n\nInitial output:\n{early}"
    return msg


def _save_stdout_result(entry: _BenchmarkEntry) -> str | None:
    """Parse JSON from *stdout_lines* and save to *result_dir*.

    Returns the saved file path as a string, or ``None`` on failure.
    """
    if entry.result_dir is None:
        return None
    raw = "".join(entry.stdout_lines).strip()
    if not raw:
        return None
    try:
        data = json_mod.loads(raw)
    except json_mod.JSONDecodeError:
        logger.warning("Could not parse stdout as JSON for %s runner", entry.runner)
        return None

    result_path = pathlib.Path(entry.result_dir)
    result_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
    file_path = result_path / f"{timestamp}.json"
    file_path.write_text(json_mod.dumps(data, indent=2))
    logger.info("Saved %s result to %s", entry.runner, file_path)
    return str(file_path)


async def _check_benchmark_status(benchmark_id: str) -> str:
    """Check the status of a running benchmark (shared implementation)."""
    entry = _benchmarks.get(benchmark_id)
    if entry is None:
        logger.debug("Status check for unknown benchmark: %s", benchmark_id)
        active = list(_benchmarks.keys()) or ["none"]
        return (
            f"Error: benchmark '{benchmark_id}' not found. "
            f"Active benchmarks: {', '.join(active)}"
        )

    proc = entry.proc
    new_output = "".join(entry.output_lines[entry._read_cursor :]).strip()

    if proc.returncode is None:
        logger.debug(
            "Status check for %s: still running (PID: %d)", benchmark_id, proc.pid
        )
        entry._read_cursor = len(entry.output_lines)
        msg = f"Benchmark {benchmark_id} is still running (PID: {proc.pid})."
        if new_output:
            msg += f"\n\nOutput:\n{new_output}"
        return msg

    # Process has finished — save results if applicable, then clean up.
    saved_path = None
    if proc.returncode == 0 and entry.result_dir:
        saved_path = _save_stdout_result(entry)

    del _benchmarks[benchmark_id]

    if proc.returncode != 0:
        logger.warning(
            "Benchmark %s failed (exit code %d)", benchmark_id, proc.returncode
        )
        return (
            f"Benchmark {benchmark_id} failed "
            f"(exit code {proc.returncode}):\n{new_output}"
        )

    logger.info("Benchmark %s completed (exit code %d)", benchmark_id, proc.returncode)
    msg = f"Benchmark {benchmark_id} completed:\n{new_output}"
    if saved_path:
        msg += f"\n\nResults saved to: {saved_path}"
    return msg


async def _stop_benchmark(benchmark_id: str) -> str:
    """Terminate a running benchmark (shared implementation)."""
    entry = _benchmarks.get(benchmark_id)
    if entry is None:
        logger.debug("Stop request for unknown benchmark: %s", benchmark_id)
        active = list(_benchmarks.keys()) or ["none"]
        return (
            f"Error: benchmark '{benchmark_id}' not found. "
            f"Active benchmarks: {', '.join(active)}"
        )

    proc = entry.proc
    if proc.returncode is not None:
        logger.debug("Stop request for already-finished benchmark: %s", benchmark_id)
        del _benchmarks[benchmark_id]
        return (
            f"Benchmark {benchmark_id} has already finished "
            f"(exit code {proc.returncode})."
        )

    logger.info("Terminating benchmark %s (PID: %d)", benchmark_id, proc.pid)
    proc.terminate()
    try:
        await asyncio.wait_for(proc.wait(), timeout=10.0)
    except asyncio.TimeoutError:
        logger.warning(
            "Benchmark %s did not stop within 10s, sending SIGKILL (PID: %d)",
            benchmark_id,
            proc.pid,
        )
        proc.kill()
        await proc.wait()

    del _benchmarks[benchmark_id]
    return f"Benchmark {benchmark_id} terminated (PID: {proc.pid})."


@mcp.tool()
def ping() -> str:
    """Check if the server is alive and responsive."""
    return "pong"


@mcp.tool()
def list_benchmarks() -> str:
    """List all currently running benchmarks with their IDs, runners, and status."""
    if not _benchmarks:
        return "No benchmarks are currently running."
    lines = []
    for bid, entry in _benchmarks.items():
        status = "running" if entry.proc.returncode is None else "finished"
        lines.append(
            f"- {bid}: {entry.runner} (PID: {entry.proc.pid}, status: {status})"
        )
    return "\n".join(lines)


def _list_runner_results(runner: str, base_dir: pathlib.Path) -> list[str]:
    """Return formatted lines for a single runner's results."""
    if not base_dir.exists():
        return []
    model_dirs = sorted(d for d in base_dir.iterdir() if d.is_dir())
    lines: list[str] = []
    for model_dir in model_dirs:
        if runner in ("vllm", "llamabench", "ollama"):
            runs = sorted(
                (f.stem for f in model_dir.glob("*.json")),
                reverse=True,
            )
        elif runner == "aiperf":
            runs = sorted(
                (
                    d.name
                    for d in model_dir.iterdir()
                    if d.is_dir() and (d / "profile_export_aiperf.json").exists()
                ),
                reverse=True,
            )
        else:  # guidellm
            runs = sorted(
                (
                    d.name
                    for d in model_dir.iterdir()
                    if d.is_dir() and (d / "benchmarks.json").exists()
                ),
                reverse=True,
            )
        if runs:
            lines.append(f"  {model_dir.name}:")
            for run in runs:
                lines.append(f"    - {run}")
    return lines


@mcp.tool()
def list_results(runner: str | None = None) -> str:
    """List available benchmark result files.

    Scans result directories and returns a summary of saved results
    organized by runner and model.

    Args:
        runner: Filter by runner — "vllm", "aiperf", "guidellm",
            "llamabench", "ollama", or None for all.
    """
    if runner is not None and runner not in _RESULT_DIRS:
        valid = ", ".join(sorted(_RESULT_DIRS))
        return f"Error: unknown runner '{runner}'. Valid runners: {valid}"

    runners = [runner] if runner else list(_RESULT_DIRS)
    labels = {
        "vllm": "vLLM Bench",
        "aiperf": "AIPerf",
        "guidellm": "GuideLLM",
        "llamabench": "llama-bench",
        "ollama": "Ollama Bench",
    }
    sections: list[str] = []

    for name in runners:
        base_dir = _RESULT_DIRS[name]
        lines = _list_runner_results(name, base_dir)
        if lines:
            model_count = sum(1 for ln in lines if not ln.startswith("    -"))
            result_count = sum(1 for ln in lines if ln.startswith("    -"))
            header = (
                f"{labels[name]} ({model_count} model(s), {result_count} result(s)):"
            )
            sections.append(header + "\n" + "\n".join(lines))
        else:
            sections.append(f"No results found for {labels[name]}.")

    return "\n\n".join(sections)


def _resolve_result_path(runner: str, model: str, run: str) -> pathlib.Path | str:
    """Resolve the filesystem path for a result file.

    Returns a `pathlib.Path` on success or an error string on failure.
    """
    if runner not in _RESULT_DIRS:
        valid = ", ".join(sorted(_RESULT_DIRS))
        return f"Error: unknown runner '{runner}'. Valid runners: {valid}"

    base_dir = _RESULT_DIRS[runner]
    if runner in ("vllm", "llamabench", "ollama"):
        result_path = base_dir / model / f"{run}.json"
    elif runner == "aiperf":
        result_path = base_dir / model / run / "profile_export_aiperf.json"
    else:  # guidellm
        result_path = base_dir / model / run / "benchmarks.json"

    # Guard against path traversal.
    try:
        result_path.resolve().relative_to(base_dir.resolve())
    except ValueError:
        return "Error: invalid path — the result must be inside the results directory."

    if not result_path.exists():
        return f"Error: result not found at {result_path.relative_to(_RESULTS_ROOT)}"

    return result_path


@mcp.tool()
def read_result(runner: str, model: str, run: str) -> str:
    """Read a specific benchmark result file.

    Returns the JSON content of the result file, pretty-printed for
    readability.  Use ``list_results`` to discover available results.

    Args:
        runner: Runner name — "vllm", "aiperf", "guidellm", "llamabench",
            or "ollama".
        model: Model directory name (e.g. "ibm-granite_granite-4.0-h-tiny").
        run: Run identifier — for vLLM/llama-bench/ollama this is the
            filename stem, for AIPerf/GuideLLM this is the timestamp
            directory name.
    """
    resolved = _resolve_result_path(runner, model, run)
    if isinstance(resolved, str):
        return resolved
    data = json_mod.loads(resolved.read_text())
    return json_mod.dumps(data, indent=2)


# Normalized metrics for cross-runner comparison.
# Each tuple: (display_name, {runner: metric_key_or_None}).
_NORMALIZED_METRICS: list[tuple[str, dict[str, str | None]]] = [
    (
        "Request throughput (req/s)",
        {
            "vllm": "request_throughput",
            "aiperf": "request_throughput",
            "guidellm": "requests_per_second",
            "llamabench": None,
            "ollama": None,
        },
    ),
    (
        "Output throughput (tok/s)",
        {
            "vllm": "output_throughput",
            "aiperf": "output_token_throughput",
            "guidellm": "output_tokens_per_second",
            "llamabench": None,
            "ollama": None,
        },
    ),
    (
        "Mean TTFT (ms)",
        {
            "vllm": "mean_ttft_ms",
            "aiperf": "time_to_first_token",
            "guidellm": "time_to_first_token_ms",
            "llamabench": None,
            "ollama": None,
        },
    ),
    (
        "Mean ITL (ms)",
        {
            "vllm": None,
            "aiperf": "inter_token_latency",
            "guidellm": "inter_token_latency_ms",
            "llamabench": None,
            "ollama": None,
        },
    ),
    (
        "Mean latency (ms)",
        {
            "vllm": None,
            "aiperf": "request_latency",
            "guidellm": "request_latency",
            "llamabench": None,
            "ollama": None,
        },
    ),
    (
        "Completed requests",
        {
            "vllm": "completed",
            "aiperf": "request_count",
            "guidellm": None,
            "llamabench": None,
            "ollama": None,
        },
    ),
    (
        "Prompt eval (tok/s)",
        {
            "vllm": None,
            "aiperf": None,
            "guidellm": None,
            "llamabench": "avg_ts_pp",
            "ollama": "avg_prompt_eval_rate",
        },
    ),
    (
        "Generation (tok/s)",
        {
            "vllm": None,
            "aiperf": None,
            "guidellm": None,
            "llamabench": "avg_ts_tg",
            "ollama": "avg_eval_rate",
        },
    ),
]


def _extract_metric(runner: str, data: dict | list, metric_key: str | None) -> str:
    """Extract and format a single metric value from a result.

    *data* is usually a dict but may be a list for runners that store
    a JSON array (e.g. llamabench).
    """
    if metric_key is None:
        return "\u2014"
    if runner == "vllm":
        val = data.get(metric_key)
    elif runner == "aiperf":
        val = _metric_val(data, metric_key, "avg")
    elif runner == "llamabench":
        entries = data if isinstance(data, list) else []
        pp_vals, tg_vals = _split_pp_tg(entries)
        vals = pp_vals if metric_key == "avg_ts_pp" else tg_vals
        if not vals:
            return "—"
        val = sum(vals) / len(vals)
    elif runner == "ollama":
        agg = data.get("aggregated", {})
        val = agg.get(metric_key)
    else:  # guidellm
        benchmarks = data.get("benchmarks", [])
        if not benchmarks:
            return "\u2014"
        metrics = benchmarks[0].get("metrics", {})
        val = _guidellm_stat(metrics, metric_key, "mean")
    if val is None or val == "\u2014":
        return "\u2014"
    return str(_fmt(val))


@mcp.tool()
def compare_results(
    results: list[dict[str, str]],
    metrics: list[str] | None = None,
) -> str:
    """Compare metrics across multiple benchmark result files.

    Loads the specified results and returns a side-by-side markdown
    comparison table with normalized metric names.  Use ``list_results``
    to discover available results.

    Args:
        results: List of result references.  Each entry is a dict with
            keys ``runner``, ``model``, and ``run`` (same identifiers
            used by ``read_result``).
        metrics: Specific metric display names to include (e.g.
            ``["Request throughput (req/s)", "Mean TTFT (ms)"]``).
            If None, all normalized metrics are shown.
    """
    if not results:
        return "Error: provide at least one result to compare."

    # Load each result.
    loaded: list[tuple[str, str, dict]] = []  # (label, runner, data)
    for ref in results:
        be = ref.get("runner", "")
        mdl = ref.get("model", "")
        rn = ref.get("run", "")
        resolved = _resolve_result_path(be, mdl, rn)
        if isinstance(resolved, str):
            return resolved
        data = json_mod.loads(resolved.read_text())
        label = f"{mdl} / {rn} ({be})"
        loaded.append((label, be, data))

    # Determine which metrics to show.
    if metrics:
        rows = [(d, kd) for d, kd in _NORMALIZED_METRICS if d in metrics]
        if not rows:
            available = ", ".join(d for d, _ in _NORMALIZED_METRICS)
            return (
                "Error: none of the requested metrics "
                f"were recognized. Available: {available}"
            )
    else:
        rows = list(_NORMALIZED_METRICS)

    # Build the markdown table.
    col_headers = ["Metric"] + [label for label, _, _ in loaded]
    header_line = " | ".join(col_headers)
    sep_line = " | ".join("-" * max(len(h), 3) for h in col_headers)

    table_lines = [header_line, sep_line]
    for display, key_dict in rows:
        cells = [display]
        for _, be, data in loaded:
            cells.append(_extract_metric(be, data, key_dict.get(be)))
        table_lines.append(" | ".join(cells))

    return "\n".join(table_lines)


@mcp.tool()
async def run_vllm_benchmark(
    model: str,
    base_url: str,
    served_model_name: str,
    backend: str = "openai",
    endpoint: str = "/v1/completions",
    num_prompts: int = 10,
    dataset_name: str = "random",
    max_concurrency: int = 1,
    random_input_len: int = 10,
    random_output_len: int = 100,
    result_dir: str = "results_vllm_bench",
    ready_check_timeout_sec: int = 10,
    api_token: str | None = None,
    auth_header_name: str | None = None,
    request_rate: float | None = None,
) -> str:
    """Launch a vLLM benchmark against an LLM service.

    Starts ``vllm bench serve`` as a background subprocess and returns a
    benchmark ID along with the first 30 seconds of output.  Use
    ``check_vllm_benchmark_status`` to poll for new output and final
    results.  Requires ``vllm`` to be installed on the system.

    Args:
        model: Name of the model to benchmark.
        base_url: Base URL of the LLM service.
        served_model_name: Name of the served model.
        backend: Backend type — "openai", "openai-chat", or "vllm".
        endpoint: API endpoint path.
        num_prompts: Number of prompts to send.
        dataset_name: Dataset to use — "random", "sharegpt", etc.
        max_concurrency: Maximum number of concurrent requests.
        random_input_len: Input token length (random dataset).
        random_output_len: Output token length (random dataset).
        result_dir: Directory to save results in.
        ready_check_timeout_sec: Seconds to wait for server readiness.
        api_token: API authentication token. When ``auth_header_name``
            is omitted the token is sent as a standard
            ``Authorization: Bearer`` header. When ``auth_header_name``
            is provided it is sent as a custom header instead.
        auth_header_name: Custom header name for authentication (e.g.
            ``"CUSTOM_API_KEY_NAME"``). When ``None`` (default), uses standard
            ``Authorization: Bearer`` header.
        request_rate: Requests per second (omit for unlimited).
    """
    cmd: list[str] = [
        "vllm",
        "bench",
        "serve",
        "--model",
        model,
        "--base-url",
        base_url,
        "--served-model-name",
        served_model_name,
        "--backend",
        backend,
        "--endpoint",
        endpoint,
        "--num-prompts",
        str(num_prompts),
        "--dataset-name",
        dataset_name,
        "--max-concurrency",
        str(max_concurrency),
        "--random-input-len",
        str(random_input_len),
        "--random-output-len",
        str(random_output_len),
        "--result-dir",
        f"{result_dir}/{model.replace('/', '_')}",
        "--result-filename",
        f"{datetime.now().strftime('%Y%m%d%H%M%S')}_VLLM_curr={max_concurrency}_input={random_input_len}_output={random_output_len}.json",
        "--ready-check-timeout-sec",
        str(ready_check_timeout_sec),
        "--save-result",
    ]

    if api_token is not None:
        if auth_header_name:
            cmd.extend(["--header", f"{auth_header_name}={api_token}"])
        else:
            cmd.extend(["--header", f"Authorization=Bearer {api_token}"])
    if request_rate is not None:
        cmd.extend(["--request-rate", str(request_rate)])

    return await _run_benchmark(
        cmd,
        install_hint="Install it with: pip install vllm",
        check_tool_name="check_vllm_benchmark_status",
        runner="vllm",
    )


@mcp.tool()
async def check_vllm_benchmark_status(benchmark_id: str) -> str:
    """Check the status of a running vLLM benchmark.

    Returns any new output produced since the last check.

    Args:
        benchmark_id: The ID returned by ``run_vllm_benchmark``.
    """
    return await _check_benchmark_status(benchmark_id)


@mcp.tool()
async def stop_vllm_benchmark(benchmark_id: str) -> str:
    """Terminate a running vLLM benchmark.

    Args:
        benchmark_id: The ID returned by ``run_vllm_benchmark``.
    """
    return await _stop_benchmark(benchmark_id)


# ── aiperf tools ─────────────────────────────────────────────────────


@mcp.tool()
async def run_aiperf_benchmark(
    model: str,
    tokenizer: str,
    url: str,
    endpoint_type: str = "chat",
    streaming: bool = True,
    concurrency: int = 1,
    request_count: int = 10,
    request_rate: float | None = None,
    isl: int | None = None,
    osl: int | None = None,
    benchmark_duration: float | None = None,
    api_key: str | None = None,
    auth_header_name: str | None = None,
    artifact_dir: str = "results_aiperf",
    ui_type: str = "none",
    warmup_request_count: int | None = None,
) -> str:
    """Launch an aiperf benchmark against an LLM service.

    Starts ``aiperf profile`` as a background subprocess and returns a
    benchmark ID along with the first 30 seconds of output.  Use
    ``check_aiperf_benchmark_status`` to poll for new output and final
    results.  Requires ``aiperf`` to be installed on the system.

    Args:
        model: Name of the model to benchmark.
        tokenizer: Name of the tokenizer to use.
        url: Base URL of the LLM service (e.g. ``http://localhost:8000``).
        endpoint_type: API endpoint type — "chat", "completions",
            "embeddings", etc.
        streaming: Enable streaming responses for TTFT/ITL metrics.
        concurrency: Number of concurrent requests to maintain.
        request_count: Total number of requests to send.
        request_rate: Target requests per second (omit for concurrency
            mode).
        isl: Mean input sequence length in tokens (synthetic dataset).
        osl: Mean output sequence length in tokens.
        benchmark_duration: Maximum benchmark runtime in seconds.
        api_key: API authentication token. When ``auth_header_name``
            is omitted the token is sent via aiperf's native
            ``--api-key`` flag (standard ``Authorization: Bearer``).
            When ``auth_header_name`` is provided it is sent as a
            custom header instead.
        auth_header_name: Custom header name for authentication (e.g.
            ``"CUSTOM_API_KEY_NAME"``). When ``None`` (default), uses standard
            ``Authorization: Bearer`` header via ``--api-key``.
        artifact_dir: Directory to store benchmark artifacts.
        ui_type: UI display mode — "none", "simple", or "dashboard".
        warmup_request_count: Number of warmup requests before
            benchmarking.
    """
    cmd: list[str] = [
        "aiperf",
        "profile",
        "--model",
        model,
        "--tokenizer",
        tokenizer,
        "--url",
        url,
        "--endpoint-type",
        endpoint_type,
        "--concurrency",
        str(concurrency),
        "--request-count",
        str(request_count),
        "--artifact-dir",
        f"{artifact_dir}/{model.replace('/', '_')}"
        f"/{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "--ui-type",
        ui_type,
    ]

    if streaming:
        cmd.append("--streaming")
    if request_rate is not None:
        cmd.extend(["--request-rate", str(request_rate)])
    if isl is not None:
        cmd.extend(["--isl", str(isl)])
    if osl is not None:
        cmd.extend(["--osl", str(osl)])
    if benchmark_duration is not None:
        cmd.extend(["--benchmark-duration", str(benchmark_duration)])
    if api_key is not None:
        if auth_header_name:
            cmd.extend(["--header", f"{auth_header_name}:{api_key}"])
        else:
            cmd.extend(["--api-key", api_key])
    if warmup_request_count is not None:
        cmd.extend(["--warmup-request-count", str(warmup_request_count)])

    return await _run_benchmark(
        cmd,
        install_hint="Install it with: pip install aiperf",
        check_tool_name="check_aiperf_benchmark_status",
        runner="aiperf",
    )


@mcp.tool()
async def check_aiperf_benchmark_status(benchmark_id: str) -> str:
    """Check the status of a running aiperf benchmark.

    Returns any new output produced since the last check.

    Args:
        benchmark_id: The ID returned by ``run_aiperf_benchmark``.
    """
    return await _check_benchmark_status(benchmark_id)


@mcp.tool()
async def stop_aiperf_benchmark(benchmark_id: str) -> str:
    """Terminate a running aiperf benchmark.

    Args:
        benchmark_id: The ID returned by ``run_aiperf_benchmark``.
    """
    return await _stop_benchmark(benchmark_id)


# ── guidellm tools ──────────────────────────────────────────────────


@mcp.tool()
async def run_guidellm_benchmark(
    target: str,
    data: str | None = None,
    prompt_tokens: int = 256,
    output_tokens: int = 128,
    profile: str | None = None,
    rate: float | None = None,
    request_type: str | None = None,
    max_seconds: int | None = None,
    max_requests: int = 10,
    warmup: float | None = None,
    cooldown: float | None = None,
    max_errors: int | None = None,
    processor: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    output_dir: str | None = None,
    detect_saturation: bool = False,
) -> str:
    """Launch a GuideLLM benchmark against an LLM service.

    Starts ``guidellm benchmark`` as a background subprocess and returns
    a benchmark ID along with the first 30 seconds of output.  Use
    ``check_guidellm_benchmark_status`` to poll for new output and final
    results.  Requires ``guidellm`` to be installed on the system.

    Args:
        target: URL of the OpenAI-compatible endpoint
            (e.g. ``http://localhost:8000``).
        data: Data source — synthetic spec like
            ``prompt_tokens=256,output_tokens=128``, a HuggingFace
            dataset ID, or a local file path.  When ``None``, a
            synthetic spec is built from ``prompt_tokens`` and
            ``output_tokens``.
        prompt_tokens: Number of prompt tokens for synthetic data
            (used when ``data`` is ``None``).
        output_tokens: Number of output tokens for synthetic data
            (used when ``data`` is ``None``).
        profile: Load profile — ``synchronous``, ``concurrent``,
            ``throughput``, ``constant``, ``poisson``, or ``sweep``.
        rate: Numeric rate value — meaning depends on profile (concurrent
            users, requests/sec, or number of sweeps).
        request_type: API format — ``chat_completions``, ``text_completions``,
            ``audio_transcription``, or ``audio_translation``.
        max_seconds: Maximum duration in seconds per benchmark.
        max_requests: Maximum number of requests per benchmark
            (default 100).
        warmup: Warm-up specification (0-1 = percentage, ≥1 = absolute).
        cooldown: Cool-down specification (same format as warmup).
        max_errors: Maximum errors before stopping.
        processor: Tokenizer/processor name for synthetic data.
        model: Model name to pass in the generated requests.
        api_key: API authentication key — passed via
            ``--backend-kwargs`` as a Bearer token for all requests.
        output_dir: Directory for output files (json, csv, html).
        detect_saturation: Enable over-saturation detection.
    """

    if profile is None:
        profile = "sweep"
    if request_type is None:
        request_type = "chat_completions"
    if output_dir is None:
        output_dir = "results_guidellm"

    model_safe = model.replace("/", "_") if model else "unknown_model"
    run_output_dir = (
        f"{output_dir}/{model_safe}/{datetime.now().strftime('%Y%m%d%H%M%S')}"
    )
    # Pre-create the directory so guidellm's _resolve_path recognises
    # it as a directory and appends the default filename (benchmarks.json).
    pathlib.Path(run_output_dir).mkdir(parents=True, exist_ok=True)
    if data is None:
        data = f"prompt_tokens={prompt_tokens},output_tokens={output_tokens}"
    cmd: list[str] = [
        "guidellm",
        "benchmark",
        "--target",
        target,
        "--data",
        data,
        "--profile",
        profile,
        "--request-type",
        request_type,
        "--output-dir",
        run_output_dir,
        "--max-requests",
        str(max_requests),
    ]

    if rate is not None:
        cmd.extend(["--rate", str(rate)])
    if max_seconds is not None:
        cmd.extend(["--max-seconds", str(max_seconds)])
    if warmup is not None:
        cmd.extend(["--warmup", str(warmup)])
    if cooldown is not None:
        cmd.extend(["--cooldown", str(cooldown)])
    if max_errors is not None:
        cmd.extend(["--max-errors", str(max_errors)])
    if processor is not None:
        cmd.extend(["--processor", processor])
    if model is not None:
        cmd.extend(["--model", model])
    if detect_saturation:
        cmd.append("--detect-saturation")
    if api_key is not None:
        cmd.extend(
            [
                "--backend-kwargs",
                json_mod.dumps({"api_key": api_key}),
            ]
        )

    env = os.environ.copy()
    env["GUIDELLM_LOG_LEVEL"] = "DEBUG"

    return await _run_benchmark(
        cmd,
        install_hint="Install it with: pip install guidellm[recommended]",
        check_tool_name="check_guidellm_benchmark_status",
        runner="guidellm",
        env=env,
    )


@mcp.tool()
async def check_guidellm_benchmark_status(benchmark_id: str) -> str:
    """Check the status of a running GuideLLM benchmark.

    Returns any new output produced since the last check.

    Args:
        benchmark_id: The ID returned by ``run_guidellm_benchmark``.
    """
    return await _check_benchmark_status(benchmark_id)


@mcp.tool()
async def stop_guidellm_benchmark(benchmark_id: str) -> str:
    """Terminate a running GuideLLM benchmark.

    Args:
        benchmark_id: The ID returned by ``run_guidellm_benchmark``.
    """
    return await _stop_benchmark(benchmark_id)


# ── llama-bench tools ──────────────────────────────────────────────


@mcp.tool()
async def run_llama_bench(
    model_path: str,
    n_prompt: int = 512,
    n_gen: int = 128,
    n_gpu_layers: int = 99,
    batch_size: int = 2048,
    ubatch_size: int = 512,
    threads: int | None = None,
    flash_attn: bool = False,
    cache_type_k: str = "f16",
    cache_type_v: str = "f16",
    repetitions: int = 5,
    n_depth: int = 0,
    split_mode: str = "layer",
    use_mmap: bool = True,
    result_dir: str = "",
) -> str:
    """Launch a llama-bench benchmark for raw local inference performance.

    Loads a GGUF model file directly and measures prompt processing (pp)
    and text generation (tg) throughput in tokens/second.  No server
    required.  Starts ``llama-bench`` as a background subprocess and
    returns a benchmark ID.  Use ``check_llama_bench_status`` to poll
    for results.

    Args:
        model_path: Path to a GGUF model file.
        n_prompt: Number of prompt tokens to process (-p).
        n_gen: Number of tokens to generate (-n).
        n_gpu_layers: Layers offloaded to GPU (-ngl). Use 99 for all.
        batch_size: Batch size for prompt processing (-b).
        ubatch_size: Micro-batch size (-ub).
        threads: CPU threads (-t).  Omit for system default.
        flash_attn: Enable flash attention (-fa).
        cache_type_k: KV cache key type (-ctk): f16, q8_0, q4_0, etc.
        cache_type_v: KV cache value type (-ctv).
        repetitions: Number of test repetitions (-r).
        n_depth: Context depth for prefill (-d). 0 uses n_prompt.
        split_mode: Multi-GPU split mode (-sm): none, layer, row.
        use_mmap: Use memory-mapped model loading (-mmp).
        result_dir: Base directory for saving results.  Defaults to
            the project's ``results_llama_bench`` directory.
    """
    model_file = pathlib.Path(model_path)
    if not model_file.is_file():
        return f"Error: model file not found: {model_path}"

    model_name = model_file.stem
    if not result_dir:
        result_dir = str(_RESULT_DIRS["llamabench"])

    cmd: list[str] = [
        "llama-bench",
        "-m",
        model_path,
        "-p",
        str(n_prompt),
        "-n",
        str(n_gen),
        "-ngl",
        str(n_gpu_layers),
        "-b",
        str(batch_size),
        "-ub",
        str(ubatch_size),
        "-fa",
        "1" if flash_attn else "0",
        "-ctk",
        cache_type_k,
        "-ctv",
        cache_type_v,
        "-r",
        str(repetitions),
        "-d",
        str(n_depth),
        "-sm",
        split_mode,
        "-mmp",
        "1" if use_mmap else "0",
        "-o",
        "json",
        "-oe",
        "md",
        "--progress",
    ]

    if threads is not None:
        cmd.extend(["-t", str(threads)])

    return await _run_benchmark(
        cmd,
        install_hint=(
            "Build llama.cpp and ensure llama-bench is in your PATH. "
            "See: https://github.com/ggml-org/llama.cpp#build"
        ),
        check_tool_name="check_llama_bench_status",
        runner="llamabench",
        result_dir=f"{result_dir}/{model_name}",
        model_name=model_name,
    )


@mcp.tool()
async def check_llama_bench_status(benchmark_id: str) -> str:
    """Check the status of a running llama-bench benchmark.

    Returns any new output produced since the last check.

    Args:
        benchmark_id: The ID returned by ``run_llama_bench``.
    """
    return await _check_benchmark_status(benchmark_id)


@mcp.tool()
async def stop_llama_bench(benchmark_id: str) -> str:
    """Terminate a running llama-bench benchmark.

    Args:
        benchmark_id: The ID returned by ``run_llama_bench``.
    """
    return await _stop_benchmark(benchmark_id)


# ── ollama-bench tools ─────────────────────────────────────────────


@mcp.tool()
async def run_ollama_benchmark(
    model: str,
    base_url: str = "http://localhost:11434",
    prompts: list[str] | None = None,
    num_iterations: int = 3,
    category: str = "general",
    result_dir: str = "",
) -> str:
    """Launch an Ollama benchmark for local inference performance.

    Sends prompts to an Ollama instance via its REST API and measures
    generation speed, prompt processing speed, and timing.  Starts a
    benchmark subprocess and returns a benchmark ID.  Use
    ``check_ollama_benchmark_status`` to poll for results.

    Args:
        model: Ollama model tag as shown by ``ollama list`` — use the
            ``name:tag`` format (e.g. ``"granite3.3:8b"``,
            ``"granite4:1b"``).  Do **not** use HuggingFace repo IDs
            (``org/model``).
        base_url: Ollama server URL.
        prompts: Custom prompts to benchmark.  ``None`` uses a built-in
            set of 5 diverse prompts.
        num_iterations: Number of times to repeat each prompt.
        category: Label for this benchmark category.
        result_dir: Base directory for saving results.  Defaults to
            the project's ``results_ollama_bench`` directory.
    """
    if num_iterations < 1:
        return "Error: num_iterations must be at least 1."

    model_safe = model.replace("/", "_").replace(":", "_")
    if not result_dir:
        result_dir = str(_RESULT_DIRS["ollama"])

    cmd: list[str] = [
        sys.executable,
        "-m",
        "perfbench._ollama_bench_runner",
        "--model",
        model,
        "--base-url",
        base_url,
        "--prompts",
        json_mod.dumps(prompts or []),
        "--num-iterations",
        str(num_iterations),
        "--category",
        category,
    ]

    return await _run_benchmark(
        cmd,
        install_hint=(
            "Install Ollama from https://ollama.com and ensure it is "
            "running (ollama serve)."
        ),
        check_tool_name="check_ollama_benchmark_status",
        runner="ollama",
        result_dir=f"{result_dir}/{model_safe}",
        model_name=model_safe,
    )


@mcp.tool()
async def check_ollama_benchmark_status(benchmark_id: str) -> str:
    """Check the status of a running Ollama benchmark.

    Returns any new output produced since the last check.

    Args:
        benchmark_id: The ID returned by ``run_ollama_benchmark``.
    """
    return await _check_benchmark_status(benchmark_id)


@mcp.tool()
async def stop_ollama_benchmark(benchmark_id: str) -> str:
    """Terminate a running Ollama benchmark.

    Args:
        benchmark_id: The ID returned by ``run_ollama_benchmark``.
    """
    return await _stop_benchmark(benchmark_id)


# ═══════════════════════════════════════════════════════════════════
#  Benchmark Presets
# ═══════════════════════════════════════════════════════════════════

_PRESETS: dict[str, dict] = {
    "quick": {
        "runner": "vllm",
        "description": "Fast smoke test — 10 prompts, concurrency=1, short I/O",
        "kwargs": {
            "num_prompts": 10,
            "max_concurrency": 1,
            "random_input_len": 10,
            "random_output_len": 100,
        },
    },
    "throughput": {
        "runner": "vllm",
        "description": (
            "Max throughput — 100 prompts, concurrency=10, larger payloads"
        ),
        "kwargs": {
            "num_prompts": 100,
            "max_concurrency": 10,
            "random_input_len": 128,
            "random_output_len": 256,
        },
    },
    "latency": {
        "runner": "aiperf",
        "description": "Latency profile — streaming, concurrency=1, 50 requests",
        "kwargs": {
            "streaming": True,
            "concurrency": 1,
            "request_count": 50,
        },
    },
    "stress": {
        "runner": "aiperf",
        "description": ("High-load stress test — concurrency=50, 500 requests"),
        "kwargs": {
            "streaming": True,
            "concurrency": 50,
            "request_count": 500,
        },
    },
    "sweep": {
        "runner": "guidellm",
        "description": (
            "GuideLLM load sweep — sweep profile, 100 requests, synthetic data"
        ),
        "kwargs": {
            "profile": "sweep",
            "max_requests": 100,
            "prompt_tokens": 256,
            "output_tokens": 128,
        },
    },
    "inference": {
        "runner": "llamabench",
        "description": ("Raw inference benchmark — llama-bench with pp 512 + tg 128"),
        "kwargs": {
            "n_prompt": 512,
            "n_gen": 128,
            "repetitions": 5,
        },
    },
    "ollama-quick": {
        "runner": "ollama",
        "description": "Quick Ollama benchmark — 5 prompts, 3 iterations",
        "kwargs": {
            "num_iterations": 3,
        },
    },
    "full": {
        "runner": "all",
        "description": (
            "Complete profile — vLLM (quick) + AIPerf (latency) + GuideLLM (sweep)"
            " + llama-bench (inference, if model_path provided)"
            " + Ollama (quick, if ollama_model provided)"
        ),
    },
}


@mcp.tool()
async def run_benchmark_preset(
    preset: str,
    model: str = "",
    base_url: str = "",
    served_model_name: str | None = None,
    api_token: str | None = None,
    auth_header_name: str | None = None,
    model_path: str | None = None,
    ollama_model: str | None = None,
    ollama_url: str = "http://localhost:11434",
) -> str:
    """Run a benchmark using a predefined configuration preset.

    Simplifies benchmarking by encapsulating common parameter combinations
    behind a single preset name.  Each preset delegates to the appropriate
    ``run_*`` tool(s) with tuned defaults.

    Available presets:
      - ``quick``        — vLLM smoke test (10 prompts, concurrency=1)
      - ``throughput``   — vLLM max throughput (100 prompts, concurrency=10)
      - ``latency``      — AIPerf latency profile (streaming, concurrency=1)
      - ``stress``       — AIPerf stress test (concurrency=50, 500 requests)
      - ``sweep``        — GuideLLM load sweep (sweep profile, 100 requests)
      - ``inference``    — llama-bench raw inference (pp 512 + tg 128)
      - ``ollama-quick`` — Ollama quick benchmark (5 prompts, 3 iterations)
      - ``full``         — All runners (quick + latency + sweep + inference
        + ollama)

    Args:
        preset: Preset name — "quick", "throughput", "latency", "stress",
            "sweep", "inference", "ollama-quick", or "full".
        model: Model identifier (e.g. "ibm-granite/granite-4.0-micro").
            Required for serving presets (quick, throughput, latency,
            stress, sweep, full).  For ``ollama-quick``, pass the Ollama
            tag here (e.g. ``"granite3.3:8b"``).
        base_url: URL of the LLM service (e.g. "http://localhost:8000").
            Required for serving presets.
        served_model_name: Served model name (defaults to *model*).
        api_token: API authentication token.
        auth_header_name: Custom header name for authentication (e.g.
            ``"CUSTOM_API_KEY_NAME"``). When ``None`` (default), uses standard
            ``Authorization: Bearer`` header. Applies to vLLM and AIPerf
            presets only.
        model_path: Path to a GGUF model file.  Required for the
            ``inference`` preset and optional for ``full``.
        ollama_model: Ollama model tag (e.g. "llama3.1:8b").  Only used
            by the ``full`` preset to include an Ollama run.  For
            ``ollama-quick``, pass the tag via *model* instead.
        ollama_url: Ollama server URL (default ``http://localhost:11434``).
    """
    if preset not in _PRESETS:
        valid = ", ".join(sorted(_PRESETS))
        logger.warning("Unknown preset %r requested", preset)
        return f"Unknown preset '{preset}'. Valid presets: {valid}."

    if served_model_name is None:
        served_model_name = model

    preset_cfg = _PRESETS[preset]
    runner = preset_cfg["runner"]

    if runner in ("vllm", "aiperf", "guidellm", "all") and not (model and base_url):
        return f"Error: preset '{preset}' requires 'model' and 'base_url' parameters."
    if runner == "llamabench" and not model_path:
        return (
            f"Error: preset '{preset}' requires 'model_path' parameter "
            f"(path to a GGUF file)."
        )
    if runner == "ollama" and not model:
        return (
            f"Error: preset '{preset}' requires 'model' parameter "
            f"(Ollama model name, e.g. 'llama3.1:8b')."
        )

    logger.info(
        "Running preset %r (runner=%s) for model=%s",
        preset,
        runner,
        model or model_path,
    )

    if runner == "llamabench":
        return await run_llama_bench(
            model_path=model_path,
            **preset_cfg["kwargs"],
        )

    if runner == "ollama":
        return await run_ollama_benchmark(
            model=model,
            base_url=ollama_url,
            **preset_cfg["kwargs"],
        )

    if runner == "vllm":
        return await run_vllm_benchmark(
            model=model,
            base_url=base_url,
            served_model_name=served_model_name,
            api_token=api_token,
            auth_header_name=auth_header_name,
            **preset_cfg["kwargs"],
        )

    if runner == "aiperf":
        return await run_aiperf_benchmark(
            model=model,
            tokenizer=model,
            url=base_url,
            api_key=api_token,
            auth_header_name=auth_header_name,
            **preset_cfg["kwargs"],
        )

    if runner == "guidellm":
        return await run_guidellm_benchmark(
            target=base_url,
            model=model,
            api_key=api_token,
            **preset_cfg["kwargs"],
        )

    # "full" preset — run all three sequentially.
    results: list[str] = []

    vllm_result = await run_vllm_benchmark(
        model=model,
        base_url=base_url,
        served_model_name=served_model_name,
        api_token=api_token,
        auth_header_name=auth_header_name,
        **_PRESETS["quick"]["kwargs"],
    )
    results.append(f"[vLLM — quick]\n{vllm_result}")

    aiperf_result = await run_aiperf_benchmark(
        model=model,
        tokenizer=model,
        url=base_url,
        api_key=api_token,
        auth_header_name=auth_header_name,
        **_PRESETS["latency"]["kwargs"],
    )
    results.append(f"[AIPerf — latency]\n{aiperf_result}")

    guidellm_result = await run_guidellm_benchmark(
        target=base_url,
        model=model,
        api_key=api_token,
        **_PRESETS["sweep"]["kwargs"],
    )
    results.append(f"[GuideLLM — sweep]\n{guidellm_result}")

    if model_path:
        llama_result = await run_llama_bench(
            model_path=model_path,
            **_PRESETS["inference"]["kwargs"],
        )
        results.append(f"[llama-bench — inference]\n{llama_result}")

    if ollama_model:
        ollama_result = await run_ollama_benchmark(
            model=ollama_model,
            base_url=ollama_url,
            **_PRESETS["ollama-quick"]["kwargs"],
        )
        results.append(f"[Ollama — quick]\n{ollama_result}")

    return "\n\n".join(results)


# ═══════════════════════════════════════════════════════════════════
#  Streamlit Dashboard
# ═══════════════════════════════════════════════════════════════════

_streamlit_proc: asyncio.subprocess.Process | None = None


@mcp.tool()
async def run_streamlit_dashboard(port: int = 8501) -> str:
    """Start the Streamlit results dashboard.

    Launches ``streamlit run streamlit_app.py`` as a background process
    and returns the URL where the dashboard is accessible.

    Args:
        port: Port number for the Streamlit server (default 8501).
    """
    global _streamlit_proc

    if _streamlit_proc is not None and _streamlit_proc.returncode is None:
        logger.debug("Dashboard already running (PID: %d)", _streamlit_proc.pid)
        return (
            f"Dashboard is already running (PID: {_streamlit_proc.pid}). "
            f"Visit http://localhost:{port} or stop it first."
        )

    # Resolve the project root (where streamlit_app.py lives).
    project_root = pathlib.Path(__file__).resolve().parent.parent.parent
    app_path = project_root / "streamlit_app.py"

    if not app_path.exists():
        return f"Error: streamlit_app.py not found at {app_path}"

    cmd = [
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        str(port),
        "--server.headless",
        "true",
    ]

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(project_root),
        )
    except FileNotFoundError:
        logger.warning("Streamlit not found in PATH")
        return (
            "Error: 'streamlit' is not installed or not found in PATH. "
            "Install it with: pip install streamlit"
        )

    _streamlit_proc = proc
    logger.info("Streamlit dashboard started (PID: %d, port: %d)", proc.pid, port)

    # Give Streamlit a moment to start and capture initial output.
    output_lines: list[str] = []
    try:

        async def _read_lines():
            assert proc.stderr is not None
            while True:
                line = await proc.stderr.readline()
                if not line:
                    break
                output_lines.append(line.decode(errors="replace").rstrip())

        await asyncio.wait_for(_read_lines(), timeout=5.0)
    except asyncio.TimeoutError:
        pass  # Expected — Streamlit keeps running.

    startup_output = "\n".join(output_lines) if output_lines else ""
    return (
        f"Dashboard started (PID: {proc.pid}).\n"
        f"URL: http://localhost:{port}\n\n"
        f"{startup_output}"
    )


@mcp.tool()
async def stop_streamlit_dashboard() -> str:
    """Stop the running Streamlit results dashboard."""
    global _streamlit_proc

    if _streamlit_proc is None:
        logger.debug("No dashboard to stop")
        return "No dashboard is currently running."

    if _streamlit_proc.returncode is not None:
        pid = _streamlit_proc.pid
        _streamlit_proc = None
        return f"Dashboard process (PID: {pid}) has already exited."

    pid = _streamlit_proc.pid
    _streamlit_proc.terminate()
    try:
        await asyncio.wait_for(_streamlit_proc.wait(), timeout=10.0)
    except asyncio.TimeoutError:
        _streamlit_proc.kill()
        await _streamlit_proc.wait()

    _streamlit_proc = None
    logger.info("Streamlit dashboard stopped (PID: %d)", pid)
    return f"Dashboard stopped (PID: {pid})."
