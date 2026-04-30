"""MCP prompts for the perfbench server.

Register prompts using the `@mcp.prompt()` decorator. Prompts define reusable
interaction patterns — templates that guide how LLMs should approach specific tasks.
"""

from perfbench.server import mcp


@mcp.prompt()
def benchmark_summary(model_name: str) -> str:
    """Generate a prompt that asks for a benchmark summary of a given model."""
    return (
        f"Please provide a comprehensive benchmark summary for the model "
        f"'{model_name}'. Include key performance metrics, strengths, "
        f"weaknesses, and comparison with similar models."
    )


@mcp.prompt()
def quick_benchmark(model: str, base_url: str) -> str:
    """Guide the agent through a quick benchmark of a model."""
    return (
        f"Run a quick benchmark of '{model}' at {base_url}. "
        f"Steps: "
        f"1) Use the ping tool to verify the server is alive. "
        f"2) Use run_vllm_benchmark with 10 prompts and concurrency=1 "
        f"to run a fast smoke test. "
        f"3) Poll with check_vllm_benchmark_status every 30 seconds "
        f"until the benchmark completes. "
        f"4) Summarize the key metrics: request throughput (req/s), "
        f"mean latency, and time to first token (TTFT)."
    )


@mcp.prompt()
def full_benchmark_suite(model: str, base_url: str) -> str:
    """Guide the agent through a comprehensive benchmark suite."""
    return (
        f"Run a comprehensive benchmark suite for '{model}' at {base_url}. "
        f"Steps: "
        f"1) Use run_vllm_benchmark with 100 prompts at concurrency=1, "
        f"then again at concurrency=10 to measure throughput scaling. "
        f"Poll with check_vllm_benchmark_status until both runs complete. "
        f"2) Use run_aiperf_benchmark with streaming enabled and 100 "
        f"requests to capture detailed latency metrics (TTFT, ITL). "
        f"Poll with check_aiperf_benchmark_status until complete. "
        f"3) Use run_guidellm_benchmark with a sweep profile to test "
        f"across multiple load levels. "
        f"Poll with check_guidellm_benchmark_status until complete. "
        f"4) Use run_streamlit_dashboard to start the results dashboard. "
        f"5) Use compare_results to compare metrics across all three "
        f"runners and summarize findings."
    )


@mcp.prompt()
def compare_models(model_a: str, model_b: str, base_url: str) -> str:
    """Guide the agent through comparing two models."""
    return (
        f"Compare the performance of '{model_a}' and '{model_b}' "
        f"at {base_url}. "
        f"Steps: "
        f"1) Use run_benchmark_preset with preset='quick' for "
        f"'{model_a}' and wait for completion using "
        f"check_vllm_benchmark_status. "
        f"2) Use run_benchmark_preset with preset='quick' for "
        f"'{model_b}' and wait for completion. "
        f"3) Use list_results to find the result files for both runs. "
        f"4) Use compare_results to build a side-by-side comparison "
        f"table of throughput, latency, and TTFT. "
        f"5) Summarize which model wins on each metric and provide "
        f"an overall recommendation."
    )


@mcp.prompt()
def latency_investigation(model: str, base_url: str) -> str:
    """Guide the agent through investigating latency characteristics."""
    return (
        f"Investigate the latency characteristics of '{model}' "
        f"at {base_url}. "
        f"Steps: "
        f"1) Use run_aiperf_benchmark with streaming=True and "
        f"concurrency=1 to establish baseline latency metrics. "
        f"Poll with check_aiperf_benchmark_status until complete. "
        f"2) Use run_aiperf_benchmark again with concurrency=10 "
        f"to measure latency under load. "
        f"Poll with check_aiperf_benchmark_status until complete. "
        f"3) Use list_results to find both result files. "
        f"4) Use compare_results with metrics=['Mean TTFT (ms)', "
        f"'Mean ITL (ms)', 'Mean latency (ms)'] to compare the "
        f"two runs. "
        f"5) Identify bottlenecks: does TTFT degrade under load? "
        f"Does ITL increase? Summarize findings and suggest tuning."
    )


@mcp.prompt()
def hardware_benchmark(model_path: str) -> str:
    """Guide the agent through a local inference benchmark with llama-bench."""
    return (
        f"Run a local inference benchmark of the GGUF model at "
        f"'{model_path}' using llama-bench. "
        f"Steps: "
        f"1) Use run_llama_bench with default settings to establish "
        f"a baseline for prompt processing and text generation speed. "
        f"Poll with check_llama_bench_status every 30 seconds until "
        f"complete. "
        f"2) Optionally, run again with flash_attn=True to measure "
        f"the impact of flash attention. "
        f"3) Optionally, sweep n_gpu_layers (e.g. 0, 20, 40, 99) "
        f"to find the optimal CPU/GPU offload split. "
        f"4) Summarize the results: report prompt eval tok/s, "
        f"text generation tok/s, and which settings gave the best "
        f"performance."
    )


@mcp.prompt()
def ollama_benchmark(model: str) -> str:
    """Guide the agent through benchmarking a model via Ollama."""
    return (
        f"Benchmark '{model}' running on Ollama. "
        f"IMPORTANT: the model parameter must be an Ollama tag exactly as "
        f"shown by 'ollama list' (e.g. 'llama3.1:8b', 'granite4:1b'). "
        f"Do NOT convert it to a HuggingFace repo ID. "
        f"Steps: "
        f"1) Use ping to verify the MCP server is alive. "
        f"2) Use run_ollama_benchmark with model='{model}', the default "
        f"prompts, and 3 iterations to measure inference speed. "
        f"3) Poll with check_ollama_benchmark_status until complete. "
        f"4) Summarize: generation speed (eval rate), prompt processing "
        f"speed, and average durations."
    )
