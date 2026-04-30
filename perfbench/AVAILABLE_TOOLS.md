# Available Tools

This document lists all MCP tools, resources, and prompts exposed by the perfbench server with their parameters and natural-language usage examples.

---

## Tools

### `ping`

Check if the server is alive and responsive.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| *(none)* | | | | |

**Example prompt:**
> Are you there? Ping the benchmarking server.

---

### `list_benchmarks`

List all currently running benchmarks with their IDs, runners, and status.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| *(none)* | | | | |

**Example prompt:**
> Show me all running benchmarks.

---

### `list_results`

List available benchmark result files. Scans result directories and returns a summary of saved results organized by runner and model.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `runner` | `str` | No | `None` | Filter by runner — `"vllm"`, `"aiperf"`, `"guidellm"`, `"llamabench"`, `"ollama"`, or `None` for all. |

**Example prompt:**
> List all saved benchmark results.

> Show me only the AIPerf results.

---

### `read_result`

Read a specific benchmark result file. Returns the JSON content pretty-printed. Use `list_results` to discover available results.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `runner` | `str` | Yes | | Runner name — `"vllm"`, `"aiperf"`, `"guidellm"`, `"llamabench"`, or `"ollama"`. |
| `model` | `str` | Yes | | Model directory name (e.g. `"ibm-granite_granite-4.0-h-tiny"`). |
| `run` | `str` | Yes | | Run identifier — for vLLM/llama-bench/ollama this is the filename stem, for AIPerf/GuideLLM this is the timestamp directory name. |

**Example prompt:**
> Read the vLLM result for model `ibm-granite_granite-4.0-h-tiny`, run `20260303175301_VLLM_curr=10_input=256_output=128`.

> Show me the GuideLLM benchmark data for `granite-3.3-8b-instruct` from run `20260226151316`.

---

### `compare_results`

Compare metrics across multiple benchmark result files. Returns a side-by-side markdown comparison table with normalized metric names across runners.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `results` | `list[dict]` | Yes | | List of result references. Each dict has keys `runner`, `model`, and `run` (same identifiers used by `read_result`). |
| `metrics` | `list[str]` | No | `None` | Specific metric display names to include (e.g. `["Request throughput (req/s)", "Mean TTFT (ms)"]`). If `None`, all normalized metrics are shown. |

Available normalized metric names: `Request throughput (req/s)`, `Output throughput (tok/s)`, `Mean TTFT (ms)`, `Mean ITL (ms)`, `Mean latency (ms)`, `Completed requests`, `Prompt eval (tok/s)`, `Generation (tok/s)`.

**Example prompt:**
> Compare the vLLM results for `ibm-granite_granite-4.0-h-tiny` with concurrency 1 and concurrency 10. Show request throughput and TTFT.

> Compare the AIPerf result for `granite_model` run `20260301120000` against the GuideLLM result for the same model run `20260301130000`.

---

### `run_vllm_benchmark`

Launch a vLLM benchmark against an LLM service. Starts `vllm bench serve` as a background subprocess and returns a benchmark ID along with the first 30 seconds of output.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `model` | `str` | Yes | | Name of the model to benchmark. |
| `base_url` | `str` | Yes | | Base URL of the LLM service. |
| `served_model_name` | `str` | Yes | | Name of the served model. |
| `backend` | `str` | No | `"openai"` | Backend type — `"openai"`, `"openai-chat"`, or `"vllm"`. |
| `endpoint` | `str` | No | `"/v1/completions"` | API endpoint path. |
| `num_prompts` | `int` | No | `10` | Number of prompts to send. |
| `dataset_name` | `str` | No | `"random"` | Dataset to use — `"random"`, `"sharegpt"`, etc. |
| `max_concurrency` | `int` | No | `1` | Maximum number of concurrent requests. |
| `random_input_len` | `int` | No | `10` | Input token length (random dataset). |
| `random_output_len` | `int` | No | `100` | Output token length (random dataset). |
| `result_dir` | `str` | No | `"results_vllm_bench"` | Directory to save results in. |
| `ready_check_timeout_sec` | `int` | No | `10` | Seconds to wait for server readiness. |
| `api_token` | `str` | No | `None` | API authentication token. Uses standard `Authorization: Bearer` header by default; uses a custom header when `auth_header_name` is provided. |
| `auth_header_name` | `str` | No | `None` | Custom header name for authentication (e.g. `"CUSTOM_API_KEY_NAME"`). When `None`, uses standard `Authorization: Bearer` header. |
| `request_rate` | `float` | No | `None` | Requests per second (omit for unlimited). |

**Example prompt:**
> Run a vLLM benchmark for model `ibm-granite/granite-4.0-micro` with served model name `granite-4.0-micro` against `http://localhost:11434/v1` using 50 prompts, concurrency of 10, input length 128, and output length 256.

---

### `check_vllm_benchmark_status`

Check the status of a running vLLM benchmark. Returns any new output produced since the last check.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `benchmark_id` | `str` | Yes | | The ID returned by `run_vllm_benchmark`. |

**Example prompt:**
> Check the status of vLLM benchmark `a1b2c3d4`.

---

### `stop_vllm_benchmark`

Terminate a running vLLM benchmark.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `benchmark_id` | `str` | Yes | | The ID returned by `run_vllm_benchmark`. |

**Example prompt:**
> Stop the vLLM benchmark `a1b2c3d4`.

---

### `run_aiperf_benchmark`

Launch an AIPerf benchmark against an LLM service. Starts `aiperf profile` as a background subprocess and returns a benchmark ID along with the first 30 seconds of output.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `model` | `str` | Yes | | Name of the model to benchmark. |
| `tokenizer` | `str` | Yes | | Name of the tokenizer to use. |
| `url` | `str` | Yes | | Base URL of the LLM service. |
| `endpoint_type` | `str` | No | `"chat"` | API endpoint type — `"chat"`, `"completions"`, `"embeddings"`, etc. |
| `streaming` | `bool` | No | `True` | Enable streaming responses for TTFT/ITL metrics. |
| `concurrency` | `int` | No | `1` | Number of concurrent requests to maintain. |
| `request_count` | `int` | No | `10` | Total number of requests to send. |
| `request_rate` | `float` | No | `None` | Target requests per second (omit for concurrency mode). |
| `isl` | `int` | No | `None` | Mean input sequence length in tokens (synthetic dataset). |
| `osl` | `int` | No | `None` | Mean output sequence length in tokens. |
| `benchmark_duration` | `float` | No | `None` | Maximum benchmark runtime in seconds. |
| `api_key` | `str` | No | `None` | API authentication token. Uses aiperf's native `--api-key` flag (standard `Authorization: Bearer`) by default; uses a custom header when `auth_header_name` is provided. |
| `auth_header_name` | `str` | No | `None` | Custom header name for authentication (e.g. `"CUSTOM_API_KEY_NAME"`). When `None`, uses standard `Authorization: Bearer` header via `--api-key`. |
| `artifact_dir` | `str` | No | `"results_aiperf"` | Directory to store benchmark artifacts. |
| `ui_type` | `str` | No | `"none"` | UI display mode — `"none"`, `"simple"`, or `"dashboard"`. |
| `warmup_request_count` | `int` | No | `None` | Number of warmup requests before benchmarking. |

**Example prompt:**
> Run an AIPerf benchmark for `ibm-granite/granite-4.0-micro` with tokenizer `ibm-granite/granite-4.0-micro` against `http://localhost:11434` using chat endpoint, 100 requests, concurrency of 5, streaming enabled, input length 64, and output length 128.

---

### `check_aiperf_benchmark_status`

Check the status of a running AIPerf benchmark. Returns any new output produced since the last check.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `benchmark_id` | `str` | Yes | | The ID returned by `run_aiperf_benchmark`. |

**Example prompt:**
> What's the status of AIPerf benchmark `e5f6a7b8`?

---

### `stop_aiperf_benchmark`

Terminate a running AIPerf benchmark.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `benchmark_id` | `str` | Yes | | The ID returned by `run_aiperf_benchmark`. |

**Example prompt:**
> Stop the AIPerf benchmark `e5f6a7b8`.

---

### `run_guidellm_benchmark`

Launch a GuideLLM benchmark against an LLM service. Starts `guidellm benchmark` as a background subprocess and returns a benchmark ID along with the first 30 seconds of output.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `target` | `str` | Yes | | URL of the OpenAI-compatible endpoint. |
| `data` | `str` | No | `None` | Data source — synthetic spec, HuggingFace dataset ID, or local file. When `None`, built from `prompt_tokens` and `output_tokens`. |
| `prompt_tokens` | `int` | No | `256` | Number of prompt tokens for synthetic data. |
| `output_tokens` | `int` | No | `128` | Number of output tokens for synthetic data. |
| `profile` | `str` | No | `"sweep"` | Load profile — `"synchronous"`, `"concurrent"`, `"throughput"`, `"constant"`, `"poisson"`, or `"sweep"`. |
| `rate` | `float` | No | `None` | Numeric rate value (meaning depends on profile). |
| `request_type` | `str` | No | `"chat_completions"` | API format — `"chat_completions"`, `"text_completions"`, `"audio_transcription"`, or `"audio_translation"`. |
| `max_seconds` | `int` | No | `None` | Maximum duration in seconds per benchmark. |
| `max_requests` | `int` | No | `10` | Maximum number of requests per benchmark. |
| `warmup` | `float` | No | `None` | Warm-up specification (0-1 = percentage, >=1 = absolute). |
| `cooldown` | `float` | No | `None` | Cool-down specification (same format as warmup). |
| `max_errors` | `int` | No | `None` | Maximum errors before stopping. |
| `processor` | `str` | No | `None` | Tokenizer/processor name for synthetic data. |
| `model` | `str` | No | `None` | Model name to pass in the generated requests. |
| `api_key` | `str` | No | `None` | API authentication key (passed as Bearer token). |
| `output_dir` | `str` | No | `"results_guidellm"` | Directory for output files (json, csv, html). |
| `detect_saturation` | `bool` | No | `False` | Enable over-saturation detection. |

**Example prompt:**
> Run a GuideLLM sweep benchmark against `http://localhost:11434/v1` for model `ibm-granite/granite-4.0-micro` with 200 max requests, 512 prompt tokens, and 256 output tokens.

---

### `check_guidellm_benchmark_status`

Check the status of a running GuideLLM benchmark. Returns any new output produced since the last check.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `benchmark_id` | `str` | Yes | | The ID returned by `run_guidellm_benchmark`. |

**Example prompt:**
> Check the GuideLLM benchmark `c9d0e1f2`.

---

### `stop_guidellm_benchmark`

Terminate a running GuideLLM benchmark.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `benchmark_id` | `str` | Yes | | The ID returned by `run_guidellm_benchmark`. |

**Example prompt:**
> Cancel GuideLLM benchmark `c9d0e1f2`.

---

### `run_llama_bench`

Launch a llama-bench local inference benchmark. Runs `llama-bench` against a GGUF model file to measure raw prompt processing and text generation speed. No server required.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `model_path` | `str` | Yes | | Path to the GGUF model file. |
| `n_prompt` | `int` | No | `512` | Number of prompt tokens to process. |
| `n_gen` | `int` | No | `128` | Number of tokens to generate. |
| `n_gpu_layers` | `int` | No | `99` | Number of layers to offload to GPU. |
| `batch_size` | `int` | No | `2048` | Logical batch size. |
| `ubatch_size` | `int` | No | `512` | Physical batch size. |
| `threads` | `int` | No | `None` | Number of CPU threads (omit for auto-detect). |
| `flash_attn` | `bool` | No | `False` | Enable flash attention. |
| `cache_type_k` | `str` | No | `"f16"` | KV cache type for keys — `"f16"`, `"q8_0"`, `"q4_0"`. |
| `cache_type_v` | `str` | No | `"f16"` | KV cache type for values. |
| `repetitions` | `int` | No | `5` | Number of test repetitions. |
| `n_depth` | `int` | No | `0` | KV cache depth (0 = same as `n_prompt`). |
| `split_mode` | `str` | No | `"layer"` | Multi-GPU split mode — `"layer"`, `"row"`, `"none"`. |
| `use_mmap` | `bool` | No | `True` | Use memory-mapped model loading. |
| `result_dir` | `str` | No | `""` | Base directory for saving results. Defaults to the project's `results_llama_bench` directory. |

**Example prompt:**
> Run llama-bench on `/models/granite-4.0-micro.Q4_K_M.gguf` with flash attention enabled, 99 GPU layers, and 10 repetitions.

---

### `check_llama_bench_status`

Check the status of a running llama-bench benchmark. Returns any new output produced since the last check.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `benchmark_id` | `str` | Yes | | The ID returned by `run_llama_bench`. |

**Example prompt:**
> Check the llama-bench benchmark `x1y2z3w4`.

---

### `stop_llama_bench`

Terminate a running llama-bench benchmark.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `benchmark_id` | `str` | Yes | | The ID returned by `run_llama_bench`. |

**Example prompt:**
> Stop the llama-bench benchmark `x1y2z3w4`.

---

### `run_ollama_benchmark`

Launch an Ollama benchmark for local inference performance. Sends prompts to an Ollama instance via its REST API and measures generation speed, prompt processing speed, and timing.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `model` | `str` | Yes | | Ollama model tag as shown by `ollama list` — use `name:tag` format (e.g. `"granite3.3:8b"`, `"granite4:1b"`). Do **not** use HuggingFace repo IDs. |
| `base_url` | `str` | No | `"http://localhost:11434"` | Ollama server URL. |
| `prompts` | `list[str]` | No | `None` | Custom prompts to benchmark. `None` uses a built-in set of 5 diverse prompts. |
| `num_iterations` | `int` | No | `3` | Number of times to repeat each prompt. |
| `category` | `str` | No | `"general"` | Label for this benchmark category. |
| `result_dir` | `str` | No | `""` | Base directory for saving results. Defaults to the project's `results_ollama_bench` directory. |

**Example prompt:**
> Run an Ollama benchmark for `granite3.3:8b` with 5 iterations and the default prompts.

> Benchmark `granite4:1b` on Ollama running at `http://192.168.1.100:11434`.

---

### `check_ollama_benchmark_status`

Check the status of a running Ollama benchmark. Returns any new output produced since the last check.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `benchmark_id` | `str` | Yes | | The ID returned by `run_ollama_benchmark`. |

**Example prompt:**
> Check the Ollama benchmark `a1b2c3d4`.

---

### `stop_ollama_benchmark`

Terminate a running Ollama benchmark.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `benchmark_id` | `str` | Yes | | The ID returned by `run_ollama_benchmark`. |

**Example prompt:**
> Stop the Ollama benchmark `a1b2c3d4`.

---

### `run_benchmark_preset`

Run a benchmark using a predefined configuration preset. Simplifies benchmarking by encapsulating common parameter combinations behind a single preset name.

Available presets: `quick` (vLLM smoke test), `throughput` (vLLM max throughput), `latency` (AIPerf latency profile), `stress` (AIPerf stress test), `sweep` (GuideLLM load sweep), `inference` (llama-bench local inference), `ollama-quick` (Ollama quick benchmark), `full` (all runners — includes llama-bench if `model_path` provided, Ollama if `ollama_model` provided).

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `preset` | `str` | Yes | | Preset name — `"quick"`, `"throughput"`, `"latency"`, `"stress"`, `"sweep"`, `"inference"`, `"ollama-quick"`, or `"full"`. |
| `model` | `str` | No | `""` | Model identifier — required for serving presets and for `"ollama-quick"` (pass the Ollama tag here, e.g. `"granite3.3:8b"`). |
| `base_url` | `str` | No | `""` | URL of the LLM service (required for serving presets). |
| `served_model_name` | `str` | No | `None` | Served model name (defaults to `model`). |
| `api_token` | `str` | No | `None` | API authentication token. |
| `auth_header_name` | `str` | No | `None` | Custom header name for authentication (e.g. `"CUSTOM_API_KEY_NAME"`). When `None`, uses standard `Authorization: Bearer` header. Applies to vLLM and AIPerf presets only. |
| `model_path` | `str` | No | `None` | Path to GGUF model file (required for `"inference"` preset, optional for `"full"` to include llama-bench). |
| `ollama_model` | `str` | No | `None` | Ollama model tag (e.g. `"granite3.3:8b"`). Only used by the `"full"` preset to include an Ollama run alongside the other runners. For `"ollama-quick"`, pass the model tag via `model` instead. |
| `ollama_url` | `str` | No | `"http://localhost:11434"` | Ollama server URL. |

**Example prompt:**
> Run the quick preset for `ibm-granite/granite-4.0-micro` against `http://localhost:8000` with API token `<secret>` using custom auth header `CUSTOM_API_KEY_NAME`.

> Run the inference preset for the model at `/models/granite-4.0-micro.Q4_K_M.gguf`.

> Run the ollama-quick preset for `granite3.3:8b`.

---

### `run_streamlit_dashboard`

Start the Streamlit results dashboard. Launches `streamlit run streamlit_app.py` as a background process.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `port` | `int` | No | `8501` | Port number for the Streamlit server. |

**Example prompt:**
> Start the benchmark results dashboard on port 8080.

---

### `stop_streamlit_dashboard`

Stop the running Streamlit results dashboard.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| *(none)* | | | | |

**Example prompt:**
> Stop the dashboard.

---

## Resources

### `info://server`

Returns basic information about this MCP server (name and version).

**Example prompt:**
> What version of the benchmarking server is running?

---

## Prompts

### `benchmark_summary`

Generate a prompt that asks for a comprehensive benchmark summary of a given model.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model_name` | `str` | Yes | The model to summarize. |

**Example prompt:**
> Summarize the benchmark results for `ibm-granite/granite-4.0-micro`.

---

### `quick_benchmark`

Guide the agent through a quick vLLM smoke test of a model.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model` | `str` | Yes | Model name to benchmark. |
| `base_url` | `str` | Yes | URL of the LLM service. |

---

### `full_benchmark_suite`

Guide the agent through a comprehensive benchmark suite across all serving runners (vLLM, AIPerf, GuideLLM).

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model` | `str` | Yes | Model name to benchmark. |
| `base_url` | `str` | Yes | URL of the LLM service. |

---

### `compare_models`

Guide the agent through comparing two models side-by-side.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model_a` | `str` | Yes | First model name. |
| `model_b` | `str` | Yes | Second model name. |
| `base_url` | `str` | Yes | URL of the LLM service. |

---

### `latency_investigation`

Guide the agent through investigating latency characteristics under different concurrency levels.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model` | `str` | Yes | Model name to investigate. |
| `base_url` | `str` | Yes | URL of the LLM service. |

---

### `hardware_benchmark`

Guide the agent through a local inference benchmark with llama-bench, including optional flash attention and GPU layer sweeps.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model_path` | `str` | Yes | Path to the GGUF model file. |

**Example prompt:**
> Benchmark the GGUF model at `/models/granite-4.0-micro.Q4_K_M.gguf` with different GPU offload settings.

---

### `ollama_benchmark`

Guide the agent through benchmarking a model running on Ollama — measuring generation speed, prompt processing speed, and durations.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model` | `str` | Yes | Ollama model tag (e.g. `"granite3.3:8b"`, `"granite4:1b"`). |

**Example prompt:**
> Benchmark `granite3.3:8b` running on Ollama and summarize the inference speed.
