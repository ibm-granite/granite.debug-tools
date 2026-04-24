# granite.debug.perfbench - MCP server for Granite benchmarking

MCP server that manages LLM benchmark runs as asynchronous subprocesses, wrapping five benchmark runners behind a unified [Model Context Protocol](https://modelcontextprotocol.io) tool interface.

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (recommended package manager)

## Installation

```bash
# Clone the repository
git clone https://github.com/ibm-granite/granite.debug-tools.git
cd granite.debug-tools/perfbench

# Install dependencies
uv sync
```

## Running the Server

### Development mode (with MCP Inspector)

```bash
uv run mcp dev src/perfbench/server.py
```

This launches the [MCP Inspector](https://github.com/modelcontextprotocol/inspector) where you can interactively test your tools, resources, and prompts.

### Direct execution

```bash
uv run perfbench
```

### With Claude Desktop

```bash
uv run mcp install src/perfbench/server.py --name "granite.debug.perfbench"
```

### Using the LangChain Client

The project includes a LangChain-based client that connects to the MCP server using IBM Granite 4. It supports two providers, controlled by the `LLM_PROVIDER` env var:

#### Ollama (default — local, no credentials)

```bash
ollama pull granite4:micro
uv run python examples/langchain_client.py
```

#### watsonx.ai (cloud)

```bash
export LLM_PROVIDER=watsonx
export WATSONX_APIKEY="your-api-key"
export WATSONX_URL="https://us-south.ml.cloud.ibm.com"
export WATSONX_PROJECT_ID="your-project-id"
uv run python examples/langchain_client.py
```

## Benchmark Tools

This server exposes MCP tools that wrap two benchmarking CLIs. Both run as background subprocesses and are managed with matching `run_*`, `check_*_status`, and `stop_*` tools.

### vLLM Benchmark (`vllm bench serve`)

Part of the [vLLM](https://docs.vllm.ai/en/latest/) project. Measures throughput, latency, TTFT, and TPOT for models served by vLLM-compatible endpoints.

| MCP Tool | Description |
|---|---|
| `run_vllm_benchmark` | Launch a benchmark run |
| `check_vllm_benchmark_status` | Poll for new output / final results |
| `stop_vllm_benchmark` | Terminate a running benchmark |

**Requires:** `pip install vllm`

```bash
# Equivalent CLI
vllm bench serve --model ibm-granite/granite-4.0-micro --base-url http://localhost:8000 \
    --served-model-name granite-4.0-micro --num-prompts 100 --max-concurrency 1
```

📖 [vLLM Benchmarking Documentation](https://docs.vllm.ai/en/stable/cli/bench/serve/)

Example of agent prompt:

```
Run vllm benchmarks for model ibm-granite/granite-4.0-micro and served model name granite-4.0-micro with base url https://your-vllm-endpoint.example.com and api token <secret>
```



---

### AIPerf (`aiperf profile`)

[AIPerf](https://github.com/ai-dynamo/aiperf) is a comprehensive benchmarking tool by NVIDIA that measures latency, throughput, TTFT, ITL, and many more metrics for generative AI models. It supports concurrency, request-rate, and trace-replay modes with multiprocess scalability.

| MCP Tool | Description |
|---|---|
| `run_aiperf_benchmark` | Launch an aiperf profiling run |
| `check_aiperf_benchmark_status` | Poll for new output / final results |
| `stop_aiperf_benchmark` | Terminate a running benchmark |

**Requires:** `pip install aiperf`

```bash
# Equivalent CLI
aiperf profile --model ibm-granite/granite-4.0-micro --url http://localhost:8000 \
    --endpoint-type chat --streaming --concurrency 10 --request-count 100
```

📖 [AIPerf README](https://github.com/ai-dynamo/aiperf) · [CLI Options](https://github.com/ai-dynamo/aiperf/blob/main/docs/cli_options.md) · [Metrics Reference](https://github.com/ai-dynamo/aiperf/blob/main/docs/metrics_reference.md) · [Tutorial](https://github.com/ai-dynamo/aiperf/blob/main/docs/tutorial.md)

Example of agent prompt:

```
Run aiperf benchmarks for model granite-4.0-micro and tokenizer ibm-granite/granite-4.0-micro with base url https://your-vllm-endpoint.example.com and api token <secret>
```

---

### GuideLLM (`guidellm benchmark`)

[GuideLLM](https://github.com/vllm-project/guidellm) by the vLLM project evaluates and enhances LLM deployments for real-world inference needs. It supports sweep, synchronous, concurrent, throughput, constant-rate, and Poisson load profiles with warmup/cooldown, saturation detection, and rich output formats (JSON, CSV, HTML).

| MCP Tool | Description |
|---|---|
| `run_guidellm_benchmark` | Launch a GuideLLM benchmark run |
| `check_guidellm_benchmark_status` | Poll for new output / final results |
| `stop_guidellm_benchmark` | Terminate a running benchmark |

**Requires:** `pip install guidellm[recommended]`

```bash
# Equivalent CLI
guidellm benchmark --target http://localhost:8000 \
    --profile sweep --max-seconds 30 \
    --data "prompt_tokens=256,output_tokens=128"
```

📖 [GuideLLM Documentation](https://github.com/vllm-project/guidellm)

Example of agent prompt:

```
Run guidellm benchmarks with target url https://your-vllm-endpoint.example.com using sweep profile, model granite-4.0-micro, processor ibm-granite/granite-4.0-micro, api key <secret>, and synthetic data prompt_tokens=256,output_tokens=128
```


## Benchmark Presets

The `run_benchmark_preset` tool simplifies benchmarking by encapsulating common parameter combinations behind a single preset name. Instead of specifying 10+ parameters, an agent (or user) can pick a preset and provide just the model, URL, and optional auth token.

| MCP Tool | Description |
|---|---|
| `run_benchmark_preset` | Run a benchmark using a predefined configuration |

### Available presets

| Preset | Runner | Description | Key settings |
|---|---|---|---|
| `quick` | vLLM bench | Fast smoke test | 10 prompts, concurrency=1, short I/O |
| `throughput` | vLLM bench | Max throughput test | 100 prompts, concurrency=10, larger payloads |
| `latency` | AIPerf | Latency-focused profile | streaming, concurrency=1, 50 requests |
| `stress` | AIPerf | High-load stress test | concurrency=50, 500 requests |
| `sweep` | GuideLLM | Load sweep profile | sweep profile, 100 requests, synthetic data |
| `full` | All three | Complete profile | Runs quick + latency + sweep in sequence |

### Example agent interaction

```
User: "Run a quick benchmark of granite-4.0-micro at http://localhost:8000"
Agent: [calls run_benchmark_preset(preset="quick", model="ibm-granite/granite-4.0-micro", base_url="http://localhost:8000")]
```

---

## Prompt Templates

MCP prompts are reusable interaction templates that guide agents through multi-step benchmarking workflows. They encode best practices and reference actual MCP tools by name, so the agent knows exactly which tools to call and in what order.

| Prompt | Parameters | Description |
|---|---|---|
| `benchmark_summary` | `model_name` | Ask for a comprehensive benchmark summary of a model |
| `quick_benchmark` | `model`, `base_url` | Quick smoke test: ping, run vLLM (10 prompts), poll, summarize |
| `full_benchmark_suite` | `model`, `base_url` | Full suite: vLLM + AIPerf + GuideLLM + dashboard + comparison |
| `compare_models` | `model_a`, `model_b`, `base_url` | Head-to-head model comparison using quick presets |
| `latency_investigation` | `model`, `base_url` | Deep-dive into latency: baseline vs. under-load with AIPerf |

### Example agent interaction

```
User: "I want to compare granite-4.0-micro and granite-4.0-tiny"
Agent: [uses compare_models prompt template]
  1. Runs quick benchmark preset for granite-4.0-micro
  2. Runs quick benchmark preset for granite-4.0-tiny
  3. Lists results and builds side-by-side comparison
  4. Summarizes which model wins on each metric
```

---

## Results Dashboard

A Streamlit dashboard is included to visualise benchmark results from all tools, organised in tabs.

The dashboard can be started via the MCP tools or manually from the CLI:

| MCP Tool | Description |
|---|---|
| `run_streamlit_dashboard` | Start the dashboard (returns the URL) |
| `stop_streamlit_dashboard` | Stop the running dashboard |

```bash
# Manual launch
uv run streamlit run streamlit_app.py
```

| Tab | Source directory | File pattern |
|---|---|---|
| **🚀 vLLM Bench** | `results_vllm_bench/` | `*.json` |
| **📊 AIPerf** | `results_aiperf/<timestamp>/` | `profile_export_aiperf.json` |
| **🔬 GuideLLM** | `results_guidellm/<timestamp>/` | `benchmarks.json` |

Each tab displays results as expandable sections with:

- **Key metric cards** — headline numbers (Req/s, Tok/s, TTFT, latency, etc.)
- **Toggleable detail table** — full metric breakdown including percentiles

> **Note:** Result files are generated automatically by the benchmark tools.

## Running Tests

```bash
uv run pytest tests/ -v
```

## Linting

```bash
uv run ruff check src/
uv run ruff format --check src/
```

## Project Structure

```
perfbench/
├── pyproject.toml
├── README.md
├── LICENSE
├── AVAILABLE_TOOLS.md
├── streamlit_app.py                  # Results dashboard (Streamlit)
├── examples/
│   └── langchain_client.py           # LangChain MCP client (Granite 4)
├── src/
│   └── perfbench/
│       ├── __init__.py               # Package metadata
│       ├── server.py                 # FastMCP server instance & entry point
│       ├── tools.py                  # MCP tools (~24 tools for 5 runners)
│       ├── resources.py              # MCP resources (data for LLMs)
│       ├── prompts.py                # MCP prompts (interaction templates)
│       ├── dashboard_helpers.py      # Data loading/formatting for dashboard
│       └── _ollama_bench_runner.py   # Ollama benchmark subprocess wrapper
└── tests/
    ├── __init__.py
    ├── test_tools.py
    ├── test_dashboard_helpers.py
    └── test_prompts.py
```

## Adding New Capabilities

### Tools

Add new tools in `src/perfbench/tools.py`:

```python
from perfbench.server import mcp

@mcp.tool()
def my_tool(param: str) -> str:
    """Description of what this tool does."""
    return f"Result: {param}"
```

### Resources

Add new resources in `src/perfbench/resources.py`:

```python
from perfbench.server import mcp

@mcp.resource("data://my-resource")
def my_resource() -> str:
    """Description of what data this resource exposes."""
    return "resource data"
```

### Prompts

Add new prompts in `src/perfbench/prompts.py`:

```python
from perfbench.server import mcp

@mcp.prompt()
def my_prompt(topic: str) -> str:
    """Description of this prompt template."""
    return f"Please analyze {topic}."
```

