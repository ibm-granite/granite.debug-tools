# Weekly Validation Runner

Runs the granite-validation framework against a matrix of Granite models and inference engines, storing JSON reports and generating a markdown summary.

## Prerequisites

```bash
cd runtimes-validator
uv sync --extra dev
```

## Configuration

### `weekly_matrix.toml` (checked into git)

Defines the engine x model matrix with pinned versions. This is the shared baseline config.

### `weekly_matrix.local.toml` (gitignored)

Machine-specific overrides (GGUF paths, API keys, base URLs). Create by copying `weekly_matrix.toml` and adjusting for your environment.

#### Engine config options

| Field | Description |
|-------|-------------|
| `version` | Pinned engine version (informational) |
| `mode` | `managed` (framework starts/stops) or `external` (connect to running instance) |
| `base_url` | Engine endpoint URL |
| `headers` | List of HTTP headers, e.g. `["RITS_API_KEY: abc123"]` |
| `extra` | Engine-specific options as inline table |

#### Engine-specific `extra` options

**ollama:**
- `skip_pull = true` — skip `/api/pull` (required for locally-created models not in the registry)
- `pull_timeout` — timeout in seconds for model pull (default: 300)
- `server_args` — list of extra CLI arguments passed to `ollama serve` in managed mode

**llamacpp:**
- `n_gpu_layers` — number of layers to offload to GPU
- `ctx_size` — context window size
- `jinja = true` — enable Jinja template support
- `parallel` — number of parallel sequences
- `server_args` — list of extra CLI arguments passed to `llama-server` in managed mode

**vllm:**
- `tensor_parallel_size`, `gpu_memory_utilization`, `max_model_len`, `dtype`, `quantization`
- `bind_host` — host address for vLLM to bind to in managed mode
- `server_args` — list of extra CLI arguments passed to `vllm serve` in managed mode

#### Per-model engine overrides

When an engine needs different config per model (e.g., different base URLs), use a table instead of a string:

```toml
[models.granite-4_1-3b.vllm]
model = "ibm-granite/granite-4.1-3b"
base_url = "https://my-endpoint.com/granite-4-1-3b"
```

## Usage

All commands run from `runtimes-validator/`:

```bash
# Full matrix (all engines x all models)
uv run python weekly_report.py

# Single engine
uv run python weekly_report.py --engines ollama
uv run python weekly_report.py --engines llamacpp
uv run python weekly_report.py --engines vllm

# Single model
uv run python weekly_report.py --models granite-4_1-3b

# Combine filters
uv run python weekly_report.py --engines ollama --models granite-4_1-8b

# Override results date folder
uv run python weekly_report.py --date 2026-04-29

# Use a different config file
uv run python weekly_report.py --config my_matrix.toml
```

## Output

Results are stored in `results/weekly/YYYY-MM-DD/` (inside `runtimes-validator/`):

```
results/weekly/2026-04-29/
  summary.md              # Pass/fail table + failure details
  weekly_matrix.json      # Resolved config snapshot (base + local overrides)
  ollama/
    granite-4_1-3b/
      report_ollama_granite4.1-3b_2026-04-29T10-12-57.json
    granite-4_1-8b/
      ...
  llamacpp/
    granite-4_1-3b/
      ...
  vllm/
    ...
```

## Setting Up Models

### Ollama (unpublished models)

For models not yet in the ollama registry, you can import a GGUF file directly:

```bash
# Simple import from GGUF
ollama create granite4.1:3b -f - <<EOF
FROM /path/to/granite-4.1-3b-f16.gguf
EOF

ollama create granite4.1:8b -f - <<EOF
FROM /path/to/granite-4.1-8b-Q4_K_M.gguf
EOF
```

You can also create a `Modelfile` with additional options:

```
FROM /path/to/model.gguf
PARAMETER temperature 0
PARAMETER num_ctx 8192
TEMPLATE """{{ .Prompt }}"""
SYSTEM """You are a helpful assistant."""
```

Then run:

```bash
ollama create granite4.1:3b -f Modelfile
```

Verify the model is available:

```bash
ollama list | grep granite4.1
```

Then set `skip_pull = true` in your local config to avoid registry pull errors:

```toml
[engines.ollama]
extra = { skip_pull = true }
```

### llama.cpp (converting from safetensors)

If you have safetensors weights, convert to GGUF:

```bash
git clone --depth 1 https://github.com/ggml-org/llama.cpp /tmp/llama.cpp

uv run --with gguf --with numpy --with torch --with safetensors \
  python /tmp/llama.cpp/convert_hf_to_gguf.py \
  /path/to/model-dir/ \
  --outfile /path/to/output-f16.gguf \
  --outtype f16
```

For large models, quantize to reduce memory usage:

```bash
llama-quantize /path/to/model-f16.gguf /path/to/model-Q4_K_M.gguf Q4_K_M
```

### vLLM (remote endpoints)

For remote vLLM endpoints (e.g., RITS), use per-model base URLs and set headers for auth:

```toml
[engines.vllm]
mode = "external"
headers = ["RITS_API_KEY: your-api-key"]

[models.granite-4_1-3b.vllm]
model = "ibm-granite/granite-4.1-3b"
base_url = "https://your-endpoint.com/granite-4-1-3b"
```

## Weekly Workflow

```bash
cd runtimes-validator

# 1. Run the matrix
uv run python weekly_report.py

# 2. Review results
cat results/weekly/2026-04-29/summary.md

# 3. Commit and push
git add results/weekly/2026-04-29/
git commit -m "Weekly validation results 2026-04-29"
git push
```
