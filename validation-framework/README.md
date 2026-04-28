# Granite Validation Framework

Unified validation framework for running model checks across inference engines
(vLLM, llama.cpp, Ollama).

[**Code**](https://github.com/ibm-granite/granite.debug-tools/tree/main/validation-framework)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/ibm-granite/granite.debug-tools.git
cd granite.debug-tools/validation-framework/
```

### 2. Install Dependencies

Python 3.12+ required. Install with [uv](https://docs.astral.sh/uv/):

```bash
uv sync --extra dev
```

## Quick Start

```bash
# Install in development mode
uv sync --extra dev

# Run validation against an external Ollama instance
uv run granite-validate --engine ollama --model granite3.3:8b --mode external

# Run specific tests
uv run granite-validate --engine vllm --model ibm-granite/granite-3.3-8b-instruct --tests basic_generation

# Run framework unit tests
uv run pytest
```

## Execution Modes

The CLI supports two execution modes:

- `external` (default): connect to an engine that is already running, run a
  health check, then execute the selected tests. The framework does not start
  or stop the engine process.
- `managed`: start a fresh engine process, wait for it to become healthy, run
  the selected tests, then stop the process in cleanup.

Use `--base-url` in either mode to override the engine default endpoint:

| Engine | Default URL | Managed process |
| --- | --- | --- |
| `ollama` | `http://localhost:11434` | `ollama serve` |
| `vllm` | `http://localhost:8000` | `vllm serve <model>` |
| `llamacpp` | `http://localhost:8080` | `llama-server --model <model>` |

### Running managed mode

Managed mode requires the engine binary to be installed locally, or provided via
`--extra`. The model argument is passed to the managed process:

- Ollama: model tag/name, for example `granite3.3:8b`. The framework starts
  `ollama serve` and ensures the model is available with `/api/pull`.
- vLLM: model identifier or local model path, for example
  `ibm-granite/granite-3.3-8b-instruct`. The framework starts `vllm serve`.
- llama.cpp: path to a local GGUF file. The framework starts `llama-server`.

```bash
# Start and stop an Ollama server for this validation run
uv run granite-validate \
  --engine ollama \
  --model granite3.3:8b \
  --mode managed

# Start and stop vLLM on the default port
uv run granite-validate \
  --engine vllm \
  --model ibm-granite/granite-3.3-8b-instruct \
  --mode managed

# Start and stop llama.cpp with a local GGUF model
uv run granite-validate \
  --engine llamacpp \
  --model /path/to/model.gguf \
  --mode managed
```

If the default port is already in use, set `--base-url` to the port the managed
process should serve on:

```bash
uv run granite-validate \
  --engine vllm \
  --model ibm-granite/granite-3.3-8b-instruct \
  --mode managed \
  --base-url http://localhost:8001
```

Engine-specific managed options can be passed as JSON through `--extra`:

```bash
# vLLM: custom binary, memory/model settings, and raw server flags
uv run granite-validate \
  --engine vllm \
  --model ibm-granite/granite-3.3-8b-instruct \
  --mode managed \
  --extra '{"vllm_bin": "/usr/local/bin/vllm", "max_model_len": 4096, "gpu_memory_utilization": 0.9, "server_args": ["--trust-remote-code"]}'

# llama.cpp: custom binary and server settings
uv run granite-validate \
  --engine llamacpp \
  --model /path/to/model.gguf \
  --mode managed \
  --extra '{"llamacpp_bin": "/usr/local/bin/llama-server", "ctx_size": 8192, "n_gpu_layers": 99, "jinja": true}'

# Ollama: custom binary and pull/startup timeouts
uv run granite-validate \
  --engine ollama \
  --model granite3.3:8b \
  --mode managed \
  --extra '{"ollama_bin": "/usr/local/bin/ollama", "startup_timeout": 60, "pull_timeout": 1200}'
```

Common `--extra` keys across managed engines include:

- `startup_timeout`: seconds to wait for the server to become healthy.
- `stop_timeout`: seconds to wait for graceful shutdown before killing.
- `server_args`: additional arguments appended to the engine server command.
- `request_timeout`: per-request timeout used while running validation tests.

### Running external mode

External mode is useful when the engine is already running, is remote, or has
custom launch settings managed outside the framework:

```bash
uv run granite-validate \
  --engine vllm \
  --model ibm-granite/granite-3.3-8b-instruct \
  --mode external \
  --base-url http://gpu-host:8000
```

## Architecture

```
src/granite_validation/
  domain/       # Pure data models (no dependencies)
  engines/      # Engine adapters behind AbstractEngine ABC
  tests/        # Validation tests behind AbstractValidationTest ABC
    common/     # Tests shared across all engines
    engine_specific/  # Tests for specific engines
  reporting/    # Report generators behind AbstractReporter ABC
  runner.py     # Orchestrator
  cli.py        # CLI entry point
```

### Adding a new engine

1. Create `src/granite_validation/engines/your_engine.py`
2. Implement `AbstractEngine`
3. Decorate with `@register_engine("your_engine")`

### Adding a new test

1. Create a file in `tests/common/` (shared) or `tests/engine_specific/` (targeted)
2. Implement `AbstractValidationTest`
3. Decorate with `@register_test("your_test_id")`
4. For engine-specific tests, override `applicable_engines()` to return the target engine list
