# Runtimes Validator

Part of the `granite.debug-tools` monorepo. A Python CLI that runs validation tests against Granite LLM inference engines.

## Repository Layout

```
granite.debug-tools/          # Monorepo root
  runtimes-validator/         # This package
    src/runtimes_validator/   # Source (src-layout, built with Hatchling)
    tests/                    # Pytest unit tests
  perfbench/                  # Performance benchmarking tool
  STaD/                       # Other tooling
```

## Dev Setup

```bash
cd runtimes-validator
uv sync --extra dev
```

## Commands

All commands run from `runtimes-validator/`:

```bash
uv run pytest                  # Unit tests
uv run pytest -x               # Stop on first failure
uv run ruff check .            # Lint
uv run ruff format .           # Format
uv run ruff format --check .   # Verify formatting (no changes)
uv run mypy src/               # Type check (strict mode)
```

**CLI usage** (from `runtimes-validator/`):

```bash
uv run runtimes-validator --engine ollama --model granite3.3:8b                  # External mode (default)
uv run runtimes-validator --engine ollama --model granite3.3:8b --mode managed   # Managed mode (framework starts/stops engine)
uv run runtimes-validator --engine llamacpp --model model.gguf --mode managed    # Managed llama.cpp
uv run runtimes-validator --header 'Authorization: Bearer TOKEN'                 # Custom HTTP headers (repeatable)
uv run runtimes-validator --list-engines                                         # List available engines
uv run runtimes-validator --list-tests                                           # List available tests
```

## Code Conventions

- Python >= 3.12, `from __future__ import annotations` in every file
- Line length: 100 (ruff + mypy configured in `pyproject.toml`)
- Src-layout: package is `src/runtimes_validator/`
- Type annotations everywhere, mypy strict mode

## Architecture

**Two kinds of "tests" exist in this project:**
- **Validation tests** (`src/runtimes_validator/tests/`) — test classes that run against live inference engines. Implement `AbstractValidationTest`, registered via `@register_test("test_id")`.
- **Pytest tests** (`tests/`) — standard unit tests for the framework itself, using mocks.

**Engine abstraction:**
- `AbstractEngine` → `OpenAICompatibleEngine` → `OllamaEngine`, `LlamaCppEngine`, `VllmEngine`
- Engines register via `@register_engine("engine_id")` decorator
- Two modes: `managed` (framework starts/stops the process) and `external` (connect to running instance)

**Runner lifecycle:** resolve engine info → start (if managed) → health_check → run tests → stop (if managed) → report

**Adding a new engine:** subclass `OpenAICompatibleEngine` in `engines/`, decorate with `@register_engine("engine_id")`, implement required methods, and add the import to `cli.py` registration imports to trigger registration.

## Testing

- Unit tests (`tests/`) use `unittest.mock` — never call live inference endpoints in pytest
- Validation tests (`src/runtimes_validator/tests/`) subclass `AbstractValidationTest` and register via `@register_test("test_id")`
- Run `uv run pytest -x -q` before pushing

## PR Workflow

- Target `main` for all PRs
- Squash merge only
- Reference issue numbers in PR body (`Fixes #N`)
- Run full test + lint + type check before marking ready

## Git Conventions

- Imperative subject line with PR number: `Add feature X (#42)`
- Squash merges to `main`
