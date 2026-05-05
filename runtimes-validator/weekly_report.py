"""Weekly validation matrix runner for Granite models.

Reads weekly_matrix.toml, iterates over engine × model combinations,
calls granite-validate for each, and generates a summary report.

Usage:
    uv run python weekly_report.py
    uv run python weekly_report.py --engines ollama --models granite-4_1-8b
    uv run python weekly_report.py --date 2026-04-21
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import tomllib
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "weekly_matrix.toml"
RESULTS_BASE = SCRIPT_DIR / "results" / "weekly"
DELAY_BETWEEN_RUNS = 10

ENGINE_DISPLAY_NAMES: dict[str, str] = {
    "ollama": "ollama",
    "llamacpp": "llama.cpp",
    "vllm": "vLLM",
}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_config(config_path: Path) -> dict[str, Any]:
    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    local_path = config_path.with_suffix(".local.toml")
    if local_path.exists():
        with open(local_path, "rb") as f:
            local = tomllib.load(f)
        config = deep_merge(config, local)

    return config


def build_matrix(
    config: dict[str, Any],
    engine_filter: list[str] | None = None,
    model_filter: list[str] | None = None,
) -> list[tuple[str, str, str, dict[str, Any]]]:
    """Build (engine_id, model_key, model_id, engine_config) tuples."""
    engines = config.get("engines", {})
    models = config.get("models", {})
    matrix: list[tuple[str, str, str, dict[str, Any]]] = []

    for engine_id, engine_cfg in engines.items():
        if engine_filter and engine_id not in engine_filter:
            continue
        for model_key, model_map in models.items():
            if model_filter and model_key not in model_filter:
                continue
            entry = model_map.get(engine_id)
            if not entry:
                continue
            if isinstance(entry, dict):
                model_id = entry["model"]
                merged_cfg = deep_merge(engine_cfg, entry)
                del merged_cfg["model"]
            else:
                model_id = entry
                merged_cfg = engine_cfg
            matrix.append((engine_id, model_key, model_id, merged_cfg))

    return matrix


def run_combination(
    engine_id: str,
    model_id: str,
    mode: str,
    output_dir: Path,
    base_url: str | None = None,
    extra: dict[str, Any] | None = None,
    headers: list[str] | None = None,
) -> int:
    from granite_validation.cli import main as validate_main

    argv = [
        "--engine",
        engine_id,
        "--model",
        model_id,
        "--mode",
        mode,
        "--output-dir",
        str(output_dir),
    ]
    if base_url:
        argv.extend(["--base-url", base_url])
    if extra:
        argv.extend(["--extra", json.dumps(extra)])
    if headers:
        for h in headers:
            argv.extend(["--header", h])

    try:
        return validate_main(argv)
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 1
    except Exception as exc:
        print(f"  ERROR: {exc}", file=sys.stderr)
        return 1


def collect_reports(results_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """Read all JSON reports, keyed by (engine_id, model_key)."""
    reports: dict[tuple[str, str], dict[str, Any]] = {}
    for engine_dir in sorted(results_dir.iterdir()):
        if not engine_dir.is_dir() or engine_dir.name == "summary.md":
            continue
        engine_id = engine_dir.name
        for model_dir in sorted(engine_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            model_key = model_dir.name
            for json_file in sorted(model_dir.glob("report_*.json")):
                with open(json_file) as f:
                    reports[(engine_id, model_key)] = json.load(f)
    return reports


def generate_summary(
    date_str: str,
    config: dict[str, Any],
    results_dir: Path,
) -> str:
    reports = collect_reports(results_dir)
    engines = config.get("engines", {})
    models = config.get("models", {})
    engine_ids = list(engines.keys())
    model_keys = list(models.keys())

    engine_headers: list[str] = []
    for eid in engine_ids:
        display = ENGINE_DISPLAY_NAMES.get(eid, eid)
        version = engines[eid].get("version", "?")
        for mk in model_keys:
            report = reports.get((eid, mk))
            if report:
                version = report.get("engine_info", {}).get("version", version)
                break
        engine_headers.append(f"{display} {version}")

    lines: list[str] = []
    lines.append(f"# Weekly Validation — {date_str}\n")
    header_row = "| Model | " + " | ".join(engine_headers) + " |"
    sep_row = "|-------|" + "|".join(":---:" for _ in engine_ids) + "|"
    lines.append(header_row)
    lines.append(sep_row)

    failures: list[str] = []

    for model_key in model_keys:
        label = model_key.replace("granite-4_1-", "4.1 ").replace("granite-", "")
        cells: list[str] = []
        for engine_id in engine_ids:
            report = reports.get((engine_id, model_key))
            if report is None:
                cells.append("—")
                continue

            results_list = report.get("results", [])
            passed_count = sum(1 for r in results_list if r.get("passed"))
            total_count = len(results_list)

            has_lifecycle_error = report.get("lifecycle_error") is not None
            all_passed = report.get("all_passed", False)

            if has_lifecycle_error:
                cells.append(f"ERROR ({passed_count}/{total_count})")
                err = report["lifecycle_error"]
                display = ENGINE_DISPLAY_NAMES.get(engine_id, engine_id)
                failures.append(f"### {display} — {model_key} — lifecycle error")
                failures.append(f"- {err}\n")
            elif all_passed:
                cells.append(f"PASS ({passed_count}/{total_count})")
            else:
                cells.append(f"FAIL ({passed_count}/{total_count})")
                for result in results_list:
                    if result.get("passed"):
                        continue
                    test_id = result.get("test_id", "unknown")
                    display = ENGINE_DISPLAY_NAMES.get(engine_id, engine_id)
                    failures.append(f"### {display} — {model_key} — `{test_id}`")
                    if result.get("error"):
                        failures.append(f"- Error: {result['error']}")
                    for check in result.get("checks", []):
                        if not check.get("passed"):
                            detail = check.get("detail", "")
                            name = check.get("name", "unknown")
                            failures.append(f"- Check `{name}`: {detail}")
                    failures.append("")

        row = f"| {label} | " + " | ".join(cells) + " |"
        lines.append(row)

    if failures:
        lines.append("")
        lines.append("## Failures")
        lines.extend(failures)

    lines.append("")
    return "\n".join(lines)


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run weekly Granite validation matrix",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Path to matrix TOML config (default: {DEFAULT_CONFIG.name})",
    )
    parser.add_argument(
        "--engines",
        default=None,
        help="Comma-separated engine IDs to run (default: all)",
    )
    parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated model keys to run (default: all)",
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Override date for results directory (YYYY-MM-DD, default: today)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_cli_parser()
    args = parser.parse_args(argv)

    config = load_config(args.config)

    engine_filter = [e.strip() for e in args.engines.split(",")] if args.engines else None
    model_filter = [m.strip() for m in args.models.split(",")] if args.models else None
    date_str = args.date or datetime.now(timezone.utc).strftime("%Y-%m-%d")

    matrix = build_matrix(config, engine_filter, model_filter)
    if not matrix:
        print("No combinations to run. Check config and filters.", file=sys.stderr)
        return 1

    results_dir = RESULTS_BASE / date_str
    results_dir.mkdir(parents=True, exist_ok=True)

    total = len(matrix)
    all_exit_codes: list[int] = []

    print(f"Weekly Granite Validation — {date_str}")
    print(f"Running {total} combination(s)\n")

    for i, (engine_id, model_key, model_id, engine_cfg) in enumerate(matrix, 1):
        display = ENGINE_DISPLAY_NAMES.get(engine_id, engine_id)
        print(f"[{i}/{total}] {display} × {model_key} ({model_id})")

        output_dir = results_dir / engine_id / model_key
        output_dir.mkdir(parents=True, exist_ok=True)

        mode = engine_cfg.get("mode", "external")
        base_url = engine_cfg.get("base_url")
        extra = engine_cfg.get("extra")
        headers = engine_cfg.get("headers")

        exit_code = run_combination(
            engine_id=engine_id,
            model_id=model_id,
            mode=mode,
            output_dir=output_dir,
            base_url=base_url,
            extra=extra,
            headers=headers,
        )
        status = "PASS" if exit_code == 0 else "FAIL"
        print(f"  Result: {status}\n")
        all_exit_codes.append(exit_code)

        if i < total:
            print(f"  Waiting {DELAY_BETWEEN_RUNS}s before next run...")
            time.sleep(DELAY_BETWEEN_RUNS)

    summary = generate_summary(date_str, config, results_dir)
    summary_path = results_dir / "summary.md"
    summary_path.write_text(summary)
    print(f"Summary written to {summary_path}\n")
    print(summary)

    config_path = results_dir / "weekly_matrix.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    print(f"Resolved config written to {config_path}")

    return 0 if all(c == 0 for c in all_exit_codes) else 1


if __name__ == "__main__":
    sys.exit(main())
