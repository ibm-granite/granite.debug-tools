from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from runtimes_validator.engines.base import EngineConfig
from runtimes_validator.engines.registry import create_engine, list_engines

# Import engine modules to trigger registration
import runtimes_validator.engines.ollama  # noqa: F401
import runtimes_validator.engines.vllm  # noqa: F401
import runtimes_validator.engines.llamacpp  # noqa: F401

from runtimes_validator.reporting.console import ConsoleReporter
from runtimes_validator.reporting.inspection import InspectionLogger
from runtimes_validator.reporting.json_reporter import JsonReporter
from runtimes_validator.runner import ValidationRunner
from runtimes_validator.tests.registry import (
    discover_tests,
    get_test_by_id,
    get_tests,
    list_tests,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="runtimes-validator",
        description="Unified validation framework for Granite models across inference engines",
    )
    parser.add_argument(
        "--engine",
        choices=list_engines(),
        help="Inference engine to validate against",
    )
    parser.add_argument(
        "--model",
        help="Model identifier (e.g. granite3.3:8b, ibm-granite/granite-3.3-8b-instruct)",
    )
    parser.add_argument(
        "--mode",
        choices=["managed", "external"],
        default="external",
        help="Engine execution mode (default: external)",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Base URL for external mode (uses engine default if not set)",
    )
    parser.add_argument(
        "--tests",
        default=None,
        help="Comma-separated list of test IDs to run (default: all applicable)",
    )
    parser.add_argument(
        "--modalities",
        default="text",
        help=(
            "Comma-separated input modalities to validate: 'text', 'vision', 'speech' "
            "(default: text). Ignored when --tests is used."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for JSON report output",
    )
    parser.add_argument(
        "--header",
        action="append",
        default=[],
        metavar="'Key: Value'",
        help="Custom HTTP header (repeatable, e.g. --header 'Authorization: Bearer TOKEN')",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Show full check details for every test (default: summary + failed tests only)",
    )
    parser.add_argument(
        "--extra",
        default=None,
        metavar="JSON",
        help=(
            "Engine-specific config as a JSON object, merged into EngineConfig.extra "
            "(e.g. '{\"max_model_len\": 4096}')"
        ),
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        default=False,
        help="Log raw JSON request/response pairs as JSONL for post-hoc inspection",
    )
    parser.add_argument(
        "--inspection-log",
        default=None,
        metavar="PATH",
        help=(
            "Path to the JSONL inspection log (default: "
            "inspection_{engine}_{model}_{timestamp}.jsonl in the current directory). "
            "Requires --inspect."
        ),
    )
    parser.add_argument(
        "--list-tests",
        action="store_true",
        help="List available tests and exit",
    )
    parser.add_argument(
        "--list-engines",
        action="store_true",
        help="List available engines and exit",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    discover_tests()
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_engines:
        for engine_id in list_engines():
            print(engine_id)
        return 0

    if args.list_tests:
        for test_id in list_tests():
            mods = ",".join(get_test_by_id(test_id)().modalities())
            print(f"{test_id}\t[{mods}]")
        return 0

    if not args.engine:
        parser.error("--engine is required for validation runs")
    if not args.model:
        parser.error("--model is required for validation runs")

    headers: dict[str, str] = {}
    for h in args.header:
        key, _, value = h.partition(":")
        if not value:
            parser.error(f"Invalid header format: {h!r} (expected 'Key: Value')")
        headers[key.strip()] = value.strip()

    extra: dict[str, object] = {}
    if args.extra:
        try:
            parsed_extra = json.loads(args.extra)
        except json.JSONDecodeError as exc:
            parser.error(f"--extra must be valid JSON: {exc}")
        if not isinstance(parsed_extra, dict):
            parser.error("--extra must be a JSON object")
        extra.update(parsed_extra)
    if headers:
        extra["headers"] = headers

    inspection_logger: InspectionLogger | None = None
    if args.inspection_log and not args.inspect:
        parser.error("--inspection-log requires --inspect")
    if args.inspect:
        if args.inspection_log:
            log_path = Path(args.inspection_log)
            if log_path.suffix.lower() != ".jsonl":
                print(
                    f"warning: --inspection-log '{log_path}' does not end in "
                    ".jsonl; the file will still be written as JSON Lines "
                    "(one JSON object per line).",
                    file=sys.stderr,
                )
        else:
            timestamp = datetime.now(timezone.utc).isoformat()
            filename = f"inspection_{args.engine}_{args.model}_{timestamp}.jsonl"
            filename = filename.replace("/", "_").replace(":", "-").replace(" ", "_")
            log_path = Path(filename)
        inspection_logger = InspectionLogger(path=log_path)
        extra["inspection_logger"] = inspection_logger

    config = EngineConfig(
        mode=args.mode,
        base_url=args.base_url,
        model_id=args.model,
        extra=extra,
    )
    engine = create_engine(args.engine, config)

    if args.tests:
        test_ids = [t.strip() for t in args.tests.split(",")]
        test_classes = [get_test_by_id(tid) for tid in test_ids]
    else:
        modality_set = {m.strip() for m in args.modalities.split(",") if m.strip()}
        test_classes = get_tests(engine_id=args.engine, modalities=modality_set)

    tests = [cls() for cls in test_classes]

    reporters = [ConsoleReporter(verbose=args.verbose)]
    if args.output_dir:
        reporters.append(JsonReporter(output_dir=args.output_dir))

    runner = ValidationRunner(
        engine=engine,
        model=args.model,
        tests=tests,
        reporters=reporters,
    )

    try:
        report = runner.run()
    finally:
        if inspection_logger is not None:
            inspection_logger.close()
    return 0 if report.all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
