from __future__ import annotations

import sys
from datetime import datetime, timezone
from typing import TextIO

from granite_validation.domain.models import EngineInfo, Report, TestResult
from granite_validation.reporting.base import AbstractReporter


class ConsoleReporter(AbstractReporter):
    """Prints a formatted validation report to the terminal."""

    def __init__(self, *, verbose: bool = False) -> None:
        self._verbose = verbose
        self._failed_results: list[TestResult] = []

    def on_run_start(self, info: EngineInfo, model: str) -> None:
        out = sys.stdout
        out.write("\n")
        out.write("=" * 60 + "\n")
        out.write("  Granite Validation Report\n")
        out.write("=" * 60 + "\n")
        out.write(f"  Engine:    {info.engine_id} ({info.mode})\n")
        out.write(f"  Base URL:  {info.base_url}\n")
        out.write(f"  Model:     {model}\n")
        out.write(f"  Timestamp: {datetime.now(timezone.utc).isoformat()}\n")
        out.write("-" * 60 + "\n")
        out.flush()

    def on_test_complete(self, result: TestResult) -> None:
        out = sys.stdout
        status = "PASS" if result.passed else "FAIL"
        if self._verbose:
            out.write(f"  [{status}] {result.test_name} ({result.elapsed_seconds:.2f}s)\n")
            self._write_check_details(out, result)
        else:
            passed_checks = sum(1 for c in result.checks if c.passed)
            total_checks = len(result.checks)
            out.write(
                f"  [{status}] {result.test_name}"
                f" ({passed_checks}/{total_checks})"
                f" ({result.elapsed_seconds:.2f}s)\n"
            )
        if not result.passed:
            self._failed_results.append(result)
        out.flush()

    @staticmethod
    def _write_check_details(
        out: TextIO, result: TestResult, *, show_passing: bool = True
    ) -> None:
        for check in result.checks:
            if check.passed and not show_passing:
                continue
            mark = "+" if check.passed else "-"
            out.write(f"    [{mark}] {check.name}\n")
            if not check.passed and check.detail:
                out.write(f"        {check.detail}\n")
        if result.error:
            out.write(f"    ERROR: {result.error}\n")

    def report(self, report: Report) -> None:
        out = sys.stdout
        passed = sum(1 for r in report.results if r.passed)
        failed = len(report.results) - passed

        if self._failed_results:
            out.write("\n")
            out.write("-" * 60 + "\n")
            out.write("  Failed Tests\n")
            out.write("-" * 60 + "\n")
            for result in self._failed_results:
                passed_checks = sum(1 for c in result.checks if c.passed)
                total_checks = len(result.checks)
                out.write(
                    f"  [FAIL] {result.test_name}"
                    f" ({passed_checks}/{total_checks})"
                    f" ({result.elapsed_seconds:.2f}s)\n"
                )
                self._write_check_details(
                    out, result, show_passing=False
                )

        if report.lifecycle_error:
            out.write(f"  LIFECYCLE ERROR: {report.lifecycle_error}\n")
            out.write("-" * 60 + "\n")

        if report.abort_reason:
            out.write(f"  ABORTED: {report.abort_reason}\n")

        if report.cleanup_error:
            out.write(f"  WARNING: {report.cleanup_error}\n")
        out.write("-" * 60 + "\n")
        outcome = "FAIL" if report.lifecycle_error else ("PASS" if report.all_passed else "FAIL")
        out.write(f"  Result: {outcome}\n")
        out.write(f"  Total: {len(report.results)} | Passed: {passed} | Failed: {failed}\n")
        out.write(f"  Elapsed: {report.total_elapsed_seconds:.2f}s\n")
        out.write("=" * 60 + "\n\n")
