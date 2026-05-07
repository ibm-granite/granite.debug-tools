from __future__ import annotations

from runtimes_validator.domain.models import (
    CheckResult,
    EngineInfo,
    Report,
    TestResult,
)
from runtimes_validator.reporting.console import ConsoleReporter


def _make_engine_info() -> EngineInfo:
    return EngineInfo(
        engine_id="mock",
        version="1.0",
        mode="external",
        base_url="http://mock",
    )


def _make_passing_result() -> TestResult:
    return TestResult(
        test_id="pass_test",
        test_name="Passing Test",
        engine_id="mock",
        model="m",
        checks=[CheckResult(name="check_ok", passed=True)],
        elapsed_seconds=0.50,
    )


def _make_failing_result() -> TestResult:
    return TestResult(
        test_id="fail_test",
        test_name="Failing Test",
        engine_id="mock",
        model="m",
        checks=[
            CheckResult(name="check_ok", passed=True),
            CheckResult(name="check_bad", passed=False, detail="expected foo got bar"),
        ],
        elapsed_seconds=1.23,
    )


def _make_error_result() -> TestResult:
    return TestResult(
        test_id="err_test",
        test_name="Error Test",
        engine_id="mock",
        model="m",
        checks=[],
        elapsed_seconds=0.01,
        error="connection refused",
    )


def _make_skipped_result() -> TestResult:
    return TestResult(
        test_id="skip_test",
        test_name="Skipped Test",
        engine_id="mock",
        model="m",
        checks=[],
        elapsed_seconds=0.0,
        skipped=True,
        skip_reason="engine 'mock' does not support modality/modalities: speech",
    )


def _make_report(results: list[TestResult]) -> Report:
    return Report(
        engine_info=_make_engine_info(),
        model="m",
        results=results,
        timestamp="2026-01-01T00:00:00Z",
        total_elapsed_seconds=2.0,
    )


class TestDefaultMode:
    def test_passing_test_shows_one_line_with_check_counts(self, capsys):
        reporter = ConsoleReporter()
        result = _make_passing_result()
        reporter.on_test_complete(result)
        output = capsys.readouterr().out
        assert "[PASS] Passing Test (1/1)" in output
        assert "[+]" not in output

    def test_failing_test_shows_one_line_with_check_counts(self, capsys):
        reporter = ConsoleReporter()
        result = _make_failing_result()
        reporter.on_test_complete(result)
        output = capsys.readouterr().out
        assert "[FAIL] Failing Test (1/2)" in output
        assert "[-]" not in output
        assert "check_bad" not in output

    def test_failed_details_shown_in_report(self, capsys):
        reporter = ConsoleReporter()
        failing = _make_failing_result()
        passing = _make_passing_result()
        reporter.on_test_complete(passing)
        reporter.on_test_complete(failing)
        capsys.readouterr()

        report = _make_report([passing, failing])
        reporter.report(report)
        output = capsys.readouterr().out
        assert "Failed Tests" in output
        assert "[FAIL] Failing Test (1/2)" in output
        assert "[-] check_bad" in output
        assert "expected foo got bar" in output
        failed_section = output.split("Failed Tests")[1].split("Result:")[0]
        assert "Passing Test" not in failed_section
        assert "[+]" not in failed_section

    def test_error_test_hides_details_during_streaming(self, capsys):
        reporter = ConsoleReporter()
        result = _make_error_result()
        reporter.on_test_complete(result)
        output = capsys.readouterr().out
        assert "[FAIL] Error Test" in output
        assert "ERROR:" not in output
        assert "connection refused" not in output

    def test_error_details_shown_in_report(self, capsys):
        reporter = ConsoleReporter()
        err = _make_error_result()
        reporter.on_test_complete(err)
        capsys.readouterr()

        report = _make_report([err])
        reporter.report(report)
        output = capsys.readouterr().out
        assert "Failed Tests" in output
        assert "ERROR: connection refused" in output

    def test_all_pass_no_failed_section(self, capsys):
        reporter = ConsoleReporter()
        passing = _make_passing_result()
        reporter.on_test_complete(passing)
        capsys.readouterr()

        report = _make_report([passing])
        reporter.report(report)
        output = capsys.readouterr().out
        assert "Failed Tests" not in output
        assert "Result: PASS" in output

    def test_summary_counts(self, capsys):
        reporter = ConsoleReporter()
        passing = _make_passing_result()
        failing = _make_failing_result()
        reporter.on_test_complete(passing)
        reporter.on_test_complete(failing)
        capsys.readouterr()

        report = _make_report([passing, failing])
        reporter.report(report)
        output = capsys.readouterr().out
        assert "Total: 2 | Passed: 1 | Failed: 1" in output

    def test_skipped_test_shows_skip_line_with_reason(self, capsys):
        reporter = ConsoleReporter()
        result = _make_skipped_result()
        reporter.on_test_complete(result)
        output = capsys.readouterr().out
        assert "[SKIP] Skipped Test" in output
        assert "speech" in output

    def test_skipped_test_not_listed_in_failed_section(self, capsys):
        reporter = ConsoleReporter()
        passing = _make_passing_result()
        skipped = _make_skipped_result()
        reporter.on_test_complete(passing)
        reporter.on_test_complete(skipped)
        capsys.readouterr()

        report = _make_report([passing, skipped])
        reporter.report(report)
        output = capsys.readouterr().out
        assert "Failed Tests" not in output
        assert "Result: PASS" in output

    def test_summary_counts_include_skipped(self, capsys):
        reporter = ConsoleReporter()
        passing = _make_passing_result()
        skipped = _make_skipped_result()
        reporter.on_test_complete(passing)
        reporter.on_test_complete(skipped)
        capsys.readouterr()

        report = _make_report([passing, skipped])
        reporter.report(report)
        output = capsys.readouterr().out
        assert "Total: 2 | Passed: 1 | Failed: 0 | Skipped: 1" in output


class TestVerboseMode:
    def test_passing_test_shows_check_details(self, capsys):
        reporter = ConsoleReporter(verbose=True)
        result = _make_passing_result()
        reporter.on_test_complete(result)
        output = capsys.readouterr().out
        assert "[PASS] Passing Test" in output
        assert "[+] check_ok" in output

    def test_failing_test_shows_check_details_inline(self, capsys):
        reporter = ConsoleReporter(verbose=True)
        result = _make_failing_result()
        reporter.on_test_complete(result)
        output = capsys.readouterr().out
        assert "[FAIL] Failing Test" in output
        assert "[+] check_ok" in output
        assert "[-] check_bad" in output
        assert "expected foo got bar" in output

    def test_failed_section_shown_in_report(self, capsys):
        reporter = ConsoleReporter(verbose=True)
        failing = _make_failing_result()
        reporter.on_test_complete(failing)
        capsys.readouterr()

        report = _make_report([failing])
        reporter.report(report)
        output = capsys.readouterr().out
        assert "Failed Tests" in output
        assert "[FAIL] Failing Test (1/2)" in output
        assert "[-] check_bad" in output
        assert "expected foo got bar" in output
        failed_section = output.split("Failed Tests")[1].split("Result:")[0]
        assert "[+]" not in failed_section

    def test_error_shown_inline(self, capsys):
        reporter = ConsoleReporter(verbose=True)
        err = _make_error_result()
        reporter.on_test_complete(err)
        output = capsys.readouterr().out
        assert "ERROR: connection refused" in output
