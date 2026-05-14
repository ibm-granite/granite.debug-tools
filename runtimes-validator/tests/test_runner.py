from __future__ import annotations

from typing import Any

from runtimes_validator.domain.models import CheckResult, EngineInfo, Report, TestResult
from runtimes_validator.engines.base import AbstractEngine, EngineConfig
from runtimes_validator.reporting.base import AbstractReporter
from runtimes_validator.runner import ValidationRunner
from runtimes_validator.tests.base import AbstractValidationTest


class MockEngine(AbstractEngine):
    """A mock engine that tracks lifecycle calls."""

    def __init__(
        self,
        config: EngineConfig | None = None,
        *,
        healthy: bool = True,
        start_error: Exception | None = None,
    ) -> None:
        self._config = config or EngineConfig()
        self._healthy = healthy
        self._start_error = start_error
        self.started = False
        self.stopped = False
        self.health_checked = False

    def engine_id(self) -> str:
        return "mock"

    def start(self, model: str) -> None:
        if self._start_error:
            raise self._start_error
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> dict[str, Any]:
        return {
            "role": "assistant",
            "content": "The capital of France is Paris. 391.",
            "tool_calls": None,
            "finish_reason": "stop",
            "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
        }

    def health_check(self) -> bool:
        self.health_checked = True
        return self._healthy

    def get_info(self) -> EngineInfo:
        return EngineInfo(
            engine_id="mock",
            version="1.0",
            mode=self._config.mode,
            base_url="http://mock",
        )


class PassingTest(AbstractValidationTest):
    def test_id(self) -> str:
        return "passing"

    def test_name(self) -> str:
        return "Always Passes"

    def run(self, engine: AbstractEngine, model: str) -> TestResult:
        return TestResult(
            test_id="passing",
            test_name="Always Passes",
            engine_id=engine.engine_id(),
            model=model,
            checks=[CheckResult(name="check_1", passed=True)],
            elapsed_seconds=0.01,
        )


class FailingTest(AbstractValidationTest):
    def test_id(self) -> str:
        return "failing"

    def test_name(self) -> str:
        return "Always Fails"

    def run(self, engine: AbstractEngine, model: str) -> TestResult:
        return TestResult(
            test_id="failing",
            test_name="Always Fails",
            engine_id=engine.engine_id(),
            model=model,
            checks=[CheckResult(name="check_1", passed=False, expected="yes", actual="no")],
            elapsed_seconds=0.01,
        )


class CollectingReporter(AbstractReporter):
    def __init__(self) -> None:
        self.reports: list = []

    def report(self, report) -> None:
        self.reports.append(report)


# --- Basic execution tests ---


def test_runner_all_pass():
    engine = MockEngine()
    reporter = CollectingReporter()
    runner = ValidationRunner(
        engine=engine, model="test-model", tests=[PassingTest()], reporters=[reporter]
    )
    report = runner.run()
    assert report.all_passed
    assert len(report.results) == 1
    assert len(reporter.reports) == 1


def test_runner_with_failure():
    engine = MockEngine()
    runner = ValidationRunner(
        engine=engine, model="test-model", tests=[PassingTest(), FailingTest()]
    )
    report = runner.run()
    assert not report.all_passed
    assert len(report.results) == 2


def test_runner_handles_test_exception():
    class CrashingTest(AbstractValidationTest):
        def test_id(self) -> str:
            return "crash"

        def test_name(self) -> str:
            return "Crashes"

        def run(self, engine: AbstractEngine, model: str) -> TestResult:
            raise RuntimeError("boom")

    engine = MockEngine()
    runner = ValidationRunner(engine=engine, model="test-model", tests=[CrashingTest()])
    report = runner.run()
    assert not report.all_passed
    assert report.results[0].error == "boom"


# --- Lifecycle / mode tests ---


def test_external_mode_never_calls_start_stop():
    engine = MockEngine(config=EngineConfig(mode="external"))
    runner = ValidationRunner(engine=engine, model="m", tests=[PassingTest()])
    runner.run()
    assert not engine.started
    assert not engine.stopped
    assert engine.health_checked


def test_managed_mode_calls_start_and_stop():
    engine = MockEngine(config=EngineConfig(mode="managed"))
    runner = ValidationRunner(engine=engine, model="m", tests=[PassingTest()])
    runner.run()
    assert engine.started
    assert engine.stopped
    assert engine.health_checked


def test_managed_mode_stops_even_on_test_failure():
    class CrashingTest(AbstractValidationTest):
        def test_id(self) -> str:
            return "crash"

        def test_name(self) -> str:
            return "Crashes"

        def run(self, engine: AbstractEngine, model: str) -> TestResult:
            raise RuntimeError("boom")

    engine = MockEngine(config=EngineConfig(mode="managed"))
    runner = ValidationRunner(engine=engine, model="m", tests=[CrashingTest()])
    report = runner.run()
    assert engine.started
    assert engine.stopped
    assert not report.all_passed


# --- Lifecycle failure produces structured report ---


def test_external_unhealthy_produces_report():
    engine = MockEngine(config=EngineConfig(mode="external"), healthy=False)
    reporter = CollectingReporter()
    runner = ValidationRunner(engine=engine, model="m", tests=[PassingTest()], reporters=[reporter])
    report = runner.run()
    assert not report.all_passed
    assert report.lifecycle_error is not None
    assert "not reachable" in report.lifecycle_error
    assert len(report.results) == 0
    assert len(reporter.reports) == 1


def test_managed_unhealthy_produces_report_and_stops():
    engine = MockEngine(config=EngineConfig(mode="managed"), healthy=False)
    reporter = CollectingReporter()
    runner = ValidationRunner(engine=engine, model="m", tests=[PassingTest()], reporters=[reporter])
    report = runner.run()
    assert not report.all_passed
    assert report.lifecycle_error is not None
    assert "failed health check" in report.lifecycle_error
    assert engine.started
    assert engine.stopped
    assert len(reporter.reports) == 1


def test_managed_start_failure_produces_report():
    engine = MockEngine(
        config=EngineConfig(mode="managed"),
        start_error=RuntimeError("port in use"),
    )
    reporter = CollectingReporter()
    runner = ValidationRunner(engine=engine, model="m", tests=[PassingTest()], reporters=[reporter])
    report = runner.run()
    assert not report.all_passed
    assert report.lifecycle_error is not None
    assert "port in use" in report.lifecycle_error
    assert len(reporter.reports) == 1


def test_managed_start_failure_calls_stop_for_cleanup():
    engine = MockEngine(
        config=EngineConfig(mode="managed"),
        start_error=RuntimeError("timeout"),
    )
    runner = ValidationRunner(engine=engine, model="m", tests=[PassingTest()])
    runner.run()
    assert engine.stopped


def test_managed_stop_failure_still_returns_report():
    """If stop() raises during cleanup, report includes cleanup_error and fails the run."""

    class StopFailsEngine(MockEngine):
        def stop(self) -> None:
            self.stopped = True
            raise RuntimeError("stop failed")

    engine = StopFailsEngine(config=EngineConfig(mode="managed"))
    reporter = CollectingReporter()
    runner = ValidationRunner(engine=engine, model="m", tests=[PassingTest()], reporters=[reporter])
    report = runner.run()
    assert not report.all_passed
    assert report.cleanup_error is not None
    assert "stop failed" in report.cleanup_error
    assert reporter.reports[0].cleanup_error == report.cleanup_error
    assert len(reporter.reports) == 1
    assert engine.stopped


def test_get_info_failure_before_start_produces_report():
    """If get_info() raises before start, runner returns a structured report."""

    class BrokenInfoEngine(MockEngine):
        def get_info(self) -> EngineInfo:
            raise RuntimeError("info unavailable before start")

    engine = BrokenInfoEngine(config=EngineConfig(mode="managed"))
    reporter = CollectingReporter()
    runner = ValidationRunner(engine=engine, model="m", tests=[PassingTest()], reporters=[reporter])
    report = runner.run()
    assert not report.all_passed
    assert report.lifecycle_error is not None
    assert "get_info()" in report.lifecycle_error
    assert report.engine_info.mode == "unknown"
    assert not engine.started
    assert not engine.stopped
    assert len(reporter.reports) == 1


def test_mode_read_from_engine_config():
    """Runner reads mode from engine.get_info().mode, not a separate parameter."""
    managed_engine = MockEngine(config=EngineConfig(mode="managed"))
    runner = ValidationRunner(engine=managed_engine, model="m", tests=[PassingTest()])
    runner.run()
    assert managed_engine.started
    assert managed_engine.stopped

    external_engine = MockEngine(config=EngineConfig(mode="external"))
    runner = ValidationRunner(engine=external_engine, model="m", tests=[PassingTest()])
    runner.run()
    assert not external_engine.started
    assert not external_engine.stopped


def test_managed_report_uses_post_start_engine_info():
    """Managed runs should report metadata discovered during start()."""

    class DynamicInfoEngine(MockEngine):
        def __init__(self, config: EngineConfig) -> None:
            super().__init__(config=config)
            self._base_url = "http://before"

        def start(self, model: str) -> None:
            self.started = True
            self._base_url = "http://after"

        def get_info(self) -> EngineInfo:
            return EngineInfo(
                engine_id="mock",
                version="1.0",
                mode=self._config.mode,
                base_url=self._base_url,
            )

    engine = DynamicInfoEngine(config=EngineConfig(mode="managed"))
    runner = ValidationRunner(engine=engine, model="m", tests=[PassingTest()])
    report = runner.run()
    assert report.engine_info.base_url == "http://after"


# --- Streaming / on_test_complete tests ---


class StreamCollectingReporter(AbstractReporter):
    def __init__(self) -> None:
        self.reports: list[Report] = []
        self.streamed_results: list[TestResult] = []
        self.run_start_calls: list[tuple[EngineInfo, str]] = []

    def report(self, report: Report) -> None:
        self.reports.append(report)

    def on_run_start(self, info: EngineInfo, model: str) -> None:
        self.run_start_calls.append((info, model))

    def on_test_complete(self, result: TestResult) -> None:
        self.streamed_results.append(result)


class BombReporter(AbstractReporter):
    """Reporter that raises from every hook."""

    def report(self, report: Report) -> None:
        pass

    def on_run_start(self, info: EngineInfo, model: str) -> None:
        raise RuntimeError("boom from on_run_start")

    def on_test_complete(self, result: TestResult) -> None:
        raise RuntimeError("boom from on_test_complete")


def test_on_test_complete_called_per_test():
    engine = MockEngine()
    reporter = StreamCollectingReporter()
    runner = ValidationRunner(
        engine=engine,
        model="test-model",
        tests=[PassingTest(), FailingTest()],
        reporters=[reporter],
    )
    runner.run()
    assert len(reporter.streamed_results) == 2
    assert reporter.streamed_results[0].test_id == "passing"
    assert reporter.streamed_results[1].test_id == "failing"
    assert len(reporter.reports) == 1


def test_bombing_reporter_does_not_abort_run():
    """A reporter that raises from streaming hooks must not prevent other reporters or the run."""
    engine = MockEngine()
    bomb = BombReporter()
    good = StreamCollectingReporter()
    runner = ValidationRunner(
        engine=engine,
        model="m",
        tests=[PassingTest(), FailingTest()],
        reporters=[bomb, good],
    )
    report = runner.run()
    assert len(report.results) == 2
    assert len(good.run_start_calls) == 1
    assert len(good.streamed_results) == 2
    assert len(good.reports) == 1


def test_on_run_start_called_on_lifecycle_failure():
    """on_run_start fires even when health check fails and no tests execute."""
    engine = MockEngine(config=EngineConfig(mode="external"), healthy=False)
    reporter = StreamCollectingReporter()
    runner = ValidationRunner(
        engine=engine,
        model="m",
        tests=[PassingTest()],
        reporters=[reporter],
    )
    report = runner.run()
    assert report.lifecycle_error is not None
    assert len(reporter.run_start_calls) == 1
    assert reporter.run_start_calls[0][0].engine_id == "mock"
    assert len(reporter.streamed_results) == 0
    assert len(reporter.reports) == 1


def test_on_run_start_called_on_managed_start_failure():
    """on_run_start fires even when engine start() fails."""
    engine = MockEngine(
        config=EngineConfig(mode="managed"),
        start_error=RuntimeError("port in use"),
    )
    reporter = StreamCollectingReporter()
    runner = ValidationRunner(
        engine=engine,
        model="m",
        tests=[PassingTest()],
        reporters=[reporter],
    )
    report = runner.run()
    assert report.lifecycle_error is not None
    assert len(reporter.run_start_calls) == 1
    assert len(reporter.reports) == 1


# --- Abort on engine timeout ---


class TimeoutEngine(MockEngine):
    """Engine that sets _last_timeout on the first chat() call."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._last_timeout = False

    def chat(self, messages, **kwargs):
        self._last_timeout = True
        raise Exception("Request timed out after 120s")


class ChatCallingTest(AbstractValidationTest):
    """Test that calls engine.chat() — will trigger TimeoutEngine."""

    def __init__(self, label: str = "chat_test") -> None:
        self._label = label

    def test_id(self) -> str:
        return self._label

    def test_name(self) -> str:
        return self._label

    def run(self, engine: AbstractEngine, model: str) -> TestResult:
        try:
            engine.chat([{"role": "user", "content": "hi"}])
        except Exception as e:
            return TestResult(
                test_id=self._label,
                test_name=self._label,
                engine_id=engine.engine_id(),
                model=model,
                checks=[CheckResult(name="error", passed=False, detail=str(e))],
                elapsed_seconds=120.0,
            )
        return TestResult(
            test_id=self._label,
            test_name=self._label,
            engine_id=engine.engine_id(),
            model=model,
            checks=[CheckResult(name="ok", passed=True)],
            elapsed_seconds=0.01,
        )


class StickyTimeoutEngine(MockEngine):
    """Engine that remembers a timeout even if a later request succeeds."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._timeout_observed = False
        self._calls = 0

    def reset_timeout_observed(self) -> None:
        self._timeout_observed = False

    def timed_out_since_last_check(self) -> bool:
        return self._timeout_observed

    def chat(self, messages, **kwargs):
        self._calls += 1
        if self._calls == 1:
            self._timeout_observed = True
            raise Exception("Request timed out after 120s")
        return super().chat(messages, **kwargs)


class TimeoutThenSuccessTest(AbstractValidationTest):
    def test_id(self) -> str:
        return "timeout_then_success"

    def test_name(self) -> str:
        return "Timeout Then Success"

    def run(self, engine: AbstractEngine, model: str) -> TestResult:
        checks: list[CheckResult] = []
        try:
            engine.chat([{"role": "user", "content": "first"}])
        except Exception as e:
            checks.append(CheckResult(name="first_error", passed=False, detail=str(e)))
        response = engine.chat([{"role": "user", "content": "second"}])
        checks.append(CheckResult(name="second_success", passed=bool(response.get("content"))))
        return TestResult(
            test_id=self.test_id(),
            test_name=self.test_name(),
            engine_id=engine.engine_id(),
            model=model,
            checks=checks,
            elapsed_seconds=0.01,
        )


def test_runner_aborts_on_engine_timeout():
    """When the engine sets _last_timeout, remaining tests are skipped."""
    engine = TimeoutEngine()
    tests = [ChatCallingTest("timeout_test"), PassingTest()]
    runner = ValidationRunner(engine=engine, model="m", tests=tests)

    report = runner.run()

    assert report.abort_reason is not None
    assert "timeout_test" in report.abort_reason
    assert len(report.results) == 1


def test_runner_abort_sets_all_passed_false():
    engine = TimeoutEngine()
    runner = ValidationRunner(engine=engine, model="m", tests=[ChatCallingTest()])

    report = runner.run()

    assert not report.all_passed
    assert report.abort_reason is not None


def test_runner_aborts_when_timeout_is_followed_by_success_in_same_test():
    engine = StickyTimeoutEngine()
    tests = [TimeoutThenSuccessTest(), PassingTest()]
    runner = ValidationRunner(engine=engine, model="m", tests=tests)

    report = runner.run()

    assert report.abort_reason is not None
    assert "Timeout Then Success" in report.abort_reason
    assert len(report.results) == 1


def test_runner_no_abort_without_timeout():
    engine = MockEngine()
    runner = ValidationRunner(engine=engine, model="m", tests=[PassingTest(), FailingTest()])

    report = runner.run()

    assert report.abort_reason is None
    assert len(report.results) == 2


# --- Tool test preflight check (vLLM server_args) ---


class MockVllmEngine(MockEngine):
    def engine_id(self) -> str:
        return "vllm"

    def get_info(self) -> EngineInfo:
        return EngineInfo(
            engine_id="vllm",
            version="1.0",
            mode=self._config.mode,
            base_url="http://mock",
        )


class FakeToolTest(AbstractValidationTest):
    def test_id(self) -> str:
        return "tool_calling"

    def test_name(self) -> str:
        return "Tool Calling"

    def run(self, engine: AbstractEngine, model: str) -> TestResult:
        return TestResult(
            test_id="tool_calling",
            test_name="Tool Calling",
            engine_id=engine.engine_id(),
            model=model,
            checks=[CheckResult(name="check_1", passed=True)],
            elapsed_seconds=0.01,
        )


def test_vllm_managed_tool_test_no_server_args_blocks():
    engine = MockVllmEngine(config=EngineConfig(mode="managed"))
    runner = ValidationRunner(engine=engine, model="m", tests=[FakeToolTest()])
    report = runner.run()
    assert report.lifecycle_error is not None
    assert "--enable-auto-tool-choice" in report.lifecycle_error
    assert "--tool-call-parser" in report.lifecycle_error
    assert not engine.started


def test_vllm_managed_tool_test_with_both_flags_runs():
    engine = MockVllmEngine(
        config=EngineConfig(
            mode="managed",
            extra={
                "server_args": [
                    "--enable-auto-tool-choice",
                    "--tool-call-parser",
                    "granite",
                ]
            },
        )
    )
    runner = ValidationRunner(engine=engine, model="m", tests=[FakeToolTest()])
    report = runner.run()
    assert report.lifecycle_error is None or "tool" not in report.lifecycle_error.lower()
    assert engine.started


def test_vllm_managed_tool_test_missing_one_flag_blocks():
    engine = MockVllmEngine(
        config=EngineConfig(
            mode="managed",
            extra={"server_args": ["--enable-auto-tool-choice"]},
        )
    )
    runner = ValidationRunner(engine=engine, model="m", tests=[FakeToolTest()])
    report = runner.run()
    assert report.lifecycle_error is not None
    assert "missing required flags: --tool-call-parser" in report.lifecycle_error
    assert not engine.started


def test_vllm_managed_no_tool_tests_no_block():
    engine = MockVllmEngine(config=EngineConfig(mode="managed"))
    runner = ValidationRunner(engine=engine, model="m", tests=[PassingTest()])
    report = runner.run()
    assert report.lifecycle_error is None
    assert engine.started


def test_vllm_external_tool_test_no_flags_no_block():
    engine = MockVllmEngine(config=EngineConfig(mode="external"))
    runner = ValidationRunner(engine=engine, model="m", tests=[FakeToolTest()])
    report = runner.run()
    assert report.lifecycle_error is None


def test_non_vllm_engine_tool_test_no_flags_no_block():
    engine = MockEngine(config=EngineConfig(mode="managed"))
    runner = ValidationRunner(engine=engine, model="m", tests=[FakeToolTest()])
    report = runner.run()
    assert report.lifecycle_error is None
    assert engine.started


# --- Modality-gated test skipping ---


class VisionTest(AbstractValidationTest):
    """Declares vision modality and records whether run() was invoked."""

    def __init__(self) -> None:
        self.ran = False

    def test_id(self) -> str:
        return "vision_test"

    def test_name(self) -> str:
        return "Vision Test"

    def modalities(self) -> list[str]:
        return ["vision"]

    def run(self, engine: AbstractEngine, model: str) -> TestResult:
        self.ran = True
        return TestResult(
            test_id=self.test_id(),
            test_name=self.test_name(),
            engine_id=engine.engine_id(),
            model=model,
            checks=[CheckResult(name="ok", passed=True)],
            elapsed_seconds=0.01,
        )


class TextOnlyEngine(MockEngine):
    def supported_modalities(self) -> set[str]:
        return {"text"}


class TextAndVisionEngine(MockEngine):
    def supported_modalities(self) -> set[str]:
        return {"text", "vision"}


def test_runner_skips_test_when_engine_missing_modality():
    engine = TextOnlyEngine()
    test = VisionTest()
    runner = ValidationRunner(engine=engine, model="m", tests=[test])
    report = runner.run()

    assert report.all_passed is False
    assert report.abort_reason == "No tests were executed; all selected tests were skipped."
    assert len(report.results) == 1
    result = report.results[0]
    assert result.skipped is True
    assert result.skip_reason is not None
    assert "vision" in result.skip_reason
    assert not test.ran


def test_runner_runs_test_when_engine_has_modality():
    engine = TextAndVisionEngine()
    test = VisionTest()
    runner = ValidationRunner(engine=engine, model="m", tests=[test])
    report = runner.run()

    assert len(report.results) == 1
    assert report.results[0].skipped is False
    assert test.ran


def test_runner_mixed_skipped_and_passed_tests_exits_clean():
    engine = TextOnlyEngine()
    runner = ValidationRunner(engine=engine, model="m", tests=[PassingTest(), VisionTest()])
    report = runner.run()

    assert report.all_passed is True
    assert report.abort_reason is None
    assert len(report.results) == 2
    statuses = {r.test_id: r.status for r in report.results}
    assert statuses["passing"] == "pass"
    assert statuses["vision_test"] == "skip"


def test_runner_no_tests_selected_fails_report():
    engine = TextOnlyEngine()
    runner = ValidationRunner(engine=engine, model="m", tests=[])
    report = runner.run()

    assert report.all_passed is False
    assert report.results == []
    assert report.abort_reason == "No tests were selected."


def test_runner_skip_streams_to_reporter():
    engine = TextOnlyEngine()
    reporter = StreamCollectingReporter()
    runner = ValidationRunner(engine=engine, model="m", tests=[VisionTest()], reporters=[reporter])
    runner.run()

    assert len(reporter.streamed_results) == 1
    assert reporter.streamed_results[0].skipped is True
