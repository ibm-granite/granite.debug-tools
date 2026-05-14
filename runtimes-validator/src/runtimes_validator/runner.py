from __future__ import annotations

import time
from datetime import datetime, timezone

from runtimes_validator.domain.models import EngineInfo, Report, TestResult
from runtimes_validator.engines.base import AbstractEngine
from runtimes_validator.reporting.base import AbstractReporter
from runtimes_validator.tests.base import AbstractValidationTest


class ValidationRunner:
    """Orchestrates test execution against an engine and produces reports.

    The runner owns lifecycle policy based on the engine's configured mode:
    - managed: start(model) -> health_check -> run tests -> stop (in finally)
    - external: health_check -> run tests (never calls start/stop)

    Lifecycle failures (start/health_check) produce a structured Report with
    lifecycle_error set, rather than raising exceptions.
    """

    def __init__(
        self,
        engine: AbstractEngine,
        model: str,
        tests: list[AbstractValidationTest],
        reporters: list[AbstractReporter] | None = None,
    ) -> None:
        self._engine = engine
        self._model = model
        self._tests = tests
        self._reporters = reporters or []

    def run(self) -> Report:
        info = self._resolve_engine_info()

        if info.mode == "managed":
            report = self._run_managed(info)
        elif info.mode == "external":
            report = self._run_external(info)
        else:
            self._emit_run_start(info)
            report = self._build_report(
                info,
                lifecycle_error=f"Engine.get_info() failed: {info.extra.get('error', 'unknown')}",
            )

        self._emit_report(report)
        return report

    def _resolve_engine_info(self) -> EngineInfo:
        """Get engine info, returning a fallback on failure.

        A failed get_info() returns an EngineInfo with mode="unknown" and
        the error stashed in extra["error"] for downstream reporting.
        """
        try:
            return self._engine.get_info()
        except Exception as e:
            return EngineInfo(
                engine_id=self._safe_engine_id(),
                version="unknown",
                mode="unknown",
                base_url="unknown",
                extra={"error": str(e)},
            )

    def _safe_engine_id(self) -> str:
        try:
            return self._engine.engine_id()
        except Exception:
            return "unknown"

    def _refresh_engine_info(self, current_info: EngineInfo) -> EngineInfo:
        """Refresh metadata after lifecycle transitions, falling back to prior info."""
        refreshed_info = self._resolve_engine_info()
        if refreshed_info.mode == "unknown":
            return current_info
        return refreshed_info

    def _check_tool_test_requirements(self) -> str | None:
        """Return a warning if tool tests need missing vLLM server flags, else None."""
        has_tool_tests = any(t.test_id() == "tool_calling" for t in self._tests)
        if not has_tool_tests:
            return None

        if self._engine.engine_id() != "vllm":
            return None

        config = getattr(self._engine, "_config", None)
        if config is None:
            return None

        server_args = config.extra.get("server_args", [])
        missing: list[str] = []
        if "--enable-auto-tool-choice" not in server_args:
            missing.append("--enable-auto-tool-choice")
        if "--tool-call-parser" not in server_args:
            missing.append("--tool-call-parser <parser>")

        if not missing:
            return None

        return (
            "Tool tests are selected but vLLM server_args is missing required flags: "
            f"{', '.join(missing)}. "
            'Add them via --extra \'{"server_args": ["--enable-auto-tool-choice", '
            '"--tool-call-parser", "<parser>"]}\' '
            "(e.g. --tool-call-parser granite for Granite models)."
        )

    def _run_managed(self, info: EngineInfo) -> Report:
        tool_warning = self._check_tool_test_requirements()
        if tool_warning:
            self._emit_run_start(info)
            return self._build_report(info, lifecycle_error=tool_warning)

        try:
            self._engine.start(self._model)
        except Exception as e:
            self._emit_run_start(info)
            cleanup_error = self._try_stop()
            return self._build_report(
                self._refresh_engine_info(info),
                lifecycle_error=f"Engine start failed: {e}",
                cleanup_error=cleanup_error,
            )

        managed_info = self._refresh_engine_info(info)
        self._emit_run_start(managed_info)

        try:
            try:
                healthy = self._engine.health_check()
            except Exception as e:
                report = self._build_report(
                    managed_info,
                    lifecycle_error=f"Health check error after start: {e}",
                )
                return report
            if not healthy:
                report = self._build_report(
                    managed_info,
                    lifecycle_error=(
                        f"Engine '{self._engine.engine_id()}' failed health check after start"
                    ),
                )
                return report
            report = self._execute_tests(managed_info)
            return report
        finally:
            cleanup_error = self._try_stop()
            if cleanup_error:
                report.cleanup_error = cleanup_error

    def _try_stop(self) -> str | None:
        """Attempt stop(); return the error message if it raises, else None."""
        try:
            self._engine.stop()
        except Exception as e:
            return f"Engine stop failed: {e}"
        return None

    def _run_external(self, info: EngineInfo) -> Report:
        self._emit_run_start(info)
        try:
            healthy = self._engine.health_check()
        except Exception as e:
            return self._build_report(
                info,
                lifecycle_error=f"Health check error: {e}",
            )
        if not healthy:
            return self._build_report(
                info,
                lifecycle_error=f"Engine '{self._engine.engine_id()}' is not reachable",
            )
        info = self._refresh_engine_info(info)
        return self._execute_tests(info)

    def _build_report(
        self,
        info: EngineInfo,
        *,
        results: list[TestResult] | None = None,
        total_elapsed_seconds: float = 0.0,
        lifecycle_error: str | None = None,
        cleanup_error: str | None = None,
        abort_reason: str | None = None,
    ) -> Report:
        return Report(
            engine_info=info,
            model=self._model,
            results=results or [],
            timestamp=datetime.now(timezone.utc).isoformat(),
            total_elapsed_seconds=total_elapsed_seconds,
            lifecycle_error=lifecycle_error,
            cleanup_error=cleanup_error,
            abort_reason=abort_reason,
        )

    def _emit_report(self, report: Report) -> None:
        for reporter in self._reporters:
            reporter.report(report)

    def _emit_run_start(self, info: EngineInfo) -> None:
        for reporter in self._reporters:
            try:
                reporter.on_run_start(info, self._model)
            except Exception:
                pass

    def _emit_test_complete(self, result: TestResult) -> None:
        for reporter in self._reporters:
            try:
                reporter.on_test_complete(result)
            except Exception:
                pass

    def _execute_tests(self, info: EngineInfo) -> Report:
        results: list[TestResult] = []
        start = time.time()
        abort_reason: str | None = None

        for test in self._tests:
            self._reset_timeout_observation()
            self._set_inspection_test(test.test_id())

            missing = set(test.modalities()) - self._engine.supported_modalities()
            if missing:
                reason = (
                    f"engine '{self._engine.engine_id()}' does not support "
                    f"modality/modalities: {', '.join(sorted(missing))}"
                )
                result = TestResult(
                    test_id=test.test_id(),
                    test_name=test.test_name(),
                    engine_id=self._engine.engine_id(),
                    model=self._model,
                    checks=[],
                    elapsed_seconds=0.0,
                    skipped=True,
                    skip_reason=reason,
                )
                results.append(result)
                self._emit_test_complete(result)
                continue

            try:
                result = test.run(self._engine, self._model)
            except Exception as e:
                result = TestResult(
                    test_id=test.test_id(),
                    test_name=test.test_name(),
                    engine_id=self._engine.engine_id(),
                    model=self._model,
                    checks=[],
                    elapsed_seconds=0.0,
                    error=str(e),
                )
            finally:
                self._set_inspection_test(None)
            results.append(result)
            self._emit_test_complete(result)

            if self._engine_timed_out():
                abort_reason = (
                    f"Engine timed out during '{test.test_name()}'. "
                    f"Aborting remaining tests to avoid further timeouts."
                )
                if info.mode == "managed":
                    abort_reason += (
                        " For large models, consider launching the engine "
                        "externally and running in external mode instead."
                    )
                break

        total_elapsed = time.time() - start
        if abort_reason is None:
            if not results:
                abort_reason = "No tests were selected."
            elif all(r.skipped for r in results):
                abort_reason = "No tests were executed; all selected tests were skipped."

        return self._build_report(
            info,
            results=results,
            total_elapsed_seconds=total_elapsed,
            abort_reason=abort_reason,
        )

    def _reset_timeout_observation(self) -> None:
        reset = getattr(self._engine, "reset_timeout_observed", None)
        if callable(reset):
            reset()

    def _set_inspection_test(self, test_id: str | None) -> None:
        inspection = getattr(self._engine, "_inspection", None)
        if inspection is None:
            return
        setter = getattr(inspection, "set_current_test", None)
        if callable(setter):
            setter(test_id)

    def _engine_timed_out(self) -> bool:
        checker = getattr(self._engine, "timed_out_since_last_check", None)
        if callable(checker):
            return bool(checker())
        return bool(getattr(self._engine, "_last_timeout", False))
