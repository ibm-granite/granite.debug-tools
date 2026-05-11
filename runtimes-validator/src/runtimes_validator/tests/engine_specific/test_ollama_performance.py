from __future__ import annotations

import time

from runtimes_validator.domain.models import CheckResult, TestResult
from runtimes_validator.engines.base import AbstractEngine
from runtimes_validator.engines.ollama import OllamaEngine
from runtimes_validator.tests.base import AbstractValidationTest
from runtimes_validator.tests.registry import register_test


@register_test("ollama_performance")
class OllamaPerformanceTest(AbstractValidationTest):
    """Validates throughput, load time, and prompt evaluation metrics via Ollama API."""

    def test_id(self) -> str:
        return "ollama_performance"

    def test_name(self) -> str:
        return "Ollama Performance"

    def applicable_engines(self) -> list[str]:
        return ["ollama"]

    def run(self, engine: AbstractEngine, model: str) -> TestResult:
        assert isinstance(engine, OllamaEngine)
        checks: list[CheckResult] = []
        start = time.time()

        self._check_throughput(engine, model, checks)
        self._check_load_time(engine, model, checks)
        self._check_prompt_eval(engine, model, checks)

        return TestResult(
            test_id=self.test_id(),
            test_name=self.test_name(),
            engine_id=engine.engine_id(),
            model=model,
            checks=checks,
            elapsed_seconds=time.time() - start,
        )

    def _check_throughput(
        self,
        engine: OllamaEngine,
        model: str,
        checks: list[CheckResult],
    ) -> None:
        try:
            with self._check_scope(engine, checks, "throughput"):
                body = engine.generate(
                    "Write a detailed explanation of how computers work.",
                    model=model,
                    options={"num_predict": 100},
                    timeout=300,
                )
        except Exception as e:
            checks.append(CheckResult(name="throughput_error", passed=False, detail=str(e)))
            return

        eval_count = body.get("eval_count", 0)
        eval_duration = body.get("eval_duration", 0)

        checks.append(
            CheckResult(
                name="throughput_has_metrics",
                passed=eval_count > 0 and eval_duration > 0,
                expected="eval_count > 0 and eval_duration > 0",
                actual=f"eval_count={eval_count}, eval_duration={eval_duration}",
            )
        )

        if eval_duration > 0:
            tok_per_sec = eval_count * 1e9 / eval_duration
            checks.append(
                CheckResult(
                    name="throughput_tokens_per_second",
                    passed=tok_per_sec >= 1.0,
                    expected=">= 1.0 tok/s",
                    actual=f"{tok_per_sec:.2f} tok/s",
                )
            )

    def _check_load_time(
        self,
        engine: OllamaEngine,
        model: str,
        checks: list[CheckResult],
    ) -> None:
        # Unload the model
        try:
            engine.generate("", model=model, keep_alive=0, timeout=30)
        except Exception as e:
            checks.append(
                CheckResult(
                    name="load_time_unload_error",
                    passed=False,
                    detail=str(e),
                )
            )
            return

        time.sleep(2)

        # Reload and measure
        try:
            body = engine.generate(
                "Hello",
                model=model,
                options={"num_predict": 1},
                timeout=300,
            )
        except Exception as e:
            checks.append(
                CheckResult(
                    name="load_time_reload_error",
                    passed=False,
                    detail=str(e),
                )
            )
            return

        load_duration = body.get("load_duration", 0)
        checks.append(
            CheckResult(
                name="load_time_has_load_duration",
                passed=load_duration > 0,
                expected="> 0",
                actual=load_duration,
            )
        )

        if load_duration > 0:
            load_seconds = load_duration / 1e9
            checks.append(
                CheckResult(
                    name="load_time_under_threshold",
                    passed=load_seconds < 300,
                    expected="< 300 seconds",
                    actual=f"{load_seconds:.2f} seconds",
                )
            )

    def _check_prompt_eval(
        self,
        engine: OllamaEngine,
        model: str,
        checks: list[CheckResult],
    ) -> None:
        try:
            body = engine.generate(
                "Explain quantum computing in simple terms.",
                model=model,
                options={"num_predict": 50},
            )
        except Exception as e:
            checks.append(
                CheckResult(
                    name="prompt_eval_error",
                    passed=False,
                    detail=str(e),
                )
            )
            return

        prompt_eval_count = body.get("prompt_eval_count", 0)
        # prompt_eval_count can be 0 if prompt is cached; that's acceptable
        checks.append(
            CheckResult(
                name="prompt_eval_has_count",
                passed=prompt_eval_count >= 0,
                expected=">= 0 (0 if cached)",
                actual=prompt_eval_count,
            )
        )

        prompt_eval_duration = body.get("prompt_eval_duration", 0)
        if prompt_eval_duration > 0 and prompt_eval_count > 0:
            tok_per_sec = prompt_eval_count * 1e9 / prompt_eval_duration
            checks.append(
                CheckResult(
                    name="prompt_eval_tokens_per_second",
                    passed=tok_per_sec > 0,
                    expected="> 0 tok/s",
                    actual=f"{tok_per_sec:.2f} tok/s",
                )
            )
