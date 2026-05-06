from __future__ import annotations

import time

import requests

from runtimes_validator.domain.models import CheckResult, TestResult
from runtimes_validator.engines.base import AbstractEngine
from runtimes_validator.engines.vllm import VllmEngine
from runtimes_validator.tests.base import AbstractValidationTest
from runtimes_validator.tests.registry import register_test


@register_test("vllm_tensor_parallel")
class TensorParallelTest(AbstractValidationTest):
    """Validates model loading and generation under vLLM tensor parallelism."""

    def test_id(self) -> str:
        return "vllm_tensor_parallel"

    def test_name(self) -> str:
        return "vLLM Tensor Parallelism"

    def applicable_engines(self) -> list[str]:
        return ["vllm"]

    def run(self, engine: AbstractEngine, model: str) -> TestResult:
        assert isinstance(engine, VllmEngine)
        checks: list[CheckResult] = []
        start = time.time()

        if self._check_tp_config(engine, checks):
            self._check_tp_health(engine, checks)
            self._check_tp_model_listed(engine, checks)
            self._check_tp_generation(engine, model, checks)

        return TestResult(
            test_id=self.test_id(),
            test_name=self.test_name(),
            engine_id=engine.engine_id(),
            model=model,
            checks=checks,
            elapsed_seconds=time.time() - start,
        )

    def _check_tp_config(
        self,
        engine: VllmEngine,
        checks: list[CheckResult],
    ) -> bool:
        tp_size = engine._config.extra.get("tensor_parallel_size")

        if tp_size is None or int(tp_size) <= 1:
            checks.append(
                CheckResult(
                    name="tp_config_present",
                    passed=True,
                    detail=(
                        "tensor_parallel_size not configured or is 1; "
                        "skipping tensor parallelism checks"
                    ),
                )
            )
            return False

        checks.append(
            CheckResult(
                name="tp_config_present",
                passed=True,
                expected="> 1",
                actual=str(tp_size),
            )
        )
        return True

    def _check_tp_health(
        self,
        engine: VllmEngine,
        checks: list[CheckResult],
    ) -> None:
        try:
            healthy = engine.health_check()
        except Exception as e:
            checks.append(
                CheckResult(
                    name="tp_health",
                    passed=False,
                    detail=str(e),
                )
            )
            return

        checks.append(
            CheckResult(
                name="tp_health",
                passed=healthy,
                expected=True,
                actual=healthy,
            )
        )

    def _check_tp_model_listed(
        self,
        engine: VllmEngine,
        checks: list[CheckResult],
    ) -> None:
        try:
            resp = requests.get(
                f"{engine._base_url}/v1/models",
                headers=engine._headers or None,
                timeout=engine._http_timeout,
            )
            resp.raise_for_status()
            models = resp.json().get("data", [])
        except Exception as e:
            checks.append(
                CheckResult(
                    name="tp_model_listed",
                    passed=False,
                    detail=str(e),
                )
            )
            return

        checks.append(
            CheckResult(
                name="tp_model_listed",
                passed=len(models) > 0,
                expected=">= 1 model",
                actual=len(models),
            )
        )

    def _check_tp_generation(
        self,
        engine: VllmEngine,
        model: str,
        checks: list[CheckResult],
    ) -> None:
        try:
            resp = engine.chat(
                [{"role": "user", "content": "What is the capital of Japan?"}],
                max_tokens=64,
                temperature=0.0,
            )
        except Exception as e:
            checks.append(
                CheckResult(
                    name="tp_generation_content",
                    passed=False,
                    detail=str(e),
                )
            )
            return

        content = resp.get("content") or ""
        checks.append(
            CheckResult(
                name="tp_generation_content",
                passed=bool(content.strip()),
                expected="non-empty response",
                actual=content[:200],
            )
        )

        fr = resp.get("finish_reason", "")
        checks.append(
            CheckResult(
                name="tp_generation_finish_reason",
                passed=fr in ("stop", "length"),
                expected="stop or length",
                actual=fr,
            )
        )
