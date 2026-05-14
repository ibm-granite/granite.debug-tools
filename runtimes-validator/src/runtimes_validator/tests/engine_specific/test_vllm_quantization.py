from __future__ import annotations

import time

import requests

from runtimes_validator.domain.models import CheckResult, TestResult
from runtimes_validator.engines.base import AbstractEngine
from runtimes_validator.engines.vllm import VllmEngine
from runtimes_validator.tests.base import AbstractValidationTest
from runtimes_validator.tests.registry import register_test

_KNOWN_QUANTIZATIONS = {"awq", "gptq", "fp8", "squeezellm", "marlin", "gptq_marlin"}


@register_test("vllm_quantization")
class QuantizationTest(AbstractValidationTest):
    """Validates that a quantized model loads and generates correctly on vLLM."""

    def test_id(self) -> str:
        return "vllm_quantization"

    def test_name(self) -> str:
        return "vLLM Quantization"

    def applicable_engines(self) -> list[str]:
        return ["vllm"]

    def run(self, engine: AbstractEngine, model: str) -> TestResult:
        assert isinstance(engine, VllmEngine)
        checks: list[CheckResult] = []
        start = time.time()

        if self._check_quantization_config(engine, checks):
            self._check_model_listed(engine, checks)
            self._check_generation(engine, model, checks)

        return TestResult(
            test_id=self.test_id(),
            test_name=self.test_name(),
            engine_id=engine.engine_id(),
            model=model,
            checks=checks,
            elapsed_seconds=time.time() - start,
        )

    def _check_quantization_config(
        self,
        engine: VllmEngine,
        checks: list[CheckResult],
    ) -> bool:
        quant = engine._config.extra.get("quantization")

        if quant is None:
            checks.append(
                CheckResult(
                    name="quantization_configured",
                    passed=True,
                    detail=("No quantization method configured; skipping quantization checks"),
                )
            )
            return False

        checks.append(
            CheckResult(
                name="quantization_configured",
                passed=True,
                expected="a quantization method",
                actual=str(quant),
            )
        )

        if str(quant).lower() not in _KNOWN_QUANTIZATIONS:
            checks.append(
                CheckResult(
                    name="quantization_method_known",
                    passed=True,
                    expected=f"one of {sorted(_KNOWN_QUANTIZATIONS)}",
                    actual=str(quant),
                    detail="Unrecognized quantization method; checks will still run",
                )
            )

        return True

    def _check_model_listed(
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
                    name="quantized_model_listed",
                    passed=False,
                    detail=str(e),
                )
            )
            return

        checks.append(
            CheckResult(
                name="quantized_model_listed",
                passed=len(models) > 0,
                expected=">= 1 model",
                actual=len(models),
            )
        )

    def _check_generation(
        self,
        engine: VllmEngine,
        model: str,
        checks: list[CheckResult],
    ) -> None:
        try:
            resp = engine.chat(
                [{"role": "user", "content": "What is the capital of France?"}],
                max_tokens=64,
                temperature=0.0,
            )
        except Exception as e:
            checks.append(
                CheckResult(
                    name="quantized_generation_content",
                    passed=False,
                    detail=str(e),
                )
            )
            return

        content = resp.get("content") or ""
        checks.append(
            CheckResult(
                name="quantized_generation_content",
                passed=bool(content.strip()),
                expected="non-empty response",
                actual=content[:200],
            )
        )

        fr = resp.get("finish_reason", "")
        checks.append(
            CheckResult(
                name="quantized_generation_finish_reason",
                passed=fr in ("stop", "length"),
                expected="stop or length",
                actual=fr,
            )
        )

        checks.append(
            CheckResult(
                name="quantized_generation_quality",
                passed="paris" in content.lower(),
                expected="'paris' in response",
                actual=content[:200],
            )
        )
