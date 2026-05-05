from __future__ import annotations

import time

from runtimes_validator.domain.models import CheckResult, TestResult
from runtimes_validator.engines.base import AbstractEngine
from runtimes_validator.engines.ollama import OllamaEngine
from runtimes_validator.tests.base import AbstractValidationTest
from runtimes_validator.tests.registry import register_test


@register_test("ollama_model_metadata")
class OllamaModelMetadataTest(AbstractValidationTest):
    """Validates Ollama model metadata via /api/show and model loading."""

    def test_id(self) -> str:
        return "ollama_model_metadata"

    def test_name(self) -> str:
        return "Ollama Model Metadata"

    def applicable_engines(self) -> list[str]:
        return ["ollama"]

    def run(self, engine: AbstractEngine, model: str) -> TestResult:
        assert isinstance(engine, OllamaEngine)
        checks: list[CheckResult] = []
        start = time.time()

        self._check_show_metadata(engine, model, checks)
        self._check_model_loads(engine, model, checks)

        return TestResult(
            test_id=self.test_id(),
            test_name=self.test_name(),
            engine_id=engine.engine_id(),
            model=model,
            checks=checks,
            elapsed_seconds=time.time() - start,
        )

    def _check_show_metadata(
        self, engine: OllamaEngine, model: str, checks: list[CheckResult],
    ) -> None:
        try:
            body = engine.show(model=model)
        except Exception as e:
            checks.append(CheckResult(
                name="show_status_200", passed=False,
                expected=200, actual=str(e),
            ))
            return

        checks.append(CheckResult(
            name="show_status_200",
            passed=True,
            expected=200,
            actual=200,
        ))
        checks.append(CheckResult(
            name="show_valid_json",
            passed=True,
            expected="valid JSON",
            actual="valid JSON",
        ))
        checks.append(CheckResult(
            name="show_has_modelfile",
            passed=bool(body.get("modelfile")),
            expected="non-empty modelfile",
            actual=str(body.get("modelfile", ""))[:100],
        ))

        details = body.get("details", {})
        checks.append(CheckResult(
            name="show_has_family",
            passed=bool(details.get("family")),
            expected="non-empty family",
            actual=details.get("family"),
        ))
        checks.append(CheckResult(
            name="show_has_parameter_size",
            passed=bool(details.get("parameter_size")),
            expected="non-empty parameter_size",
            actual=details.get("parameter_size"),
        ))
        checks.append(CheckResult(
            name="show_has_quantization_level",
            passed=bool(details.get("quantization_level")),
            expected="non-empty quantization_level",
            actual=details.get("quantization_level"),
        ))
        checks.append(CheckResult(
            name="show_has_template",
            passed=bool(body.get("template")),
            expected="non-empty template",
            actual=str(body.get("template", ""))[:100],
        ))

        model_info = body.get("model_info", {})
        checks.append(CheckResult(
            name="show_has_model_info",
            passed=len(model_info) > 0,
            expected="at least 1 key in model_info",
            actual=f"{len(model_info)} keys",
        ))

    def _check_model_loads(
        self, engine: OllamaEngine, model: str, checks: list[CheckResult],
    ) -> None:
        try:
            body = engine.generate("", model=model)
        except Exception as e:
            checks.append(CheckResult(
                name="model_loads", passed=False,
                expected=200, actual=str(e),
            ))
            return

        checks.append(CheckResult(
            name="model_loads",
            passed=bool(body.get("model")),
            expected="model field present",
            actual=body.get("model"),
        ))
