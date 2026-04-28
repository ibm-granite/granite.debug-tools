from __future__ import annotations

import time

from granite_validation.domain.models import CheckResult, TestResult
from granite_validation.engines.base import AbstractEngine
from granite_validation.engines.llamacpp import LlamaCppEngine
from granite_validation.tests.base import AbstractValidationTest
from granite_validation.tests.registry import register_test


@register_test("llamacpp_props")
class PropsEndpointTest(AbstractValidationTest):
    """Validates the /props metadata endpoint on llama.cpp server."""

    def test_id(self) -> str:
        return "llamacpp_props"

    def test_name(self) -> str:
        return "llama.cpp Props Endpoint"

    def applicable_engines(self) -> list[str]:
        return ["llamacpp"]

    def run(self, engine: AbstractEngine, model: str) -> TestResult:
        assert isinstance(engine, LlamaCppEngine)
        checks: list[CheckResult] = []
        start = time.time()

        try:
            body = engine.props()
        except Exception as e:
            checks.append(CheckResult(
                name="props_status_200", passed=False,
                expected=200, actual=str(e),
            ))
            return TestResult(
                test_id=self.test_id(),
                test_name=self.test_name(),
                engine_id=engine.engine_id(),
                model=model,
                checks=checks,
                elapsed_seconds=time.time() - start,
            )

        checks.append(CheckResult(
            name="props_status_200",
            passed=True,
            expected=200,
            actual=200,
        ))

        has_metadata = (
            "chat_template" in body or "default_generation_settings" in body
        )
        checks.append(CheckResult(
            name="props_has_metadata",
            passed=has_metadata,
            expected="chat_template or default_generation_settings",
            actual=str(list(body.keys()))[:200],
        ))

        return TestResult(
            test_id=self.test_id(),
            test_name=self.test_name(),
            engine_id=engine.engine_id(),
            model=model,
            checks=checks,
            elapsed_seconds=time.time() - start,
        )
