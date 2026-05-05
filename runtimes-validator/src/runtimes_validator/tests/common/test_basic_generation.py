from __future__ import annotations

import time
from typing import Any

from runtimes_validator.domain.models import CheckResult, TestResult
from runtimes_validator.engines.base import AbstractEngine
from runtimes_validator.tests.base import AbstractValidationTest
from runtimes_validator.tests.registry import register_test

QUESTIONS: list[dict[str, Any]] = [
    {
        "id": "factual",
        "prompt": "What is the capital of France?",
        "expected_contains": ["Paris"],
    },
    {
        "id": "math",
        "prompt": "What is 17 * 23? Return only the number.",
        "expected_contains": ["391"],
    },
]


@register_test("basic_generation")
class BasicGenerationTest(AbstractValidationTest):
    """Verifies that the model can produce correct basic responses."""

    def test_id(self) -> str:
        return "basic_generation"

    def test_name(self) -> str:
        return "Basic Generation Quality"

    def run(self, engine: AbstractEngine, model: str) -> TestResult:
        checks: list[CheckResult] = []
        start = time.time()

        for question in QUESTIONS:
            try:
                response = engine.chat(
                    [{"role": "user", "content": question["prompt"]}]
                )
                content = (response.get("content") or "").lower()
            except Exception as e:
                checks.append(
                    CheckResult(
                        name=f"{question['id']}_error",
                        passed=False,
                        detail=str(e),
                    )
                )
                continue

            for expected in question.get("expected_contains", []):
                checks.append(
                    CheckResult(
                        name=f"{question['id']}_contains_{expected.lower()}",
                        passed=expected.lower() in content,
                        expected=expected,
                        actual=content[:200],
                    )
                )

        return TestResult(
            test_id=self.test_id(),
            test_name=self.test_name(),
            engine_id=engine.engine_id(),
            model=model,
            checks=checks,
            elapsed_seconds=time.time() - start,
        )
