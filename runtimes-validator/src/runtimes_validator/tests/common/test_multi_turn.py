from __future__ import annotations

import time

from runtimes_validator.domain.models import CheckResult, TestResult
from runtimes_validator.engines.base import AbstractEngine
from runtimes_validator.tests.base import AbstractValidationTest
from runtimes_validator.tests.registry import register_test


@register_test("multi_turn")
class MultiTurnTest(AbstractValidationTest):
    """Validates that the model retains information across a multi-turn conversation."""

    def test_id(self) -> str:
        return "multi_turn"

    def test_name(self) -> str:
        return "Multi-Turn Context Retention"

    def run(self, engine: AbstractEngine, model: str) -> TestResult:
        checks: list[CheckResult] = []
        start = time.time()

        try:
            response = engine.chat(
                [
                    {"role": "system", "content": "You are a helpful assistant. Be brief."},
                    {"role": "user", "content": "My name is Alice and I live in Paris."},
                    {
                        "role": "assistant",
                        "content": (
                            "Hello Alice! Nice to know you're in Paris. How can I help you?"
                        ),
                    },
                    {"role": "user", "content": "What is my name and where do I live?"},
                ],
                max_tokens=64,
            )
        except Exception as e:
            checks.append(CheckResult(name="context_error", passed=False, detail=str(e)))
            return TestResult(
                test_id=self.test_id(),
                test_name=self.test_name(),
                engine_id=engine.engine_id(),
                model=model,
                checks=checks,
                elapsed_seconds=time.time() - start,
            )

        content = (response.get("content") or "").lower()

        checks.append(
            CheckResult(
                name="context_retention_alice",
                passed="alice" in content,
                expected="'alice' in response",
                actual=content[:200],
            )
        )
        checks.append(
            CheckResult(
                name="response_role",
                passed=response.get("role") == "assistant",
                expected="assistant",
                actual=response.get("role"),
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
