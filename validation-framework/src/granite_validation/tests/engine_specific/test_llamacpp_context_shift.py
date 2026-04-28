from __future__ import annotations

import time

from granite_validation.domain.models import CheckResult, TestResult
from granite_validation.engines.base import AbstractEngine
from granite_validation.engines.llamacpp import LlamaCppEngine
from granite_validation.tests.base import AbstractValidationTest
from granite_validation.tests.registry import register_test


@register_test("llamacpp_context_shift")
class ContextShiftTest(AbstractValidationTest):
    """Validates that llama.cpp handles prompts exceeding the context window.

    Requires the server to be started with context shift enabled
    (e.g. ``--ctx-size 512 --ctx-shift``).
    """

    def test_id(self) -> str:
        return "llamacpp_context_shift"

    def test_name(self) -> str:
        return "llama.cpp Context Shift"

    def applicable_engines(self) -> list[str]:
        return ["llamacpp"]

    def run(self, engine: AbstractEngine, model: str) -> TestResult:
        assert isinstance(engine, LlamaCppEngine)
        checks: list[CheckResult] = []
        start = time.time()

        # Build a prompt that exceeds a typical 512-token context window
        long_text = "The quick brown fox jumps over the lazy dog. " * 80

        try:
            result = engine.chat(
                [{"role": "user", "content": f"{long_text} Summarize the above."}],
                max_tokens=32,
                temperature=0.0,
            )
        except Exception as e:
            checks.append(CheckResult(
                name="context_shift_status_200", passed=False,
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
            name="context_shift_status_200",
            passed=True,
            expected=200,
            actual=200,
        ))

        finish_reason = result.get("finish_reason")
        checks.append(CheckResult(
            name="context_shift_finish_reason",
            passed=finish_reason in ("stop", "length"),
            expected="stop or length",
            actual=finish_reason,
        ))

        return TestResult(
            test_id=self.test_id(),
            test_name=self.test_name(),
            engine_id=engine.engine_id(),
            model=model,
            checks=checks,
            elapsed_seconds=time.time() - start,
        )
