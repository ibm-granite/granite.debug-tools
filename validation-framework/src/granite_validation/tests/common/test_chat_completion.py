from __future__ import annotations

import time

from granite_validation.domain.models import CheckResult, TestResult
from granite_validation.engines.base import AbstractEngine
from granite_validation.tests.base import AbstractValidationTest
from granite_validation.tests.registry import register_test


@register_test("chat_completion")
class ChatCompletionTest(AbstractValidationTest):
    """Validates basic chat completion behaviour across all engines."""

    def test_id(self) -> str:
        return "chat_completion"

    def test_name(self) -> str:
        return "Chat Completion"

    def run(self, engine: AbstractEngine, model: str) -> TestResult:
        checks: list[CheckResult] = []
        start = time.time()

        self._check_basic(engine, checks)
        self._check_no_system(engine, checks)
        self._check_max_tokens(engine, checks)

        return TestResult(
            test_id=self.test_id(),
            test_name=self.test_name(),
            engine_id=engine.engine_id(),
            model=model,
            checks=checks,
            elapsed_seconds=time.time() - start,
        )

    def _check_basic(self, engine: AbstractEngine, checks: list[CheckResult]) -> None:
        try:
            response = engine.chat(
                [
                    {"role": "system", "content": "You are a helpful assistant. Be brief."},
                    {"role": "user", "content": "What is 2+2?"},
                ],
                max_tokens=64,
            )
        except Exception as e:
            checks.append(CheckResult(name="basic_error", passed=False, detail=str(e)))
            return

        checks.append(CheckResult(
            name="basic_role",
            passed=response.get("role") == "assistant",
            expected="assistant",
            actual=response.get("role"),
        ))
        checks.append(CheckResult(
            name="basic_content_nonempty",
            passed=bool(response.get("content")),
            expected="non-empty content",
            actual=response.get("content", "")[:200],
        ))
        checks.append(CheckResult(
            name="basic_finish_reason",
            passed=response.get("finish_reason") in ("stop", "length"),
            expected="stop or length",
            actual=response.get("finish_reason"),
        ))

        usage = response.get("usage")
        if usage is not None:
            checks.append(CheckResult(
                name="basic_prompt_tokens",
                passed=usage.get("prompt_tokens", 0) > 0,
                expected="> 0",
                actual=usage.get("prompt_tokens"),
            ))
            checks.append(CheckResult(
                name="basic_completion_tokens",
                passed=usage.get("completion_tokens", 0) > 0,
                expected="> 0",
                actual=usage.get("completion_tokens"),
            ))

    def _check_no_system(
        self, engine: AbstractEngine, checks: list[CheckResult]
    ) -> None:
        try:
            response = engine.chat(
                [{"role": "user", "content": "Say hello."}],
                max_tokens=32,
            )
        except Exception as e:
            checks.append(CheckResult(name="no_system_error", passed=False, detail=str(e)))
            return

        checks.append(CheckResult(
            name="no_system_role",
            passed=response.get("role") == "assistant",
            expected="assistant",
            actual=response.get("role"),
        ))
        checks.append(CheckResult(
            name="no_system_content_nonempty",
            passed=bool(response.get("content")),
            expected="non-empty content",
            actual=response.get("content", "")[:200],
        ))

    def _check_max_tokens(
        self, engine: AbstractEngine, checks: list[CheckResult]
    ) -> None:
        try:
            response = engine.chat(
                [
                    {
                        "role": "user",
                        "content": "Write a very long essay about the history of computing.",
                    },
                ],
                max_tokens=5,
            )
        except Exception as e:
            checks.append(CheckResult(name="max_tokens_error", passed=False, detail=str(e)))
            return

        checks.append(CheckResult(
            name="max_tokens_finish_reason",
            passed=response.get("finish_reason") == "length",
            expected="length",
            actual=response.get("finish_reason"),
        ))

        usage = response.get("usage")
        if usage is not None:
            completion_tokens = usage.get("completion_tokens", 0)
            checks.append(CheckResult(
                name="max_tokens_token_count",
                passed=completion_tokens <= 6,
                expected="<= 6",
                actual=completion_tokens,
            ))
