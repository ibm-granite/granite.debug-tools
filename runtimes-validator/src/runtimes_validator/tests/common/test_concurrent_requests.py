from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from runtimes_validator.domain.models import CheckResult, TestResult
from runtimes_validator.engines.base import AbstractEngine
from runtimes_validator.tests.base import AbstractValidationTest
from runtimes_validator.tests.registry import register_test


@register_test("concurrent_requests")
class ConcurrentRequestTest(AbstractValidationTest):
    """Validates that the engine handles concurrent chat requests."""

    def test_id(self) -> str:
        return "concurrent_requests"

    def test_name(self) -> str:
        return "Concurrent Requests"

    def run(self, engine: AbstractEngine, model: str) -> TestResult:
        checks: list[CheckResult] = []
        start = time.time()

        questions = {
            "q1": "What is 1+1?",
            "q2": "What is 2+2?",
        }

        def _do_chat(question: str) -> dict[str, Any]:
            return engine.chat(
                [{"role": "user", "content": question}],
                max_tokens=32,
            )

        results: dict[str, dict[str, Any] | Exception] = {}

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {executor.submit(_do_chat, q): key for key, q in questions.items()}
            for future in as_completed(futures):
                key = futures[future]
                try:
                    results[key] = future.result()
                except Exception as e:
                    results[key] = e

        all_ok = True
        for key in ("q1", "q2"):
            result = results.get(key)
            if isinstance(result, Exception):
                checks.append(
                    CheckResult(
                        name=f"concurrent_{key}_error",
                        passed=False,
                        detail=str(result),
                    )
                )
                all_ok = False
                continue

            if result is None:
                checks.append(
                    CheckResult(
                        name=f"concurrent_{key}_error",
                        passed=False,
                        detail="No result returned",
                    )
                )
                all_ok = False
                continue

            checks.append(
                CheckResult(
                    name=f"concurrent_{key}_role",
                    passed=result.get("role") == "assistant",
                    expected="assistant",
                    actual=result.get("role"),
                )
            )
            checks.append(
                CheckResult(
                    name=f"concurrent_{key}_content_nonempty",
                    passed=bool(result.get("content")),
                    expected="non-empty content",
                    actual=(result.get("content") or "")[:200],
                )
            )

        checks.append(
            CheckResult(
                name="concurrent_both_completed",
                passed=all_ok,
                expected="both requests succeed",
                actual=f"{len([r for r in results.values() if not isinstance(r, Exception)])}/2",
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
