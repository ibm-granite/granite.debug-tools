from __future__ import annotations

import time

from runtimes_validator.domain.models import CheckResult, TestResult
from runtimes_validator.engines.base import AbstractEngine
from runtimes_validator.tests.base import AbstractValidationTest
from runtimes_validator.tests.registry import register_test


@register_test("no_tool_tags_without_tools")
class NoToolTagsWithoutToolsTest(AbstractValidationTest):
    """Validates that the model does not emit tool-calling markup when no tools are provided."""

    def test_id(self) -> str:
        return "no_tool_tags_without_tools"

    def test_name(self) -> str:
        return "No Tool Tags Without Tools"

    def run(self, engine: AbstractEngine, model: str) -> TestResult:
        checks: list[CheckResult] = []
        start = time.time()

        try:
            with self._check_scope(engine, "no_tool_tags"):
                response = engine.chat(
                    [{"role": "user", "content": "What is 2 + 2? Just answer with the number."}],
                    max_tokens=64,
                )
        except Exception as e:
            checks.append(CheckResult(name="no_tool_tags_error", passed=False, detail=str(e)))
            return TestResult(
                test_id=self.test_id(),
                test_name=self.test_name(),
                engine_id=engine.engine_id(),
                model=model,
                checks=checks,
                elapsed_seconds=time.time() - start,
            )

        content = response.get("content", "") or ""
        checks.append(
            CheckResult(
                name="no_tool_tags_has_content",
                passed=bool(content),
                expected="non-empty content",
                actual=content[:200],
            )
        )
        checks.append(
            CheckResult(
                name="no_tool_tags_no_open_tag",
                passed="<tool_call>" not in content,
                expected="no <tool_call> in output",
                actual="found <tool_call>" if "<tool_call>" in content else "clean",
            )
        )
        checks.append(
            CheckResult(
                name="no_tool_tags_no_close_tag",
                passed="</tool_call>" not in content,
                expected="no </tool_call> in output",
                actual="found </tool_call>" if "</tool_call>" in content else "clean",
            )
        )
        checks.append(
            CheckResult(
                name="no_tool_tags_finish_reason",
                passed=response.get("finish_reason") != "tool_calls",
                expected="not tool_calls",
                actual=response.get("finish_reason"),
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
