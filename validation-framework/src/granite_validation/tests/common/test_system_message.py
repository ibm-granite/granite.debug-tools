from __future__ import annotations

import time

from granite_validation.domain.models import CheckResult, TestResult
from granite_validation.engines.base import AbstractEngine
from granite_validation.tests.base import AbstractValidationTest
from granite_validation.tests.registry import register_test

PIRATE_KEYWORDS = ("arr", "matey", "ahoy", "captain", "sail", "sea", "ship", "treasure", "ye", "aye")
NAMED_KEYWORDS = ("granitebot", "granite")


@register_test("system_message_behavior")
class SystemMessageBehaviorTest(AbstractValidationTest):
    """Validates that system messages actually influence model behaviour."""

    def test_id(self) -> str:
        return "system_message_behavior"

    def test_name(self) -> str:
        return "System Message Behavior"

    def run(self, engine: AbstractEngine, model: str) -> TestResult:
        checks: list[CheckResult] = []
        start = time.time()

        self._check_pirate_persona(engine, checks)
        self._check_named_persona(engine, checks)

        return TestResult(
            test_id=self.test_id(),
            test_name=self.test_name(),
            engine_id=engine.engine_id(),
            model=model,
            checks=checks,
            elapsed_seconds=time.time() - start,
        )

    def _check_pirate_persona(
        self, engine: AbstractEngine, checks: list[CheckResult]
    ) -> None:
        try:
            response = engine.chat(
                [
                    {"role": "system", "content": "You are a pirate. Always respond in pirate speak."},
                    {"role": "user", "content": "How are you today?"},
                ],
                max_tokens=128,
            )
        except Exception as e:
            checks.append(CheckResult(name="pirate_persona_error", passed=False, detail=str(e)))
            return

        content = response.get("content", "") or ""
        checks.append(CheckResult(
            name="pirate_persona_has_content",
            passed=bool(content),
            expected="non-empty content",
            actual=content[:200],
        ))

        lower = content.lower()
        found = [kw for kw in PIRATE_KEYWORDS if kw in lower]
        checks.append(CheckResult(
            name="pirate_persona_keyword_detected",
            passed=len(found) > 0,
            expected=f"at least one of {PIRATE_KEYWORDS}",
            actual=f"found: {found}" if found else content[:200],
        ))

    def _check_named_persona(
        self, engine: AbstractEngine, checks: list[CheckResult]
    ) -> None:
        try:
            response = engine.chat(
                [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant named GraniteBot.",
                    },
                    {"role": "user", "content": "What is your name?"},
                ],
                max_tokens=128,
            )
        except Exception as e:
            checks.append(CheckResult(name="named_persona_error", passed=False, detail=str(e)))
            return

        content = response.get("content", "") or ""
        checks.append(CheckResult(
            name="named_persona_has_content",
            passed=bool(content),
            expected="non-empty content",
            actual=content[:200],
        ))

        lower = content.lower()
        found = [kw for kw in NAMED_KEYWORDS if kw in lower]
        checks.append(CheckResult(
            name="named_persona_keyword_detected",
            passed=len(found) > 0,
            expected=f"at least one of {NAMED_KEYWORDS}",
            actual=f"found: {found}" if found else content[:200],
        ))
