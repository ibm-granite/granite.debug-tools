from __future__ import annotations

import time

from runtimes_validator.domain.models import CheckResult, TestResult
from runtimes_validator.engines.base import AbstractEngine
from runtimes_validator.tests.base import AbstractValidationTest
from runtimes_validator.tests.registry import register_test

GRANITE_SPECIAL_TOKENS = (
    "<|start_of_role|>",
    "<|end_of_role|>",
    "<|end_of_text|>",
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|fim_suffix|>",
    "<|fim_pad|>",
    "<|start_of_plugin|>",
    "<|end_of_plugin|>",
    "<|pad|>",
)


def _check_no_special_tokens(
    content: str, tokens: tuple[str, ...] = GRANITE_SPECIAL_TOKENS
) -> str | None:
    """Return the first special token found in content, or None if clean."""
    for token in tokens:
        if token in content:
            return token
    return None


@register_test("special_token_leakage")
class SpecialTokenLeakageTest(AbstractValidationTest):
    """Validates that special tokens do not leak into model output."""

    def test_id(self) -> str:
        return "special_token_leakage"

    def test_name(self) -> str:
        return "Special Token Leakage"

    def run(self, engine: AbstractEngine, model: str) -> TestResult:
        checks: list[CheckResult] = []
        start = time.time()

        self._check_generate_leakage(engine, checks)
        self._check_chat_leakage(engine, checks)
        self._check_eos_stops(engine, checks)

        return TestResult(
            test_id=self.test_id(),
            test_name=self.test_name(),
            engine_id=engine.engine_id(),
            model=model,
            checks=checks,
            elapsed_seconds=time.time() - start,
        )

    def _check_generate_leakage(self, engine: AbstractEngine, checks: list[CheckResult]) -> None:
        try:
            with self._check_scope(engine, checks, "generate_leakage"):
                response = engine.chat(
                    [
                        {
                            "role": "user",
                            "content": "Write a short paragraph about artificial intelligence.",
                        }
                    ],
                    max_tokens=256,
                )
        except Exception as e:
            checks.append(CheckResult(name="leakage_generate_error", passed=False, detail=str(e)))
            return

        content = response.get("content", "") or ""
        checks.append(
            CheckResult(
                name="leakage_generate_has_content",
                passed=bool(content),
                expected="non-empty content",
                actual=content[:200],
            )
        )

        leaked = _check_no_special_tokens(content)
        checks.append(
            CheckResult(
                name="leakage_generate_no_special_tokens",
                passed=leaked is None,
                expected="no special tokens in output",
                actual=f"found: {leaked}" if leaked else "clean",
            )
        )

    def _check_chat_leakage(self, engine: AbstractEngine, checks: list[CheckResult]) -> None:
        try:
            with self._check_scope(engine, checks, "chat_leakage"):
                response = engine.chat(
                    [
                        {"role": "system", "content": "You are a helpful assistant. Be concise."},
                        {"role": "user", "content": "Explain how the internet works in simple terms."},
                    ],
                    max_tokens=256,
                )
        except Exception as e:
            checks.append(CheckResult(name="leakage_chat_error", passed=False, detail=str(e)))
            return

        content = response.get("content", "") or ""
        checks.append(
            CheckResult(
                name="leakage_chat_has_content",
                passed=bool(content),
                expected="non-empty content",
                actual=content[:200],
            )
        )

        leaked = _check_no_special_tokens(content)
        checks.append(
            CheckResult(
                name="leakage_chat_no_special_tokens",
                passed=leaked is None,
                expected="no special tokens in output",
                actual=f"found: {leaked}" if leaked else "clean",
            )
        )

    def _check_eos_stops(self, engine: AbstractEngine, checks: list[CheckResult]) -> None:
        try:
            with self._check_scope(engine, checks, "eos_stops"):
                response = engine.chat(
                    [{"role": "user", "content": "Say just the word 'hello' and nothing else."}],
                    max_tokens=500,
                )
        except Exception as e:
            checks.append(CheckResult(name="eos_stops_error", passed=False, detail=str(e)))
            return

        checks.append(
            CheckResult(
                name="eos_stops_generation",
                passed=response.get("finish_reason") == "stop",
                expected="stop",
                actual=response.get("finish_reason"),
            )
        )
