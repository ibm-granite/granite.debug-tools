from __future__ import annotations

import time

from runtimes_validator.domain.models import CheckResult, TestResult
from runtimes_validator.engines.base import AbstractEngine
from runtimes_validator.engines.llamacpp import LlamaCppEngine
from runtimes_validator.tests.base import AbstractValidationTest
from runtimes_validator.tests.registry import register_test

_SPECIAL_TOKENS = [
    "<|start_of_role|>",
    "<|end_of_role|>",
    "<|end_of_text|>",
]


@register_test("llamacpp_tokenization")
class TokenizationTest(AbstractValidationTest):
    """Validates tokenize/detokenize roundtrip and special tokens on llama.cpp."""

    def test_id(self) -> str:
        return "llamacpp_tokenization"

    def test_name(self) -> str:
        return "llama.cpp Tokenization"

    def applicable_engines(self) -> list[str]:
        return ["llamacpp"]

    def run(self, engine: AbstractEngine, model: str) -> TestResult:
        assert isinstance(engine, LlamaCppEngine)
        checks: list[CheckResult] = []
        start = time.time()

        self._check_roundtrip(engine, checks)
        self._check_special_tokens(engine, checks)

        return TestResult(
            test_id=self.test_id(),
            test_name=self.test_name(),
            engine_id=engine.engine_id(),
            model=model,
            checks=checks,
            elapsed_seconds=time.time() - start,
        )

    def _check_roundtrip(
        self, engine: LlamaCppEngine, checks: list[CheckResult],
    ) -> None:
        text = "Hello, how are you today?"

        try:
            tokens = engine.tokenize(text)
        except Exception as e:
            checks.append(CheckResult(
                name="tokenize_error", passed=False, detail=str(e),
            ))
            return

        checks.append(CheckResult(
            name="tokenize_produces_tokens",
            passed=len(tokens) > 0,
            expected="> 0 tokens",
            actual=len(tokens),
        ))

        try:
            roundtrip_text = engine.detokenize(tokens)
        except Exception as e:
            checks.append(CheckResult(
                name="detokenize_error", passed=False, detail=str(e),
            ))
            return

        checks.append(CheckResult(
            name="roundtrip_matches_original",
            passed=roundtrip_text == text,
            expected=text,
            actual=roundtrip_text,
        ))

    def _check_special_tokens(
        self, engine: LlamaCppEngine, checks: list[CheckResult],
    ) -> None:
        for token_str in _SPECIAL_TOKENS:
            safe_name = token_str.strip("<|>").replace("|", "_")
            try:
                token_ids = engine.tokenize(
                    token_str, add_special=False, parse_special=True,
                )
            except Exception as e:
                checks.append(CheckResult(
                    name=f"special_token_{safe_name}_error",
                    passed=False,
                    detail=str(e),
                ))
                continue

            checks.append(CheckResult(
                name=f"special_token_{safe_name}",
                passed=len(token_ids) == 1,
                expected=f"'{token_str}' -> 1 token",
                actual=f"{len(token_ids)} tokens",
            ))
