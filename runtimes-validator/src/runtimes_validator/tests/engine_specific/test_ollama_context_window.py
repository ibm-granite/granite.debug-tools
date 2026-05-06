from __future__ import annotations

import time

from runtimes_validator.domain.models import CheckResult, TestResult
from runtimes_validator.engines.base import AbstractEngine
from runtimes_validator.engines.ollama import OllamaEngine
from runtimes_validator.tests.base import AbstractValidationTest
from runtimes_validator.tests.common.test_long_input import WAR_AND_PEACE_PASSAGE
from runtimes_validator.tests.registry import register_test


@register_test("ollama_context_window")
class OllamaContextWindowTest(AbstractValidationTest):
    """Validates context window handling via Ollama's native API."""

    def test_id(self) -> str:
        return "ollama_context_window"

    def test_name(self) -> str:
        return "Ollama Context Window"

    def applicable_engines(self) -> list[str]:
        return ["ollama"]

    def run(self, engine: AbstractEngine, model: str) -> TestResult:
        assert isinstance(engine, OllamaEngine)
        checks: list[CheckResult] = []
        start = time.time()

        self._check_long_input(engine, model, checks)
        self._check_context_exhaustion(engine, model, checks)
        self._check_context_continuation(engine, model, checks)

        return TestResult(
            test_id=self.test_id(),
            test_name=self.test_name(),
            engine_id=engine.engine_id(),
            model=model,
            checks=checks,
            elapsed_seconds=time.time() - start,
        )

    def _check_long_input(
        self,
        engine: OllamaEngine,
        model: str,
        checks: list[CheckResult],
    ) -> None:
        prompt = WAR_AND_PEACE_PASSAGE + " What country is this passage primarily discussing?"
        try:
            with self._check_scope(engine, checks, "long_input_num_ctx"):
                body = engine.native_chat(
                    [{"role": "user", "content": prompt}],
                    model=model,
                    options={"temperature": 0, "seed": 123, "num_ctx": 4096},
                    timeout=180,
                )
        except Exception as e:
            checks.append(
                CheckResult(
                    name="long_input_num_ctx_error",
                    passed=False,
                    detail=str(e),
                )
            )
            return

        content = body.get("message", {}).get("content", "")
        checks.append(
            CheckResult(
                name="long_input_num_ctx_has_content",
                passed=bool(content),
                expected="non-empty content",
                actual=content[:200],
            )
        )

        lower = content.lower()
        keywords = ("russia", "austria", "europe", "prussia", "england", "france")
        found = [kw for kw in keywords if kw in lower]
        checks.append(
            CheckResult(
                name="long_input_num_ctx_keywords",
                passed=len(found) > 0,
                expected=f"at least one of {keywords}",
                actual=f"found: {found}" if found else content[:200],
            )
        )

    def _check_context_exhaustion(
        self,
        engine: OllamaEngine,
        model: str,
        checks: list[CheckResult],
    ) -> None:
        try:
            with self._check_scope(engine, checks, "context_exhaustion"):
                body = engine.native_chat(
                    [
                        {
                            "role": "user",
                            "content": "Write me a long story with lots of emojis and characters",
                        },
                    ],
                    model=model,
                    options={"temperature": 0, "seed": 123, "num_ctx": 128, "num_predict": 200},
                    timeout=120,
                )
        except Exception as e:
            checks.append(
                CheckResult(
                    name="context_exhaustion_error",
                    passed=False,
                    detail=str(e),
                )
            )
            return

        content = body.get("message", {}).get("content", "")
        checks.append(
            CheckResult(
                name="context_exhaustion_has_content",
                passed=bool(content),
                expected="non-empty content (no crash with num_ctx=128)",
                actual=content[:200],
            )
        )

        done_reason = body.get("done_reason", "")
        checks.append(
            CheckResult(
                name="context_exhaustion_graceful",
                passed=done_reason in ("stop", "length"),
                expected="stop or length",
                actual=done_reason,
            )
        )

    def _check_context_continuation(
        self,
        engine: OllamaEngine,
        model: str,
        checks: list[CheckResult],
    ) -> None:
        # First generate call
        try:
            with self._check_scope(engine, checks, "context_continuation_first"):
                body1 = engine.generate(
                    "The capital of France is",
                    model=model,
                    options={"temperature": 0, "seed": 42, "num_predict": 30},
                    timeout=120,
                )
        except Exception as e:
            checks.append(
                CheckResult(
                    name="context_continuation_first_error",
                    passed=False,
                    detail=str(e),
                )
            )
            return

        context = body1.get("context")
        checks.append(
            CheckResult(
                name="context_continuation_first_has_context",
                passed=context is not None and len(context) > 0,
                expected="non-empty context array",
                actual=f"{len(context)} tokens" if context else "no context",
            )
        )

        response1 = body1.get("response", "")
        checks.append(
            CheckResult(
                name="context_continuation_first_has_response",
                passed=bool(response1),
                expected="non-empty response",
                actual=response1[:200],
            )
        )

        if not context:
            return

        # Second generate call reusing context
        try:
            with self._check_scope(engine, checks, "context_continuation_second"):
                body2 = engine.generate(
                    "And what is the capital of Germany?",
                    model=model,
                    context=context,
                    options={"temperature": 0, "seed": 42, "num_predict": 30},
                    timeout=120,
                )
        except Exception as e:
            checks.append(
                CheckResult(
                    name="context_continuation_second_error",
                    passed=False,
                    detail=str(e),
                )
            )
            return

        response2 = body2.get("response", "")
        checks.append(
            CheckResult(
                name="context_continuation_second_has_response",
                passed=bool(response2),
                expected="non-empty response",
                actual=response2[:200],
            )
        )

        checks.append(
            CheckResult(
                name="context_continuation_second_keyword",
                passed="berlin" in response2.lower(),
                expected="berlin in response",
                actual=response2[:200],
            )
        )
