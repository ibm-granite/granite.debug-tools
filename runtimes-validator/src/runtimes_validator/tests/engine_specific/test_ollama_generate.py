from __future__ import annotations

import time

from runtimes_validator.domain.models import CheckResult, TestResult
from runtimes_validator.engines.base import AbstractEngine
from runtimes_validator.engines.ollama import OllamaEngine
from runtimes_validator.tests.base import AbstractValidationTest
from runtimes_validator.tests.registry import register_test


@register_test("ollama_generate")
class OllamaGenerateTest(AbstractValidationTest):
    """Validates Ollama's /api/generate and /api/chat native endpoints."""

    def test_id(self) -> str:
        return "ollama_generate"

    def test_name(self) -> str:
        return "Ollama Generate API"

    def applicable_engines(self) -> list[str]:
        return ["ollama"]

    def run(self, engine: AbstractEngine, model: str) -> TestResult:
        assert isinstance(engine, OllamaEngine)
        checks: list[CheckResult] = []
        start = time.time()

        self._check_system_prompt(engine, model, checks)
        self._check_reproducibility(engine, model, checks)
        self._check_small_context(engine, model, checks)
        self._check_generate_metrics(engine, model, checks)
        self._check_chat_metrics(engine, model, checks)

        return TestResult(
            test_id=self.test_id(),
            test_name=self.test_name(),
            engine_id=engine.engine_id(),
            model=model,
            checks=checks,
            elapsed_seconds=time.time() - start,
        )

    def _check_system_prompt(
        self, engine: OllamaEngine, model: str, checks: list[CheckResult],
    ) -> None:
        try:
            body = engine.generate(
                "What is your name?",
                model=model,
                system="You are a helpful assistant named GraniteBot.",
                options={"temperature": 0, "num_predict": 100},
            )
        except Exception as e:
            checks.append(CheckResult(name="system_prompt_error", passed=False, detail=str(e)))
            return

        content = body.get("response", "")
        checks.append(CheckResult(
            name="system_prompt_has_response",
            passed=bool(content),
            expected="non-empty response",
            actual=content[:200],
        ))

        lower = content.lower()
        keywords = ("granitebot", "granite")
        found = [kw for kw in keywords if kw in lower]
        checks.append(CheckResult(
            name="system_prompt_keyword",
            passed=len(found) > 0,
            expected=f"at least one of {keywords}",
            actual=f"found: {found}" if found else content[:200],
        ))

    def _check_reproducibility(
        self, engine: OllamaEngine, model: str, checks: list[CheckResult],
    ) -> None:
        options = {"temperature": 0, "seed": 42, "num_predict": 50}

        try:
            body1 = engine.generate(
                "Count from 1 to 10.", model=model, options=options,
            )
        except Exception as e:
            checks.append(CheckResult(
                name="reproducibility_run1_error", passed=False, detail=str(e),
            ))
            return

        output1 = body1.get("response", "")
        checks.append(CheckResult(
            name="reproducibility_run1_has_response",
            passed=bool(output1),
            expected="non-empty response",
            actual=output1[:200],
        ))

        try:
            body2 = engine.generate(
                "Count from 1 to 10.", model=model, options=options,
            )
        except Exception as e:
            checks.append(CheckResult(
                name="reproducibility_run2_error", passed=False, detail=str(e),
            ))
            return

        output2 = body2.get("response", "")
        checks.append(CheckResult(
            name="reproducibility_run2_has_response",
            passed=bool(output2),
            expected="non-empty response",
            actual=output2[:200],
        ))

        checks.append(CheckResult(
            name="reproducibility_outputs_match",
            passed=output1 == output2,
            expected="identical outputs",
            actual="match" if output1 == output2 else f"differ: {output1[:100]!r} vs {output2[:100]!r}",
        ))

    def _check_small_context(
        self, engine: OllamaEngine, model: str, checks: list[CheckResult],
    ) -> None:
        try:
            body = engine.generate(
                "Tell me a joke.",
                model=model,
                options={"temperature": 0, "num_ctx": 128, "num_predict": 50},
            )
        except Exception as e:
            checks.append(CheckResult(
                name="small_context_error", passed=False, detail=str(e),
            ))
            return

        content = body.get("response", "")
        checks.append(CheckResult(
            name="small_context_has_response",
            passed=bool(content),
            expected="non-empty response (no crash with num_ctx=128)",
            actual=content[:200],
        ))

    def _check_generate_metrics(
        self, engine: OllamaEngine, model: str, checks: list[CheckResult],
    ) -> None:
        try:
            body = engine.generate(
                "Hello",
                model=model,
                options={"num_predict": 20},
            )
        except Exception as e:
            checks.append(CheckResult(
                name="generate_metrics_error", passed=False, detail=str(e),
            ))
            return

        checks.append(CheckResult(
            name="generate_metrics_total_duration",
            passed=body.get("total_duration", 0) > 0,
            expected="> 0",
            actual=body.get("total_duration"),
        ))
        checks.append(CheckResult(
            name="generate_metrics_eval_count",
            passed=body.get("eval_count", 0) > 0,
            expected="> 0",
            actual=body.get("eval_count"),
        ))
        checks.append(CheckResult(
            name="generate_metrics_eval_duration",
            passed=body.get("eval_duration", 0) > 0,
            expected="> 0",
            actual=body.get("eval_duration"),
        ))

    def _check_chat_metrics(
        self, engine: OllamaEngine, model: str, checks: list[CheckResult],
    ) -> None:
        try:
            body = engine.native_chat(
                [{"role": "user", "content": "Say hello."}],
                model=model,
                options={"num_predict": 20},
            )
        except Exception as e:
            checks.append(CheckResult(
                name="chat_metrics_error", passed=False, detail=str(e),
            ))
            return

        checks.append(CheckResult(
            name="chat_metrics_total_duration",
            passed=body.get("total_duration", 0) > 0,
            expected="> 0",
            actual=body.get("total_duration"),
        ))
        checks.append(CheckResult(
            name="chat_metrics_eval_count",
            passed=body.get("eval_count", 0) > 0,
            expected="> 0",
            actual=body.get("eval_count"),
        ))
        checks.append(CheckResult(
            name="chat_metrics_done_reason",
            passed=bool(body.get("done_reason")),
            expected="non-empty done_reason",
            actual=body.get("done_reason"),
        ))
