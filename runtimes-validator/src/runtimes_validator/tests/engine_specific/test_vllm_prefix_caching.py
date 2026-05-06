from __future__ import annotations

import time

from runtimes_validator.domain.models import CheckResult, TestResult
from runtimes_validator.engines.base import AbstractEngine
from runtimes_validator.engines.vllm import VllmEngine
from runtimes_validator.tests.base import AbstractValidationTest
from runtimes_validator.tests.registry import register_test

_SHARED_PREFIX = (
    "You are an expert geography tutor. You provide accurate, detailed answers "
    "about world geography, including countries, capitals, rivers, mountain "
    "ranges, climate zones, and cultural regions. When answering questions, "
    "you should reference specific facts and figures where possible. You are "
    "familiar with both physical and political geography. You know about "
    "population statistics, GDP rankings, and territorial disputes. You can "
    "discuss historical changes in borders, the formation of modern nation "
    "states, and the geographical factors that influenced major historical "
    "events. You should always be precise and avoid speculation. If you are "
    "unsure about a fact, say so rather than guessing. Your goal is to help "
    "the student build a strong mental model of how the world is organized "
    "geographically, politically, and economically."
)


@register_test("vllm_prefix_caching")
class PrefixCachingTest(AbstractValidationTest):
    """Validates generation correctness under vLLM prefix caching."""

    def test_id(self) -> str:
        return "vllm_prefix_caching"

    def test_name(self) -> str:
        return "vLLM Prefix Caching"

    def applicable_engines(self) -> list[str]:
        return ["vllm"]

    def run(self, engine: AbstractEngine, model: str) -> TestResult:
        assert isinstance(engine, VllmEngine)
        checks: list[CheckResult] = []
        start = time.time()

        if self._check_prefix_caching_config(engine, checks):
            self._check_prefix_caching_correctness(engine, model, checks)

        return TestResult(
            test_id=self.test_id(),
            test_name=self.test_name(),
            engine_id=engine.engine_id(),
            model=model,
            checks=checks,
            elapsed_seconds=time.time() - start,
        )

    def _check_prefix_caching_config(
        self,
        engine: VllmEngine,
        checks: list[CheckResult],
    ) -> bool:
        server_args = engine._config.extra.get("server_args", [])
        enabled = "--enable-prefix-caching" in server_args

        if not enabled:
            checks.append(
                CheckResult(
                    name="prefix_caching_enabled",
                    passed=True,
                    detail=(
                        "--enable-prefix-caching not in server_args; skipping prefix caching checks"
                    ),
                )
            )
            return False

        checks.append(
            CheckResult(
                name="prefix_caching_enabled",
                passed=True,
                expected="--enable-prefix-caching in server_args",
                actual="present",
            )
        )
        return True

    def _check_prefix_caching_correctness(
        self,
        engine: VllmEngine,
        model: str,
        checks: list[CheckResult],
    ) -> None:
        messages_q1 = [
            {"role": "system", "content": _SHARED_PREFIX},
            {"role": "user", "content": "What is the capital of Japan?"},
        ]
        messages_q2 = [
            {"role": "system", "content": _SHARED_PREFIX},
            {"role": "user", "content": "What is the longest river in Africa?"},
        ]

        # First request
        try:
            t1_start = time.time()
            resp1 = engine.chat(
                messages_q1,
                max_tokens=64,
                temperature=0.0,
            )
            t1_elapsed = time.time() - t1_start
        except Exception as e:
            checks.append(
                CheckResult(
                    name="prefix_first_request",
                    passed=False,
                    detail=str(e),
                )
            )
            return

        content1 = resp1.get("content") or ""
        checks.append(
            CheckResult(
                name="prefix_first_request",
                passed=bool(content1.strip()),
                expected="non-empty response",
                actual=content1[:200],
            )
        )

        # Second request (same prefix, different question)
        try:
            t2_start = time.time()
            resp2 = engine.chat(
                messages_q2,
                max_tokens=64,
                temperature=0.0,
            )
            t2_elapsed = time.time() - t2_start
        except Exception as e:
            checks.append(
                CheckResult(
                    name="prefix_reuse_request",
                    passed=False,
                    detail=str(e),
                )
            )
            return

        content2 = resp2.get("content") or ""
        checks.append(
            CheckResult(
                name="prefix_reuse_request",
                passed=bool(content2.strip()),
                expected="non-empty response",
                actual=content2[:200],
            )
        )

        # Different questions should produce different answers
        checks.append(
            CheckResult(
                name="prefix_responses_distinct",
                passed=content1.strip() != content2.strip(),
                expected="different responses for different questions",
                actual=("identical" if content1.strip() == content2.strip() else "distinct"),
            )
        )

        # Timing ratio (informational only)
        if t1_elapsed > 0:
            ratio = t2_elapsed / t1_elapsed
            checks.append(
                CheckResult(
                    name="prefix_timing_ratio",
                    passed=True,
                    expected="second request benefits from cached prefix",
                    actual=f"first={t1_elapsed:.2f}s, second={t2_elapsed:.2f}s, ratio={ratio:.2f}",
                )
            )
