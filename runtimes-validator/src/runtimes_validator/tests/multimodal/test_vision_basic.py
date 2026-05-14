from __future__ import annotations

import base64
import time
from typing import Any

from runtimes_validator.domain.models import CheckResult, TestResult
from runtimes_validator.engines.base import AbstractEngine
from runtimes_validator.tests.base import AbstractValidationTest
from runtimes_validator.tests.registry import register_test
from runtimes_validator.tests.resources import load_bytes


@register_test("vision_basic")
class VisionBasicTest(AbstractValidationTest):
    """Smoke test: send a simple image and expect a plausible description."""

    def test_id(self) -> str:
        return "vision_basic"

    def test_name(self) -> str:
        return "Vision: basic image description"

    def modalities(self) -> list[str]:
        return ["vision"]

    def run(self, engine: AbstractEngine, model: str) -> TestResult:
        start = time.time()
        checks: list[CheckResult] = []

        image_bytes = load_bytes("vision/shapes.png")
        b64 = base64.b64encode(image_bytes).decode("ascii")
        data_url = f"data:image/png;base64,{b64}"

        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What shape is shown? Answer in one word."},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ]

        try:
            response = engine.chat(messages)
            content = (response.get("content") or "").lower()
        except Exception as e:
            checks.append(CheckResult(name="vision_basic_error", passed=False, detail=str(e)))
            return TestResult(
                test_id=self.test_id(),
                test_name=self.test_name(),
                engine_id=engine.engine_id(),
                model=model,
                checks=checks,
                elapsed_seconds=time.time() - start,
            )

        checks.append(
            CheckResult(
                name="response_non_empty",
                passed=bool(content.strip()),
                actual=content[:200],
            )
        )
        checks.append(
            CheckResult(
                name="response_contains_circle",
                passed="circle" in content,
                expected="circle",
                actual=content[:200],
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
