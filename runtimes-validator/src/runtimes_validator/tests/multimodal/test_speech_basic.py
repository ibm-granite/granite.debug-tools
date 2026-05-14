from __future__ import annotations

import base64
import time
from typing import Any

from runtimes_validator.domain.models import CheckResult, TestResult
from runtimes_validator.engines.base import AbstractEngine
from runtimes_validator.tests.base import AbstractValidationTest
from runtimes_validator.tests.registry import register_test
from runtimes_validator.tests.resources import load_bytes


@register_test("speech_basic")
class SpeechBasicTest(AbstractValidationTest):
    """Transcribe a short spoken clip and check expected tokens appear."""

    def test_id(self) -> str:
        return "speech_basic"

    def test_name(self) -> str:
        return "Speech: basic transcription"

    def modalities(self) -> list[str]:
        return ["speech"]

    def run(self, engine: AbstractEngine, model: str) -> TestResult:
        start = time.time()
        checks: list[CheckResult] = []

        audio_bytes = load_bytes("speech/hello_granite.wav")
        b64 = base64.b64encode(audio_bytes).decode("ascii")
        data_url = f"data:audio/wav;base64,{b64}"

        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Can you transcribe the speech into a written format?",
                    },
                    {"type": "audio_url", "audio_url": {"url": data_url}},
                ],
            }
        ]

        try:
            response = engine.chat(messages)
            content = (response.get("content") or "").lower()
        except Exception as e:
            checks.append(CheckResult(name="speech_basic_error", passed=False, detail=str(e)))
            return TestResult(
                test_id=self.test_id(),
                test_name=self.test_name(),
                engine_id=engine.engine_id(),
                model=model,
                checks=checks,
                elapsed_seconds=time.time() - start,
            )

        for expected in ("hello", "granite"):
            checks.append(
                CheckResult(
                    name=f"response_contains_{expected}",
                    passed=expected in content,
                    expected=expected,
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
