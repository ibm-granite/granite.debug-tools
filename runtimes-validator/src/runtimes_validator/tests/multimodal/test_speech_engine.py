from __future__ import annotations

import base64
import time
from typing import Any

from runtimes_validator.domain.models import CheckResult, TestResult
from runtimes_validator.engines.base import AbstractEngine
from runtimes_validator.tests.base import AbstractValidationTest
from runtimes_validator.tests.multimodal._engine_checks import (
    append_ingestion_check,
    append_shape_checks,
)
from runtimes_validator.tests.registry import register_test
from runtimes_validator.tests.resources import load_bytes

PROMPT_TEXT = "Respond briefly to the attached audio."


@register_test("speech_engine")
class SpeechEnginePlumbingTest(AbstractValidationTest):
    """Verify the engine accepts and ingests an audio payload.

    Asserts response shape and that the audio was actually consumed as input
    (prompt_tokens for an audio+text message is greater than for text alone).
    Does not assert anything about the content of the model's reply.
    """

    def test_id(self) -> str:
        return "speech_engine"

    def test_name(self) -> str:
        return "Speech: engine plumbing"

    def modalities(self) -> list[str]:
        return ["speech"]

    def run(self, engine: AbstractEngine, model: str) -> TestResult:
        start = time.time()
        checks: list[CheckResult] = []

        audio_bytes = load_bytes("speech/hello_granite.wav")
        b64 = base64.b64encode(audio_bytes).decode("ascii")
        data_url = f"data:audio/wav;base64,{b64}"

        mm_messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT_TEXT},
                    {"type": "audio_url", "audio_url": {"url": data_url}},
                ],
            }
        ]

        try:
            mm_response = engine.chat(mm_messages, max_tokens=32)
        except Exception as e:
            checks.append(CheckResult(name="speech_engine_error", passed=False, detail=str(e)))
            return TestResult(
                test_id=self.test_id(),
                test_name=self.test_name(),
                engine_id=engine.engine_id(),
                model=model,
                checks=checks,
                elapsed_seconds=time.time() - start,
            )

        append_shape_checks(mm_response, checks)
        append_ingestion_check(engine, mm_response, PROMPT_TEXT, "audio", checks)

        return TestResult(
            test_id=self.test_id(),
            test_name=self.test_name(),
            engine_id=engine.engine_id(),
            model=model,
            checks=checks,
            elapsed_seconds=time.time() - start,
        )
