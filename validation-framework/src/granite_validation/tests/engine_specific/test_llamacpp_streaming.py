from __future__ import annotations

import time
from typing import Any

from granite_validation.domain.models import CheckResult, TestResult
from granite_validation.engines.base import AbstractEngine
from granite_validation.engines.llamacpp import LlamaCppEngine
from granite_validation.tests.base import AbstractValidationTest
from granite_validation.tests.registry import register_test

WEATHER_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city name, e.g. 'Paris, France'",
                },
            },
            "required": ["location"],
        },
    },
}


@register_test("llamacpp_streaming")
class StreamingTest(AbstractValidationTest):
    """Validates streaming chat completions on llama.cpp server."""

    def test_id(self) -> str:
        return "llamacpp_streaming"

    def test_name(self) -> str:
        return "llama.cpp Streaming"

    def applicable_engines(self) -> list[str]:
        return ["llamacpp"]

    def run(self, engine: AbstractEngine, model: str) -> TestResult:
        assert isinstance(engine, LlamaCppEngine)
        checks: list[CheckResult] = []
        start = time.time()

        self._check_chat_streaming(engine, checks)
        self._check_tool_streaming(engine, checks)

        return TestResult(
            test_id=self.test_id(),
            test_name=self.test_name(),
            engine_id=engine.engine_id(),
            model=model,
            checks=checks,
            elapsed_seconds=time.time() - start,
        )

    def _check_chat_streaming(
        self, engine: LlamaCppEngine, checks: list[CheckResult],
    ) -> None:
        try:
            chunks = list(engine.chat_stream(
                [
                    {"role": "system", "content": "Be brief."},
                    {"role": "user", "content": "Say hello."},
                ],
                max_tokens=32,
                temperature=0.0,
            ))
        except Exception as e:
            checks.append(CheckResult(
                name="streaming_error", passed=False, detail=str(e),
            ))
            return

        checks.append(CheckResult(
            name="streaming_multiple_chunks",
            passed=len(chunks) > 1,
            expected="> 1 chunk",
            actual=len(chunks),
        ))

        data_chunks = [c for c in chunks if c.get("choices")]
        if data_chunks:
            last = data_chunks[-1]
            fr = last["choices"][0].get("finish_reason")
            checks.append(CheckResult(
                name="streaming_last_chunk_finish_reason",
                passed=fr in ("stop", "length"),
                expected="stop or length",
                actual=fr,
            ))

    def _check_tool_streaming(
        self, engine: LlamaCppEngine, checks: list[CheckResult],
    ) -> None:
        try:
            chunks = list(engine.chat_stream(
                [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "What is the weather in London?"},
                ],
                tools=[WEATHER_TOOL],
                tool_choice="required",
                max_tokens=256,
                temperature=0.0,
            ))
        except Exception as e:
            checks.append(CheckResult(
                name="streaming_tool_error", passed=False, detail=str(e),
            ))
            return

        tool_call_chunks = [
            c for c in chunks
            if c.get("choices")
            and c["choices"][0].get("delta", {}).get("tool_calls")
        ]
        checks.append(CheckResult(
            name="streaming_tool_has_tool_chunks",
            passed=len(tool_call_chunks) > 0,
            expected="> 0 tool call chunks",
            actual=len(tool_call_chunks),
        ))
