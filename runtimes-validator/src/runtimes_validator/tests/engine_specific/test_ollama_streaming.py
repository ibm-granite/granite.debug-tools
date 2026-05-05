from __future__ import annotations

import time
from typing import Any

from runtimes_validator.domain.models import CheckResult, TestResult
from runtimes_validator.engines.base import AbstractEngine
from runtimes_validator.engines.ollama import OllamaEngine
from runtimes_validator.tests.base import AbstractValidationTest
from runtimes_validator.tests.common.test_special_token_leakage import (
    GRANITE_SPECIAL_TOKENS,
    _check_no_special_tokens,
)
from runtimes_validator.tests.registry import register_test

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
                    "description": "The city name",
                },
            },
            "required": ["location"],
        },
    },
}


@register_test("ollama_streaming")
class OllamaStreamingTest(AbstractValidationTest):
    """Validates streaming responses via Ollama's native API."""

    def test_id(self) -> str:
        return "ollama_streaming"

    def test_name(self) -> str:
        return "Ollama Streaming"

    def applicable_engines(self) -> list[str]:
        return ["ollama"]

    def run(self, engine: AbstractEngine, model: str) -> TestResult:
        assert isinstance(engine, OllamaEngine)
        checks: list[CheckResult] = []
        start = time.time()

        self._check_generate_streaming(engine, model, checks)
        self._check_chat_streaming(engine, model, checks)
        self._check_tool_streaming(engine, model, checks)
        self._check_streaming_no_leakage(engine, model, checks)

        return TestResult(
            test_id=self.test_id(),
            test_name=self.test_name(),
            engine_id=engine.engine_id(),
            model=model,
            checks=checks,
            elapsed_seconds=time.time() - start,
        )

    def _check_generate_streaming(
        self, engine: OllamaEngine, model: str, checks: list[CheckResult],
    ) -> None:
        try:
            chunks = list(engine.generate_stream(
                "Why is the sky blue? Be brief but factual.",
                model=model,
                options={"temperature": 0, "seed": 123, "num_predict": 100},
            ))
        except Exception as e:
            checks.append(CheckResult(
                name="gen_streaming_error", passed=False, detail=str(e),
            ))
            return

        checks.append(CheckResult(
            name="gen_streaming_multiple_chunks",
            passed=len(chunks) >= 2,
            expected=">= 2 chunks",
            actual=len(chunks),
        ))

        full_text = "".join(c.get("response", "") for c in chunks)
        checks.append(CheckResult(
            name="gen_streaming_nonempty",
            passed=bool(full_text.strip()),
            expected="non-empty concatenated response",
            actual=full_text[:200],
        ))

        lower = full_text.lower()
        keywords = ("scatter", "wavelength", "atmosphere", "light", "blue")
        found = [kw for kw in keywords if kw in lower]
        checks.append(CheckResult(
            name="gen_streaming_keywords",
            passed=len(found) > 0,
            expected=f"at least one of {keywords}",
            actual=f"found: {found}" if found else full_text[:200],
        ))

    def _check_chat_streaming(
        self, engine: OllamaEngine, model: str, checks: list[CheckResult],
    ) -> None:
        try:
            chunks = list(engine.native_chat_stream(
                [{"role": "user", "content": "Say hello and be brief."}],
                model=model,
                options={"temperature": 0, "num_predict": 50},
            ))
        except Exception as e:
            checks.append(CheckResult(
                name="chat_streaming_error", passed=False, detail=str(e),
            ))
            return

        checks.append(CheckResult(
            name="chat_streaming_multiple_chunks",
            passed=len(chunks) >= 2,
            expected=">= 2 chunks",
            actual=len(chunks),
        ))

        full_text = "".join(
            c.get("message", {}).get("content", "") for c in chunks
        )
        checks.append(CheckResult(
            name="chat_streaming_nonempty",
            passed=bool(full_text.strip()),
            expected="non-empty concatenated content",
            actual=full_text[:200],
        ))

    def _check_tool_streaming(
        self, engine: OllamaEngine, model: str, checks: list[CheckResult],
    ) -> None:
        try:
            chunks = list(engine.native_chat_stream(
                [{"role": "user", "content": "Get the weather in London."}],
                model=model,
                tools=[WEATHER_TOOL],
                options={"temperature": 0},
            ))
        except Exception as e:
            checks.append(CheckResult(
                name="tool_streaming_error", passed=False, detail=str(e),
            ))
            return

        tool_chunks = [
            c for c in chunks
            if c.get("message", {}).get("tool_calls")
        ]
        checks.append(CheckResult(
            name="tool_streaming_has_tool_calls",
            passed=len(tool_chunks) > 0,
            expected="> 0 chunks with tool_calls",
            actual=len(tool_chunks),
        ))

    def _check_streaming_no_leakage(
        self, engine: OllamaEngine, model: str, checks: list[CheckResult],
    ) -> None:
        try:
            chunks = list(engine.native_chat_stream(
                [{"role": "user", "content": "Write a haiku about mountains."}],
                model=model,
                options={"temperature": 0, "num_predict": 100},
            ))
        except Exception as e:
            checks.append(CheckResult(
                name="streaming_leakage_error", passed=False, detail=str(e),
            ))
            return

        full_text = "".join(
            c.get("message", {}).get("content", "") for c in chunks
        )
        leaked = _check_no_special_tokens(full_text, GRANITE_SPECIAL_TOKENS)
        checks.append(CheckResult(
            name="streaming_no_special_token_leakage",
            passed=leaked is None,
            expected="no special tokens in streaming output",
            actual=f"found: {leaked}" if leaked else "clean",
        ))
