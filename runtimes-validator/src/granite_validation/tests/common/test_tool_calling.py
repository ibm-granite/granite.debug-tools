from __future__ import annotations

import json
import time
from typing import Any

from granite_validation.domain.models import CheckResult, TestResult
from granite_validation.engines.base import AbstractEngine
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


@register_test("tool_calling")
class ToolCallingTest(AbstractValidationTest):
    """Validates tool calling behaviour across all engines."""

    def test_id(self) -> str:
        return "tool_calling"

    def test_name(self) -> str:
        return "Tool Calling"

    def run(self, engine: AbstractEngine, model: str) -> TestResult:
        checks: list[CheckResult] = []
        start = time.time()

        self._check_required(engine, checks)
        self._check_auto(engine, checks)
        self._check_roundtrip(engine, checks)

        return TestResult(
            test_id=self.test_id(),
            test_name=self.test_name(),
            engine_id=engine.engine_id(),
            model=model,
            checks=checks,
            elapsed_seconds=time.time() - start,
        )

    def _check_required(
        self, engine: AbstractEngine, checks: list[CheckResult]
    ) -> None:
        try:
            response = engine.chat(
                [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "What is the weather in Paris?"},
                ],
                tools=[WEATHER_TOOL],
                tool_choice="required",
                max_tokens=256,
            )
        except Exception as e:
            checks.append(CheckResult(name="required_error", passed=False, detail=str(e)))
            return

        tool_calls = response.get("tool_calls")
        has_calls = bool(tool_calls) and len(tool_calls) >= 1
        checks.append(CheckResult(
            name="required_has_tool_calls",
            passed=has_calls,
            expected=">= 1 tool call",
            actual=len(tool_calls) if tool_calls else 0,
        ))
        if not has_calls:
            return

        tc = tool_calls[0]
        fn_name = tc.get("function", {}).get("name")
        checks.append(CheckResult(
            name="required_function_name",
            passed=fn_name == "get_weather",
            expected="get_weather",
            actual=fn_name,
        ))

        args = tc.get("function", {}).get("arguments", "{}")
        if isinstance(args, str):
            args = json.loads(args)
        checks.append(CheckResult(
            name="required_has_location_arg",
            passed="location" in args,
            expected="'location' in arguments",
            actual=str(args),
        ))

    def _check_auto(
        self, engine: AbstractEngine, checks: list[CheckResult]
    ) -> None:
        try:
            response = engine.chat(
                [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "What is the weather in Tokyo right now?"},
                ],
                tools=[WEATHER_TOOL],
                tool_choice="auto",
                max_tokens=256,
            )
        except Exception as e:
            checks.append(CheckResult(name="auto_error", passed=False, detail=str(e)))
            return

        checks.append(CheckResult(
            name="auto_valid_response",
            passed=response.get("finish_reason") in ("stop", "length", "tool_calls"),
            expected="stop, length, or tool_calls",
            actual=response.get("finish_reason"),
        ))

        tool_calls = response.get("tool_calls")
        if tool_calls:
            fn_name = tool_calls[0].get("function", {}).get("name")
            checks.append(CheckResult(
                name="auto_tool_name_if_called",
                passed=fn_name == "get_weather",
                expected="get_weather",
                actual=fn_name,
            ))

    def _check_roundtrip(
        self, engine: AbstractEngine, checks: list[CheckResult]
    ) -> None:
        try:
            response = engine.chat(
                [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "What is the weather in Paris?"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": json.dumps({"location": "Paris"}),
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_1",
                        "content": json.dumps(
                            {"temperature": 18, "condition": "sunny", "humidity": 65}
                        ),
                    },
                ],
                tools=[WEATHER_TOOL],
                max_tokens=128,
            )
        except Exception as e:
            checks.append(CheckResult(name="roundtrip_error", passed=False, detail=str(e)))
            return

        checks.append(CheckResult(
            name="roundtrip_role",
            passed=response.get("role") == "assistant",
            expected="assistant",
            actual=response.get("role"),
        ))
        content = response.get("content") or ""
        checks.append(CheckResult(
            name="roundtrip_has_content",
            passed=len(content) > 0,
            expected="non-empty content after tool result",
            actual=content[:200],
        ))
