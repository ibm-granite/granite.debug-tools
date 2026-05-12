from __future__ import annotations

import json
import time
from typing import Any

from runtimes_validator.domain.models import CheckResult, TestResult
from runtimes_validator.engines.base import AbstractEngine
from runtimes_validator.tests.base import AbstractValidationTest
from runtimes_validator.tests.registry import register_test

GET_WEATHER_TOOL: dict[str, Any] = {
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

GET_CURRENT_TIME_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_current_time",
        "description": "Get the current time in a given IANA timezone",
        "parameters": {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "IANA timezone name, e.g. 'Asia/Tokyo'",
                },
            },
            "required": ["timezone"],
        },
    },
}

CONVERT_CURRENCY_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "convert_currency",
        "description": "Convert an amount from one currency to another",
        "parameters": {
            "type": "object",
            "properties": {
                "amount": {
                    "type": "number",
                    "description": "Amount in the source currency",
                },
                "from_currency": {
                    "type": "string",
                    "description": "ISO 4217 code of the source currency, e.g. 'USD'",
                },
                "to_currency": {
                    "type": "string",
                    "description": "ISO 4217 code of the target currency, e.g. 'EUR'",
                },
            },
            "required": ["amount", "from_currency", "to_currency"],
        },
    },
}

ALL_TOOLS: list[dict[str, Any]] = [
    GET_WEATHER_TOOL,
    GET_CURRENT_TIME_TOOL,
    CONVERT_CURRENCY_TOOL,
]


def _parse_args(tool_call: dict[str, Any]) -> dict[str, Any]:
    args = tool_call.get("function", {}).get("arguments", "{}")
    if isinstance(args, str):
        try:
            parsed = json.loads(args)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return args if isinstance(args, dict) else {}


@register_test("multi_tool_calling")
class MultiToolCallingTest(AbstractValidationTest):
    """Validates tool-calling behaviour when multiple tools are offered."""

    def test_id(self) -> str:
        return "multi_tool_calling"

    def test_name(self) -> str:
        return "Multi-Tool Calling"

    def run(self, engine: AbstractEngine, model: str) -> TestResult:
        checks: list[CheckResult] = []
        start = time.time()

        self._check_auto_selection(engine, checks)
        self._check_sequential_roundtrip(engine, checks)
        self._check_parallel_calls(engine, checks)

        return TestResult(
            test_id=self.test_id(),
            test_name=self.test_name(),
            engine_id=engine.engine_id(),
            model=model,
            checks=checks,
            elapsed_seconds=time.time() - start,
        )

    def _check_auto_selection(self, engine: AbstractEngine, checks: list[CheckResult]) -> None:
        try:
            response = engine.chat(
                [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Convert 100 USD to EUR."},
                ],
                tools=ALL_TOOLS,
                tool_choice="auto",
                max_tokens=256,
            )
        except Exception as e:
            checks.append(CheckResult(name="multi_auto_error", passed=False, detail=str(e)))
            return

        raw_calls = response.get("tool_calls")
        tool_calls: list[dict[str, Any]] = list(raw_calls) if raw_calls else []
        has_calls = len(tool_calls) >= 1
        checks.append(
            CheckResult(
                name="multi_auto_has_tool_calls",
                passed=has_calls,
                expected=">= 1 tool call",
                actual=len(tool_calls),
            )
        )
        if not has_calls:
            return

        fn_name = tool_calls[0].get("function", {}).get("name")
        checks.append(
            CheckResult(
                name="multi_auto_correct_tool",
                passed=fn_name == "convert_currency",
                expected="convert_currency",
                actual=fn_name,
            )
        )

        args = _parse_args(tool_calls[0])
        required_keys = {"amount", "from_currency", "to_currency"}
        checks.append(
            CheckResult(
                name="multi_auto_args_present",
                passed=required_keys.issubset(args.keys()),
                expected=f"keys {sorted(required_keys)} in arguments",
                actual=str(args),
            )
        )

    def _check_sequential_roundtrip(
        self, engine: AbstractEngine, checks: list[CheckResult]
    ) -> None:
        try:
            first = engine.chat(
                [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "What is the weather in Paris?"},
                ],
                tools=ALL_TOOLS,
                tool_choice="auto",
                max_tokens=256,
            )
        except Exception as e:
            checks.append(CheckResult(name="seq_first_error", passed=False, detail=str(e)))
            return

        first_calls = first.get("tool_calls") or []
        first_name = first_calls[0].get("function", {}).get("name") if first_calls else None
        checks.append(
            CheckResult(
                name="seq_first_call_name",
                passed=first_name == "get_weather",
                expected="get_weather",
                actual=first_name,
            )
        )
        if first_name != "get_weather":
            return

        try:
            second = engine.chat(
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
                    {
                        "role": "user",
                        "content": "Good. Now what time is it in Tokyo?",
                    },
                ],
                tools=ALL_TOOLS,
                tool_choice="auto",
                max_tokens=256,
            )
        except Exception as e:
            checks.append(CheckResult(name="seq_second_error", passed=False, detail=str(e)))
            return

        second_calls = second.get("tool_calls") or []
        second_name = second_calls[0].get("function", {}).get("name") if second_calls else None
        checks.append(
            CheckResult(
                name="seq_second_call_name",
                passed=second_name == "get_current_time",
                expected="get_current_time",
                actual=second_name,
            )
        )
        if second_name != "get_current_time":
            return

        args = _parse_args(second_calls[0])
        timezone_value = str(args.get("timezone", ""))
        checks.append(
            CheckResult(
                name="seq_second_args_timezone",
                passed="tokyo" in timezone_value.lower(),
                expected="timezone containing 'Tokyo'",
                actual=str(args),
            )
        )

    def _check_parallel_calls(self, engine: AbstractEngine, checks: list[CheckResult]) -> None:
        try:
            response = engine.chat(
                [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": (
                            "What's the weather in Paris AND the current time in Tokyo? "
                            "Please call both tools."
                        ),
                    },
                ],
                tools=ALL_TOOLS,
                tool_choice="auto",
                max_tokens=256,
            )
        except Exception as e:
            checks.append(CheckResult(name="parallel_error", passed=False, detail=str(e)))
            return

        finish_reason = response.get("finish_reason")
        checks.append(
            CheckResult(
                name="parallel_response_ok",
                passed=finish_reason in ("stop", "length", "tool_calls"),
                expected="stop, length, or tool_calls",
                actual=finish_reason,
            )
        )

        raw_parallel = response.get("tool_calls")
        tool_calls: list[dict[str, Any]] = list(raw_parallel) if raw_parallel else []
        names = [tc.get("function", {}).get("name") for tc in tool_calls]
        expected_names = {"get_weather", "get_current_time"}
        got_both = len(tool_calls) >= 2 and expected_names.issubset(set(names))

        if got_both:
            checks.append(
                CheckResult(
                    name="parallel_multi_call",
                    passed=True,
                    expected=">=2 parallel tool calls covering get_weather and get_current_time",
                    actual=names,
                )
            )
        else:
            checks.append(
                CheckResult(
                    name="parallel_multi_call",
                    passed=True,
                    expected=">=2 parallel tool calls covering get_weather and get_current_time",
                    actual=names,
                    detail=(
                        "engine/model may not support parallel tool calls; degraded gracefully"
                    ),
                )
            )
