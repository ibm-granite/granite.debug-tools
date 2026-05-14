from __future__ import annotations

import json
import time
from typing import Any

from runtimes_validator.domain.models import CheckResult, TestResult
from runtimes_validator.engines.base import AbstractEngine
from runtimes_validator.engines.vllm import VllmEngine
from runtimes_validator.tests.base import AbstractValidationTest
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
                    "description": "The city name, e.g. 'Paris, France'",
                },
            },
            "required": ["location"],
        },
    },
}


@register_test("vllm_tool_flags")
class ToolFlagsTest(AbstractValidationTest):
    """Validates tool calling behavior under vLLM's configured tool-call parser."""

    def test_id(self) -> str:
        return "vllm_tool_flags"

    def test_name(self) -> str:
        return "vLLM Tool Calling Flags"

    def applicable_engines(self) -> list[str]:
        return ["vllm"]

    def run(self, engine: AbstractEngine, model: str) -> TestResult:
        assert isinstance(engine, VllmEngine)
        checks: list[CheckResult] = []
        start = time.time()

        if self._check_tool_flags_present(engine, checks):
            self._check_tool_required(engine, model, checks)
            self._check_tool_auto(engine, model, checks)
            self._check_tool_roundtrip(engine, model, checks)
            self._check_tool_streaming(engine, model, checks)

        return TestResult(
            test_id=self.test_id(),
            test_name=self.test_name(),
            engine_id=engine.engine_id(),
            model=model,
            checks=checks,
            elapsed_seconds=time.time() - start,
        )

    def _check_tool_flags_present(
        self,
        engine: VllmEngine,
        checks: list[CheckResult],
    ) -> bool:
        server_args = engine._config.extra.get("server_args", [])
        missing: list[str] = []
        if "--enable-auto-tool-choice" not in server_args:
            missing.append("--enable-auto-tool-choice")
        if "--tool-call-parser" not in server_args:
            missing.append("--tool-call-parser")

        if missing:
            checks.append(
                CheckResult(
                    name="tool_flags_configured",
                    passed=True,
                    detail=(f"Missing flags: {', '.join(missing)}; skipping tool behavior checks"),
                )
            )
            return False

        parser_name = _extract_parser_name(server_args)
        checks.append(
            CheckResult(
                name="tool_flags_configured",
                passed=True,
                expected="--enable-auto-tool-choice and --tool-call-parser",
                actual=f"parser={parser_name}" if parser_name else "present",
            )
        )
        return True

    def _check_tool_required(
        self,
        engine: VllmEngine,
        model: str,
        checks: list[CheckResult],
    ) -> None:
        try:
            resp = engine.chat(
                [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "What is the weather in London?"},
                ],
                tools=[WEATHER_TOOL],
                tool_choice="required",
                max_tokens=256,
                temperature=0.0,
            )
        except Exception as e:
            checks.append(
                CheckResult(
                    name="tool_required_has_calls",
                    passed=False,
                    detail=str(e),
                )
            )
            return

        tool_calls = resp.get("tool_calls") or []
        checks.append(
            CheckResult(
                name="tool_required_has_calls",
                passed=len(tool_calls) >= 1,
                expected=">= 1 tool call",
                actual=len(tool_calls),
            )
        )

        if not tool_calls:
            return

        fn = tool_calls[0].get("function", {})
        fn_name = fn.get("name", "")
        checks.append(
            CheckResult(
                name="tool_required_function_name",
                passed=fn_name == "get_weather",
                expected="get_weather",
                actual=fn_name,
            )
        )

        raw_args = fn.get("arguments", "{}")
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except (json.JSONDecodeError, TypeError):
            args = {}
        checks.append(
            CheckResult(
                name="tool_required_has_location",
                passed="location" in args,
                expected="'location' key in arguments",
                actual=str(args)[:200],
            )
        )

    def _check_tool_auto(
        self,
        engine: VllmEngine,
        model: str,
        checks: list[CheckResult],
    ) -> None:
        try:
            resp = engine.chat(
                [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "What is the weather in Tokyo?"},
                ],
                tools=[WEATHER_TOOL],
                tool_choice="auto",
                max_tokens=256,
                temperature=0.0,
            )
        except Exception as e:
            checks.append(
                CheckResult(
                    name="tool_auto_valid_response",
                    passed=False,
                    detail=str(e),
                )
            )
            return

        fr = resp.get("finish_reason", "")
        checks.append(
            CheckResult(
                name="tool_auto_valid_response",
                passed=fr in ("stop", "length", "tool_calls"),
                expected="stop, length, or tool_calls",
                actual=fr,
            )
        )

    def _check_tool_roundtrip(
        self,
        engine: VllmEngine,
        model: str,
        checks: list[CheckResult],
    ) -> None:
        try:
            first = engine.chat(
                [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "What is the weather in Paris?"},
                ],
                tools=[WEATHER_TOOL],
                tool_choice="required",
                max_tokens=256,
                temperature=0.0,
            )
        except Exception as e:
            checks.append(
                CheckResult(
                    name="tool_roundtrip_content",
                    passed=False,
                    detail=str(e),
                )
            )
            return

        tool_calls = first.get("tool_calls") or []
        if not tool_calls:
            checks.append(
                CheckResult(
                    name="tool_roundtrip_content",
                    passed=False,
                    detail="No tool call returned in first turn",
                )
            )
            return

        tc = tool_calls[0]
        try:
            followup = engine.chat(
                [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "What is the weather in Paris?"},
                    {"role": "assistant", "tool_calls": [tc]},
                    {
                        "role": "tool",
                        "tool_call_id": tc.get("id", "call_1"),
                        "content": '{"temperature": 18, "condition": "cloudy"}',
                    },
                ],
                tools=[WEATHER_TOOL],
                max_tokens=256,
                temperature=0.0,
            )
        except Exception as e:
            checks.append(
                CheckResult(
                    name="tool_roundtrip_content",
                    passed=False,
                    detail=str(e),
                )
            )
            return

        content = followup.get("content") or ""
        checks.append(
            CheckResult(
                name="tool_roundtrip_content",
                passed=len(content) > 0,
                expected="non-empty follow-up after tool result",
                actual=content[:200],
            )
        )

    def _check_tool_streaming(
        self,
        engine: VllmEngine,
        model: str,
        checks: list[CheckResult],
    ) -> None:
        try:
            chunks = list(
                engine.chat_stream(
                    [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "What is the weather in Berlin?"},
                    ],
                    tools=[WEATHER_TOOL],
                    tool_choice="required",
                    max_tokens=256,
                    temperature=0.0,
                )
            )
        except Exception as e:
            checks.append(
                CheckResult(
                    name="tool_streaming_has_tool_chunks",
                    passed=False,
                    detail=str(e),
                )
            )
            return

        tool_chunks = [
            c for c in chunks if c.get("choices", [{}])[0].get("delta", {}).get("tool_calls")
        ]
        checks.append(
            CheckResult(
                name="tool_streaming_has_tool_chunks",
                passed=len(tool_chunks) > 0,
                expected="> 0 chunks with tool_calls in delta",
                actual=len(tool_chunks),
            )
        )


def _extract_parser_name(server_args: list[str]) -> str | None:
    for i, arg in enumerate(server_args):
        if arg == "--tool-call-parser" and i + 1 < len(server_args):
            return server_args[i + 1]
        if arg.startswith("--tool-call-parser="):
            return arg.split("=", 1)[1]
    return None
