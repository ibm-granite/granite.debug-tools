from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from runtimes_validator.domain.models import EngineInfo
from runtimes_validator.engines.base import AbstractEngine, EngineConfig
from runtimes_validator.tests.common.test_multi_tool_calling import MultiToolCallingTest


ChatFn = Callable[[list[dict[str, Any]]], dict[str, Any]]


class StubEngine(AbstractEngine):
    """Minimal engine stub that dispatches chat() to a user-supplied callable."""

    def __init__(self, chat_fn: ChatFn) -> None:
        self._chat_fn = chat_fn
        self._config = EngineConfig()

    def engine_id(self) -> str:
        return "stub"

    def start(self, model: str) -> None:  # pragma: no cover - lifecycle no-op
        return None

    def stop(self) -> None:  # pragma: no cover - lifecycle no-op
        return None

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> dict[str, Any]:
        return self._chat_fn(messages)

    def health_check(self) -> bool:
        return True

    def get_info(self) -> EngineInfo:
        return EngineInfo(engine_id="stub", version="0", mode="external", base_url="http://stub")


def _tool_call(name: str, args: dict[str, Any], call_id: str = "call_1") -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


def _by_name(result: Any, name: str) -> Any:
    for check in result.checks:
        if check.name == name:
            return check
    raise AssertionError(f"check {name!r} not found; got {[c.name for c in result.checks]}")


def test_auto_selection_picks_convert_currency() -> None:
    def chat(messages: list[dict[str, Any]]) -> dict[str, Any]:
        user_text = " ".join(m.get("content", "") for m in messages if m.get("role") == "user")
        if "USD to EUR" in user_text:
            return {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    _tool_call(
                        "convert_currency",
                        {"amount": 100, "from_currency": "USD", "to_currency": "EUR"},
                    )
                ],
                "finish_reason": "tool_calls",
                "usage": None,
            }
        if "weather" in user_text.lower():
            return {
                "role": "assistant",
                "content": None,
                "tool_calls": [_tool_call("get_weather", {"location": "Paris"})],
                "finish_reason": "tool_calls",
                "usage": None,
            }
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [_tool_call("get_current_time", {"timezone": "Asia/Tokyo"})],
            "finish_reason": "tool_calls",
            "usage": None,
        }

    # Stub that always serves all three scenarios consistently:
    engine = StubEngine(chat)
    result = MultiToolCallingTest().run(engine, "stub-model")

    assert _by_name(result, "multi_auto_has_tool_calls").passed
    assert _by_name(result, "multi_auto_correct_tool").passed
    assert _by_name(result, "multi_auto_args_present").passed


def test_sequential_roundtrip_success() -> None:
    def chat(messages: list[dict[str, Any]]) -> dict[str, Any]:
        has_tool_result = any(m.get("role") == "tool" for m in messages)
        if has_tool_result:
            return {
                "role": "assistant",
                "content": None,
                "tool_calls": [_tool_call("get_current_time", {"timezone": "Asia/Tokyo"})],
                "finish_reason": "tool_calls",
                "usage": None,
            }
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [_tool_call("get_weather", {"location": "Paris"})],
            "finish_reason": "tool_calls",
            "usage": None,
        }

    engine = StubEngine(chat)
    result = MultiToolCallingTest().run(engine, "stub-model")

    assert _by_name(result, "seq_first_call_name").passed
    assert _by_name(result, "seq_second_call_name").passed
    assert _by_name(result, "seq_second_args_timezone").passed


def test_parallel_multi_call_success() -> None:
    def chat(messages: list[dict[str, Any]]) -> dict[str, Any]:
        user_text = " ".join(m.get("content", "") for m in messages if m.get("role") == "user")
        if "AND the current time" in user_text:
            return {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    _tool_call("get_weather", {"location": "Paris"}, "call_1"),
                    _tool_call("get_current_time", {"timezone": "Asia/Tokyo"}, "call_2"),
                ],
                "finish_reason": "tool_calls",
                "usage": None,
            }
        # Other sub-checks: a benign single call so we isolate the parallel assertion.
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [_tool_call("get_weather", {"location": "Paris"})],
            "finish_reason": "tool_calls",
            "usage": None,
        }

    engine = StubEngine(chat)
    result = MultiToolCallingTest().run(engine, "stub-model")

    parallel_check = _by_name(result, "parallel_multi_call")
    assert parallel_check.passed
    assert parallel_check.detail == ""
    assert set(parallel_check.actual) == {"get_weather", "get_current_time"}


def test_parallel_degrades_gracefully_on_single_call() -> None:
    def chat(messages: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [_tool_call("get_weather", {"location": "Paris"})],
            "finish_reason": "tool_calls",
            "usage": None,
        }

    engine = StubEngine(chat)
    result = MultiToolCallingTest().run(engine, "stub-model")

    parallel_check = _by_name(result, "parallel_multi_call")
    assert parallel_check.passed
    assert "degraded" in parallel_check.detail


def test_engine_exception_appends_error_check() -> None:
    calls = {"n": 0}

    def chat(messages: list[dict[str, Any]]) -> dict[str, Any]:
        calls["n"] += 1
        raise RuntimeError("boom")

    engine = StubEngine(chat)
    result = MultiToolCallingTest().run(engine, "stub-model")

    error_names = {c.name for c in result.checks if not c.passed}
    assert "multi_auto_error" in error_names
    assert "seq_first_error" in error_names
    assert "parallel_error" in error_names
    assert result.passed is False
