"""Unit tests for the multimodal engine-plumbing validation tests."""

from unittest.mock import MagicMock

from runtimes_validator.tests.multimodal.test_speech_engine import SpeechEnginePlumbingTest
from runtimes_validator.tests.multimodal.test_vision_engine import VisionEnginePlumbingTest


def _scripted_engine(mm_prompt_tokens: int | None, text_prompt_tokens: int = 10):
    """Return a mock engine whose first chat() call returns the multimodal response
    and subsequent calls return a text-only baseline response.
    """
    engine = MagicMock()
    engine.engine_id.return_value = "mock"

    mm_usage = None if mm_prompt_tokens is None else {"prompt_tokens": mm_prompt_tokens}
    mm_response = {
        "role": "assistant",
        "content": "ok",
        "tool_calls": None,
        "finish_reason": "stop",
        "usage": mm_usage,
    }
    text_response = {
        "role": "assistant",
        "content": "ok",
        "tool_calls": None,
        "finish_reason": "stop",
        "usage": {"prompt_tokens": text_prompt_tokens},
    }
    engine.chat.side_effect = [mm_response, text_response]
    return engine


def _check(result, name):
    for c in result.checks:
        if c.name == name:
            return c
    raise AssertionError(f"check {name!r} not found in {[c.name for c in result.checks]}")


# --- Vision ---------------------------------------------------------------


def test_vision_engine_passes_when_image_adds_tokens():
    engine = _scripted_engine(mm_prompt_tokens=500, text_prompt_tokens=10)
    result = VisionEnginePlumbingTest().run(engine, "test-model")
    assert result.passed, [(c.name, c.passed, c.detail) for c in result.checks]
    assert _check(result, "image_ingested_prompt_tokens").passed
    assert _check(result, "response_role").passed
    assert _check(result, "response_content_nonempty").passed
    assert _check(result, "response_finish_reason").passed


def test_vision_engine_fails_when_image_did_not_add_tokens():
    engine = _scripted_engine(mm_prompt_tokens=10, text_prompt_tokens=10)
    result = VisionEnginePlumbingTest().run(engine, "test-model")
    assert not result.passed
    assert not _check(result, "image_ingested_prompt_tokens").passed


def test_vision_engine_fails_when_usage_missing():
    engine = _scripted_engine(mm_prompt_tokens=None)
    result = VisionEnginePlumbingTest().run(engine, "test-model")
    assert not result.passed
    check = _check(result, "image_ingested_prompt_tokens")
    assert not check.passed
    assert "usage.prompt_tokens" in check.detail


def test_vision_engine_captures_exception():
    engine = MagicMock()
    engine.engine_id.return_value = "mock"
    engine.chat.side_effect = RuntimeError("unsupported image_url")
    result = VisionEnginePlumbingTest().run(engine, "test-model")
    assert not result.passed
    assert any("unsupported image_url" in c.detail for c in result.checks)


def test_vision_engine_payload_shape():
    engine = _scripted_engine(mm_prompt_tokens=500)
    VisionEnginePlumbingTest().run(engine, "test-model")
    first_call_args, first_call_kwargs = engine.chat.call_args_list[0]
    messages = first_call_args[0] if first_call_args else first_call_kwargs["messages"]
    parts = messages[0]["content"]
    assert parts[0]["type"] == "text"
    assert parts[1]["type"] == "image_url"
    assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")


# --- Speech ---------------------------------------------------------------


def test_speech_engine_passes_when_audio_adds_tokens():
    engine = _scripted_engine(mm_prompt_tokens=1200, text_prompt_tokens=12)
    result = SpeechEnginePlumbingTest().run(engine, "test-model")
    assert result.passed, [(c.name, c.passed, c.detail) for c in result.checks]
    assert _check(result, "audio_ingested_prompt_tokens").passed


def test_speech_engine_fails_when_audio_did_not_add_tokens():
    engine = _scripted_engine(mm_prompt_tokens=12, text_prompt_tokens=12)
    result = SpeechEnginePlumbingTest().run(engine, "test-model")
    assert not result.passed
    assert not _check(result, "audio_ingested_prompt_tokens").passed


def test_speech_engine_fails_when_usage_missing():
    engine = _scripted_engine(mm_prompt_tokens=None)
    result = SpeechEnginePlumbingTest().run(engine, "test-model")
    assert not result.passed
    assert not _check(result, "audio_ingested_prompt_tokens").passed


def test_speech_engine_payload_shape():
    engine = _scripted_engine(mm_prompt_tokens=1200)
    SpeechEnginePlumbingTest().run(engine, "test-model")
    first_call_args, first_call_kwargs = engine.chat.call_args_list[0]
    messages = first_call_args[0] if first_call_args else first_call_kwargs["messages"]
    parts = messages[0]["content"]
    assert parts[0]["type"] == "text"
    assert parts[1]["type"] == "audio_url"
    assert parts[1]["audio_url"]["url"].startswith("data:audio/wav;base64,")
