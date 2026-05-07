"""Shape tests for multimodal test payloads, run against a mock engine."""

from unittest.mock import MagicMock

from runtimes_validator.tests.multimodal.test_speech_basic import SpeechBasicTest
from runtimes_validator.tests.multimodal.test_vision_basic import VisionBasicTest
from runtimes_validator.tests.multimodal.test_vision_ocr import VisionOcrTest


def _mock_engine(reply: str):
    engine = MagicMock()
    engine.engine_id.return_value = "mock"
    engine.chat.return_value = {
        "role": "assistant",
        "content": reply,
        "tool_calls": None,
        "finish_reason": "stop",
        "usage": None,
    }
    return engine


def _get_message(engine):
    args, kwargs = engine.chat.call_args
    messages = args[0] if args else kwargs["messages"]
    assert len(messages) == 1
    return messages[0]


def test_vision_basic_payload_shape():
    engine = _mock_engine("A red circle.")
    VisionBasicTest().run(engine, "test-model")
    msg = _get_message(engine)
    assert msg["role"] == "user"
    parts = msg["content"]
    assert len(parts) == 2
    assert parts[0]["type"] == "text"
    assert parts[1]["type"] == "image_url"
    assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert len(parts[1]["image_url"]["url"]) > 100  # non-empty base64


def test_vision_basic_passes_when_response_mentions_circle():
    engine = _mock_engine("The image shows a red circle.")
    result = VisionBasicTest().run(engine, "test-model")
    assert result.passed


def test_vision_basic_fails_when_response_missing_circle():
    engine = _mock_engine("It is a square.")
    result = VisionBasicTest().run(engine, "test-model")
    assert not result.passed


def test_vision_ocr_payload_shape():
    engine = _mock_engine("GRANITE")
    VisionOcrTest().run(engine, "test-model")
    msg = _get_message(engine)
    parts = msg["content"]
    assert len(parts) == 2
    assert parts[0]["type"] == "text"
    assert parts[1]["type"] == "image_url"


def test_vision_ocr_passes_on_granite_word():
    engine = _mock_engine("GRANITE")
    result = VisionOcrTest().run(engine, "test-model")
    assert result.passed


def test_speech_basic_payload_shape():
    engine = _mock_engine("hello granite")
    SpeechBasicTest().run(engine, "test-model")
    msg = _get_message(engine)
    parts = msg["content"]
    assert len(parts) == 2
    assert parts[0]["type"] == "text"
    assert parts[1]["type"] == "audio_url"
    url = parts[1]["audio_url"]["url"]
    assert url.startswith("data:audio/wav;base64,")
    assert len(url) > 100  # base64 payload non-empty


def test_speech_basic_passes_when_both_words_present():
    engine = _mock_engine("Hello, Granite.")
    result = SpeechBasicTest().run(engine, "test-model")
    assert result.passed


def test_speech_basic_fails_when_word_missing():
    engine = _mock_engine("Hello world.")
    result = SpeechBasicTest().run(engine, "test-model")
    assert not result.passed


def test_engine_error_is_captured_as_failed_check():
    engine = MagicMock()
    engine.engine_id.return_value = "mock"
    engine.chat.side_effect = RuntimeError("unsupported modality")
    result = VisionBasicTest().run(engine, "test-model")
    assert not result.passed
    assert any("unsupported modality" in c.detail for c in result.checks)
