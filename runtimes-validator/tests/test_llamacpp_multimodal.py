"""Payload-rewrite tests for LlamaCppEngine multimodal support.

llama-server accepts ``image_url`` as-is but expects audio as ``input_audio``
(``{data, format}``), not the ``audio_url`` shape our multimodal tests emit.
These tests pin the rewrite behavior.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from runtimes_validator.engines.base import EngineConfig
from runtimes_validator.engines.llamacpp import LlamaCppEngine


def _fake_chat_response() -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "choices": [
            {
                "message": {"role": "assistant", "content": "ok", "tool_calls": None},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    return resp


def _engine() -> LlamaCppEngine:
    return LlamaCppEngine(EngineConfig(mode="external", model_id="m"))


def _sent_messages(mock_post: MagicMock) -> list[dict]:
    kwargs = mock_post.call_args.kwargs
    return kwargs["json"]["messages"]


# --- audio rewrite -------------------------------------------------------


@patch("runtimes_validator.engines.openai_compat.requests.post")
def test_audio_url_wav_is_rewritten_to_input_audio(mock_post: MagicMock) -> None:
    mock_post.return_value = _fake_chat_response()
    engine = _engine()
    engine.chat(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "transcribe"},
                    {
                        "type": "audio_url",
                        "audio_url": {"url": "data:audio/wav;base64,QUJD"},
                    },
                ],
            }
        ]
    )

    parts = _sent_messages(mock_post)[0]["content"]
    assert parts[0] == {"type": "text", "text": "transcribe"}
    assert parts[1] == {
        "type": "input_audio",
        "input_audio": {"data": "QUJD", "format": "wav"},
    }


@patch("runtimes_validator.engines.openai_compat.requests.post")
def test_audio_url_mp3_is_rewritten_with_mp3_format(mock_post: MagicMock) -> None:
    mock_post.return_value = _fake_chat_response()
    _engine().chat(
        [
            {
                "role": "user",
                "content": [
                    {"type": "audio_url", "audio_url": {"url": "data:audio/mp3;base64,QUJD"}}
                ],
            }
        ]
    )

    parts = _sent_messages(mock_post)[0]["content"]
    assert parts[0]["type"] == "input_audio"
    assert parts[0]["input_audio"]["format"] == "mp3"


@patch("runtimes_validator.engines.openai_compat.requests.post")
def test_audio_url_audio_mpeg_normalizes_to_mp3(mock_post: MagicMock) -> None:
    mock_post.return_value = _fake_chat_response()
    _engine().chat(
        [
            {
                "role": "user",
                "content": [
                    {"type": "audio_url", "audio_url": {"url": "data:audio/mpeg;base64,QUJD"}}
                ],
            }
        ]
    )
    parts = _sent_messages(mock_post)[0]["content"]
    assert parts[0]["input_audio"]["format"] == "mp3"


# --- image / text / string pass-through ----------------------------------


@patch("runtimes_validator.engines.openai_compat.requests.post")
def test_image_url_is_forwarded_unchanged(mock_post: MagicMock) -> None:
    mock_post.return_value = _fake_chat_response()
    original = {
        "role": "user",
        "content": [
            {"type": "text", "text": "describe"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"}},
        ],
    }
    _engine().chat([original])

    assert _sent_messages(mock_post) == [original]


@patch("runtimes_validator.engines.openai_compat.requests.post")
def test_string_content_is_passed_through(mock_post: MagicMock) -> None:
    mock_post.return_value = _fake_chat_response()
    _engine().chat([{"role": "user", "content": "hello"}])
    assert _sent_messages(mock_post) == [{"role": "user", "content": "hello"}]


@patch("runtimes_validator.engines.openai_compat.requests.post")
def test_mixed_parts_preserve_order_and_only_rewrite_audio(mock_post: MagicMock) -> None:
    mock_post.return_value = _fake_chat_response()
    _engine().chat(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "a"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,SU1H"}},
                    {"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64,QVVE"}},
                    {"type": "text", "text": "b"},
                ],
            }
        ]
    )

    parts = _sent_messages(mock_post)[0]["content"]
    assert [p["type"] for p in parts] == ["text", "image_url", "input_audio", "text"]
    assert parts[0] == {"type": "text", "text": "a"}
    assert parts[1] == {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,SU1H"},
    }
    assert parts[2] == {
        "type": "input_audio",
        "input_audio": {"data": "QVVE", "format": "wav"},
    }
    assert parts[3] == {"type": "text", "text": "b"}


# --- error surface -------------------------------------------------------


@patch("runtimes_validator.engines.openai_compat.requests.post")
def test_non_data_audio_url_raises_before_http(mock_post: MagicMock) -> None:
    engine = _engine()
    with pytest.raises(ValueError, match="base64 audio data URLs"):
        engine.chat(
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "audio_url",
                            "audio_url": {"url": "https://example.com/hello.wav"},
                        }
                    ],
                }
            ]
        )
    mock_post.assert_not_called()


@patch("runtimes_validator.engines.openai_compat.requests.post")
def test_non_base64_audio_data_url_raises(mock_post: MagicMock) -> None:
    engine = _engine()
    with pytest.raises(ValueError, match="base64-encoded"):
        engine.chat(
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "audio_url",
                            "audio_url": {"url": "data:audio/wav,rawbytes"},
                        }
                    ],
                }
            ]
        )
    mock_post.assert_not_called()


@patch("runtimes_validator.engines.openai_compat.requests.post")
def test_unsupported_audio_format_raises(mock_post: MagicMock) -> None:
    engine = _engine()
    with pytest.raises(ValueError, match="wav/mp3"):
        engine.chat(
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "audio_url",
                            "audio_url": {"url": "data:audio/flac;base64,QUJD"},
                        }
                    ],
                }
            ]
        )
    mock_post.assert_not_called()
