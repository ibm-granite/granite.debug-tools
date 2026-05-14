from typing import Any

from runtimes_validator.domain.models import EngineInfo
from runtimes_validator.engines.base import AbstractEngine, EngineConfig
from runtimes_validator.engines.llamacpp import LlamaCppEngine
from runtimes_validator.engines.ollama import OllamaEngine
from runtimes_validator.engines.vllm import VllmEngine
from runtimes_validator.tests.registry import (
    discover_tests,
    get_test_by_id,
    get_tests,
)


def test_existing_tests_default_to_text_modality():
    discover_tests()
    basic = get_test_by_id("basic_generation")()
    assert basic.modalities() == ["text"]


def test_multimodal_tests_declare_their_modalities():
    discover_tests()
    assert get_test_by_id("vision_basic")().modalities() == ["vision"]
    assert get_test_by_id("vision_ocr")().modalities() == ["vision"]
    assert get_test_by_id("speech_basic")().modalities() == ["speech"]


def test_get_tests_vision_filter_excludes_text_tests():
    discover_tests()
    ids = {cls().test_id() for cls in get_tests(modalities={"vision"})}
    assert "vision_basic" in ids
    assert "vision_ocr" in ids
    assert "speech_basic" not in ids
    assert "basic_generation" not in ids


def test_get_tests_text_filter_excludes_multimodal_tests():
    discover_tests()
    ids = {cls().test_id() for cls in get_tests(modalities={"text"})}
    assert "basic_generation" in ids
    assert "vision_basic" not in ids
    assert "speech_basic" not in ids


def test_get_tests_multiple_modalities_union():
    discover_tests()
    ids = {cls().test_id() for cls in get_tests(modalities={"vision", "speech"})}
    assert "vision_basic" in ids
    assert "speech_basic" in ids
    assert "basic_generation" not in ids


def test_get_tests_no_modality_filter_returns_everything():
    discover_tests()
    ids = {cls().test_id() for cls in get_tests()}
    assert "basic_generation" in ids
    assert "vision_basic" in ids
    assert "speech_basic" in ids


# --- Engine-side supported_modalities() declarations ---


def test_ollama_engine_supported_modalities():
    engine = OllamaEngine(EngineConfig())
    assert engine.supported_modalities() == {"text", "vision"}


def test_llamacpp_engine_supported_modalities():
    engine = LlamaCppEngine(EngineConfig())
    assert engine.supported_modalities() == {"text", "vision", "speech"}


def test_vllm_engine_supported_modalities():
    engine = VllmEngine(EngineConfig())
    assert engine.supported_modalities() == {"text", "vision", "speech"}


def test_abstract_engine_default_is_text_only():
    class MinimalEngine(AbstractEngine):
        def engine_id(self) -> str:
            return "minimal"

        def start(self, model: str) -> None:
            pass

        def stop(self) -> None:
            pass

        def chat(
            self,
            messages: list[dict[str, Any]],
            *,
            tools: list[dict[str, Any]] | None = None,
            tool_choice: str | None = None,
            temperature: float = 0.0,
            max_tokens: int = 512,
        ) -> dict[str, Any]:
            return {}

        def health_check(self) -> bool:
            return True

        def get_info(self) -> EngineInfo:
            return EngineInfo(
                engine_id="minimal",
                version="0",
                mode="external",
                base_url="http://minimal",
            )

    assert MinimalEngine().supported_modalities() == {"text"}
