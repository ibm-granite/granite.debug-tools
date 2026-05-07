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
