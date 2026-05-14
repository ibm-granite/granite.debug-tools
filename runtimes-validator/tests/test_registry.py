from runtimes_validator.engines.registry import list_engines
from runtimes_validator.tests.registry import discover_tests, get_tests, list_tests

# Trigger engine registration
import runtimes_validator.engines.ollama  # noqa: F401
import runtimes_validator.engines.vllm  # noqa: F401
import runtimes_validator.engines.llamacpp  # noqa: F401

COMMON_IDS = {
    "basic_generation",
    "chat_completion",
    "concurrent_requests",
    "multi_turn",
    "tool_calling",
    "system_message_behavior",
    "long_input",
    "special_token_leakage",
    "no_tool_tags_without_tools",
}

OLLAMA_IDS = {
    "ollama_model_metadata",
    "ollama_streaming",
    "ollama_generate",
    "ollama_performance",
    "ollama_context_window",
}

LLAMACPP_IDS = {
    "llamacpp_tokenization",
    "llamacpp_streaming",
    "llamacpp_props",
    "llamacpp_context_shift",
}

VLLM_IDS = {
    "vllm_tool_flags",
    "vllm_prefix_caching",
    "vllm_quantization",
    "vllm_tensor_parallel",
}


def test_engine_registry_has_all_engines():
    engines = list_engines()
    assert "ollama" in engines
    assert "vllm" in engines
    assert "llamacpp" in engines


def test_test_discovery():
    discover_tests()
    all_ids = set(list_tests())
    for tid in COMMON_IDS | OLLAMA_IDS | LLAMACPP_IDS | VLLM_IDS:
        assert tid in all_ids, f"{tid} not found in registry"


def test_get_tests_returns_classes():
    discover_tests()
    test_classes = get_tests()
    assert len(test_classes) > 0
    for cls in test_classes:
        instance = cls()
        assert instance.test_id()
        assert instance.test_name()


def test_ollama_tests_included_for_ollama_engine():
    discover_tests()
    ollama_tests = {cls().test_id() for cls in get_tests(engine_id="ollama")}
    for tid in OLLAMA_IDS | COMMON_IDS:
        assert tid in ollama_tests, f"{tid} not in ollama-filtered tests"


def test_ollama_tests_excluded_for_vllm_engine():
    discover_tests()
    vllm_tests = {cls().test_id() for cls in get_tests(engine_id="vllm")}
    for tid in OLLAMA_IDS:
        assert tid not in vllm_tests, f"ollama test {tid} should not appear for vllm"
    for tid in COMMON_IDS:
        assert tid in vllm_tests, f"common test {tid} should appear for vllm"
    for tid in VLLM_IDS:
        assert tid in vllm_tests, f"vllm test {tid} should appear for vllm"


def test_llamacpp_tests_included_for_llamacpp_engine():
    discover_tests()
    llamacpp_tests = {cls().test_id() for cls in get_tests(engine_id="llamacpp")}
    for tid in LLAMACPP_IDS | COMMON_IDS:
        assert tid in llamacpp_tests, f"{tid} not in llamacpp-filtered tests"


def test_llamacpp_tests_excluded_for_ollama_engine():
    discover_tests()
    ollama_tests = {cls().test_id() for cls in get_tests(engine_id="ollama")}
    for tid in LLAMACPP_IDS:
        assert tid not in ollama_tests, f"llamacpp test {tid} should not appear for ollama"


def test_vllm_tests_excluded_for_ollama_engine():
    discover_tests()
    ollama_tests = {cls().test_id() for cls in get_tests(engine_id="ollama")}
    for tid in VLLM_IDS:
        assert tid not in ollama_tests, f"vllm test {tid} should not appear for ollama"


def test_vllm_tests_excluded_for_llamacpp_engine():
    discover_tests()
    llamacpp_tests = {cls().test_id() for cls in get_tests(engine_id="llamacpp")}
    for tid in VLLM_IDS:
        assert tid not in llamacpp_tests, f"vllm test {tid} should not appear for llamacpp"


def test_total_test_count():
    discover_tests()
    all_ids = list_tests()
    assert len(all_ids) >= 22, f"Expected >= 22 tests, got {len(all_ids)}: {all_ids}"
