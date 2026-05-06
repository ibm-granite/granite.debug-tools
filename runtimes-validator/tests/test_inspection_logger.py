from __future__ import annotations

import json
from pathlib import Path

from runtimes_validator.reporting.inspection import InspectionLogger


def _read_entries(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_log_exchange_writes_single_line_with_payload_and_response(tmp_path: Path) -> None:
    log_path = tmp_path / "inspection.jsonl"
    logger = InspectionLogger(log_path)

    logger.log_exchange({"messages": []}, {"choices": []}, streaming=False)
    logger.close()

    entries = _read_entries(log_path)
    assert len(entries) == 1
    entry = entries[0]
    assert set(entry.keys()) == {"ts", "test_id", "streaming", "path", "payload", "response"}
    assert entry["streaming"] is False
    assert entry["payload"] == {"messages": []}
    assert entry["response"] == {"choices": []}


def test_null_response_is_recorded(tmp_path: Path) -> None:
    log_path = tmp_path / "inspection.jsonl"
    logger = InspectionLogger(log_path)

    logger.log_exchange({"a": 1}, None, streaming=False, path="/v1/chat/completions")
    logger.close()

    entries = _read_entries(log_path)
    assert len(entries) == 1
    assert entries[0]["payload"] == {"a": 1}
    assert entries[0]["response"] is None


def test_streaming_response_logs_list_of_chunks(tmp_path: Path) -> None:
    log_path = tmp_path / "inspection.jsonl"
    logger = InspectionLogger(log_path)

    logger.log_exchange(
        {"stream": True},
        [{"chunk": 1}, {"chunk": 2}, {"chunk": 3}],
        streaming=True,
    )
    logger.close()

    entries = _read_entries(log_path)
    assert len(entries) == 1
    assert entries[0]["streaming"] is True
    assert entries[0]["response"] == [{"chunk": 1}, {"chunk": 2}, {"chunk": 3}]


def test_set_current_test_tags_subsequent_entries(tmp_path: Path) -> None:
    log_path = tmp_path / "inspection.jsonl"
    logger = InspectionLogger(log_path)

    logger.log_exchange({"a": 1}, {"b": 1}, streaming=False)

    logger.set_current_test("basic_generation")
    logger.log_exchange({"a": 2}, {"b": 2}, streaming=False)

    logger.set_current_test(None)
    logger.log_exchange({"a": 3}, {"b": 3}, streaming=False)

    logger.close()

    entries = _read_entries(log_path)
    assert entries[0]["test_id"] is None
    assert entries[1]["test_id"] == "basic_generation"
    assert entries[2]["test_id"] is None


def test_close_is_idempotent(tmp_path: Path) -> None:
    log_path = tmp_path / "inspection.jsonl"
    logger = InspectionLogger(log_path)

    logger.log_exchange({"a": 1}, {"b": 1}, streaming=False)
    logger.close()
    logger.close()  # must not raise

    # Writes after close are silently dropped.
    logger.log_exchange({"a": 2}, {"b": 2}, streaming=False)

    entries = _read_entries(log_path)
    assert len(entries) == 1


def test_parent_directory_created(tmp_path: Path) -> None:
    log_path = tmp_path / "nested" / "dir" / "inspection.jsonl"
    logger = InspectionLogger(log_path)

    logger.log_exchange({"a": 1}, {"b": 1}, streaming=False)
    logger.close()

    assert log_path.exists()


def test_entries_include_endpoint_path(tmp_path: Path) -> None:
    log_path = tmp_path / "inspection.jsonl"
    logger = InspectionLogger(log_path)

    logger.log_exchange({"a": 1}, {"b": 2}, streaming=False, path="/v1/chat/completions")
    logger.log_exchange({"c": 3}, [{"d": 4}], streaming=True, path="/api/generate")
    logger.close()

    entries = _read_entries(log_path)
    assert [e["path"] for e in entries] == [
        "/v1/chat/completions",
        "/api/generate",
    ]


def test_path_defaults_to_empty_string_when_omitted(tmp_path: Path) -> None:
    log_path = tmp_path / "inspection.jsonl"
    logger = InspectionLogger(log_path)

    logger.log_exchange({"a": 1}, {"b": 2}, streaming=False)
    logger.close()

    entries = _read_entries(log_path)
    assert entries[0]["path"] == ""


def test_append_mode_across_instances(tmp_path: Path) -> None:
    log_path = tmp_path / "inspection.jsonl"

    logger1 = InspectionLogger(log_path)
    logger1.log_exchange({"run": 1}, {"ok": 1}, streaming=False)
    logger1.close()

    logger2 = InspectionLogger(log_path)
    logger2.log_exchange({"run": 2}, {"ok": 2}, streaming=False)
    logger2.close()

    entries = _read_entries(log_path)
    assert [e["payload"] for e in entries] == [{"run": 1}, {"run": 2}]
    assert [e["response"] for e in entries] == [{"ok": 1}, {"ok": 2}]
