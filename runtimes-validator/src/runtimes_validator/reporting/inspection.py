from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Any


@dataclass
class _Scope:
    checks: list[Any]
    start_idx: int
    exchanges: list[dict[str, Any]] = field(default_factory=list)


class InspectionLogger:
    """Writes one JSONL line per request/response exchange for post-hoc inspection.

    When a scope is active (``begin_scope``), exchanges are buffered and flushed
    when the next scope begins, the current test changes, or the logger closes.
    At flush time, one entry is emitted per exchange per CheckResult name added
    to the scope's ``checks`` list, with ``test_id`` formatted as
    ``"{test_id}:{check_name}"``.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._current_test: str | None = None
        self._scope: _Scope | None = None
        self._fh: IO[str] | None = None
        self._closed = False

    def set_current_test(self, test_id: str | None) -> None:
        self._flush_scope()
        self._current_test = test_id

    def begin_scope(self, checks: list[Any]) -> None:
        self._flush_scope()
        self._scope = _Scope(checks=checks, start_idx=len(checks))

    def log_exchange(
        self,
        payload: dict[str, Any],
        response: dict[str, Any] | list[dict[str, Any]] | None,
        *,
        streaming: bool,
        path: str = "",
    ) -> None:
        if self._closed:
            return
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "streaming": streaming,
            "path": path,
            "payload": payload,
            "response": response,
        }
        if self._scope is not None:
            self._scope.exchanges.append(record)
            return
        self._write_entry({**record, "test_id": self._current_test})

    def close(self) -> None:
        if self._closed:
            return
        self._flush_scope()
        self._closed = True
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def _flush_scope(self) -> None:
        scope = self._scope
        if scope is None:
            return
        self._scope = None
        names = [getattr(c, "name", None) for c in scope.checks[scope.start_idx :]]
        for record in scope.exchanges:
            if names:
                for name in names:
                    test_id = f"{self._current_test}:{name}" if self._current_test else name
                    self._write_entry({**record, "test_id": test_id})
            else:
                self._write_entry({**record, "test_id": self._current_test})

    def _write_entry(self, entry: dict[str, Any]) -> None:
        ordered = {"test_id": entry.pop("test_id"), **entry}
        fh = self._ensure_open()
        fh.write(json.dumps(ordered, default=str) + "\n")
        fh.flush()

    def _ensure_open(self) -> IO[str]:
        if self._fh is None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = self._path.open("a", encoding="utf-8")
        return self._fh
