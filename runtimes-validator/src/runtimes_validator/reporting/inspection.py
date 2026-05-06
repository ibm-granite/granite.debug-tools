from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Any


class InspectionLogger:
    """Writes one JSONL line per request/response exchange for post-hoc inspection.

    Each entry is a single JSON object:
        {"ts", "test_id", "check_id", "streaming": bool, "path": str,
         "payload": {...}, "response": {...} | [...] | null}
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._current_test: str | None = None
        self._current_check: str | None = None
        self._fh: IO[str] | None = None
        self._closed = False

    def set_current_test(self, test_id: str | None) -> None:
        self._current_test = test_id

    def set_current_check(self, check_id: str | None) -> None:
        self._current_check = check_id

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
        fh = self._ensure_open()
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "test_id": self._current_test,
            "check_id": self._current_check,
            "streaming": streaming,
            "path": path,
            "payload": payload,
            "response": response,
        }
        fh.write(json.dumps(entry, default=str) + "\n")
        fh.flush()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def _ensure_open(self) -> IO[str]:
        if self._fh is None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = self._path.open("a", encoding="utf-8")
        return self._fh
