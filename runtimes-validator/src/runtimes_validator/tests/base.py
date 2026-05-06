from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Iterator

from runtimes_validator.domain.models import TestResult
from runtimes_validator.engines.base import AbstractEngine


class AbstractValidationTest(ABC):
    """Interface that all validation tests must implement."""

    @abstractmethod
    def test_id(self) -> str:
        """Unique identifier for this test."""

    @abstractmethod
    def test_name(self) -> str:
        """Human-readable name for this test."""

    @abstractmethod
    def run(self, engine: AbstractEngine, model: str) -> TestResult:
        """Execute the test against the given engine and model."""

    def applicable_engines(self) -> list[str] | None:
        """Return None to run on all engines, or a list of engine_ids."""
        return None

    @contextmanager
    def _check_scope(
        self, engine: AbstractEngine, check_id: str
    ) -> Iterator[None]:
        """Tag inspection-log entries emitted inside the block with ``check_id``."""
        inspection = getattr(engine, "_inspection", None)
        setter = getattr(inspection, "set_current_check", None) if inspection else None
        if setter is not None:
            setter(check_id)
        try:
            yield
        finally:
            if setter is not None:
                setter(None)
