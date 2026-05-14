from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Iterator

from runtimes_validator.domain.models import CheckResult, TestResult
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
        self,
        engine: AbstractEngine,
        checks: list[CheckResult],
        check_id: str,
    ) -> Iterator[None]:
        """Buffer inspection-log entries emitted inside the block.

        The logger flushes buffered exchanges lazily (when the next scope begins,
        the test changes, or the logger closes), tagging each entry with
        ``"{test_id}:{check_name}"`` for every CheckResult appended to ``checks``
        during the scope. ``check_id`` is retained for readability at call sites
        but is not written to the log.
        """
        del check_id
        inspection = getattr(engine, "_inspection", None)
        begin = getattr(inspection, "begin_scope", None) if inspection else None
        if begin is not None:
            begin(checks)
        yield

    def modalities(self) -> list[str]:
        """Input modalities this test exercises (e.g. ``text``, ``vision``, ``speech``)."""
        return ["text"]
