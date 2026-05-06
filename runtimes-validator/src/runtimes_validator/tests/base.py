from __future__ import annotations

from abc import ABC, abstractmethod

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
