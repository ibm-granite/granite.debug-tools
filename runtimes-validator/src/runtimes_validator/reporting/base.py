from __future__ import annotations

from abc import ABC, abstractmethod

from runtimes_validator.domain.models import EngineInfo, Report, TestResult


class AbstractReporter(ABC):
    """Interface that all reporters must implement."""

    @abstractmethod
    def report(self, report: Report) -> None:
        """Generate a report from the validation results."""

    def on_run_start(self, info: EngineInfo, model: str) -> None:
        """Called before test execution begins. Override to print a header."""

    def on_test_complete(self, result: TestResult) -> None:
        """Called after each individual test finishes. Override for streaming output."""
