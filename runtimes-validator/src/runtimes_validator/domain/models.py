from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


class EngineTimeoutError(Exception):
    """Raised when an engine request times out."""


@dataclass
class CheckResult:
    """A single assertion within a test."""

    name: str
    passed: bool
    expected: Any = None
    actual: Any = None
    detail: str = ""


@dataclass
class TestResult:
    """Result of running one validation test."""

    test_id: str
    test_name: str
    engine_id: str
    model: str
    checks: list[CheckResult]
    elapsed_seconds: float
    error: str | None = None
    skipped: bool = False
    skip_reason: str | None = None

    @property
    def passed(self) -> bool:
        if self.skipped:
            return False
        return self.error is None and all(c.passed for c in self.checks)

    @property
    def status(self) -> str:
        if self.skipped:
            return "skip"
        return "pass" if self.passed else "fail"


@dataclass
class EngineInfo:
    """Engine metadata included in reports."""

    engine_id: str
    version: str
    mode: Literal["managed", "external", "unknown"]
    base_url: str
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Report:
    """Complete validation run report."""

    engine_info: EngineInfo
    model: str
    results: list[TestResult]
    timestamp: str
    total_elapsed_seconds: float
    lifecycle_error: str | None = None
    cleanup_error: str | None = None
    abort_reason: str | None = None

    @property
    def all_passed(self) -> bool:
        return (
            self.lifecycle_error is None
            and self.cleanup_error is None
            and self.abort_reason is None
            and all(r.passed or r.skipped for r in self.results)
        )
