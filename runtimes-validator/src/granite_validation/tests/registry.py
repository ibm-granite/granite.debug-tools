from __future__ import annotations

import importlib
import pkgutil
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from granite_validation.tests.base import AbstractValidationTest

_TESTS: dict[str, type[AbstractValidationTest]] = {}


def register_test(test_id: str):
    """Class decorator that registers a validation test."""

    def decorator(cls: type[AbstractValidationTest]) -> type[AbstractValidationTest]:
        if test_id in _TESTS:
            raise ValueError(f"Test '{test_id}' is already registered")
        _TESTS[test_id] = cls
        return cls

    return decorator


def discover_tests() -> None:
    """Auto-import all modules under tests/common/ and tests/engine_specific/ to trigger registration."""
    from granite_validation.tests import common, engine_specific

    for package in (common, engine_specific):
        for importer, modname, _ispkg in pkgutil.walk_packages(
            package.__path__, prefix=package.__name__ + "."
        ):
            importlib.import_module(modname)


def get_tests(engine_id: str | None = None) -> list[type[AbstractValidationTest]]:
    """Return test classes applicable to the given engine (or all if engine_id is None)."""
    results = []
    for cls in _TESTS.values():
        applicable = cls().applicable_engines()
        if applicable is None or engine_id is None or engine_id in applicable:
            results.append(cls)
    return results


def get_test_by_id(test_id: str) -> type[AbstractValidationTest]:
    """Return a specific test class by its ID."""
    if test_id not in _TESTS:
        available = ", ".join(sorted(_TESTS.keys()))
        raise KeyError(f"Unknown test '{test_id}'. Available: {available}")
    return _TESTS[test_id]


def list_tests() -> list[str]:
    """Return all registered test identifiers."""
    return sorted(_TESTS.keys())
