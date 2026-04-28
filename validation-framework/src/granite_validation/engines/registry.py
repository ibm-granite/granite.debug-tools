from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from granite_validation.engines.base import AbstractEngine, EngineConfig

_ENGINES: dict[str, type[AbstractEngine]] = {}


def register_engine(engine_id: str):
    """Class decorator that registers an engine adapter."""

    def decorator(cls: type[AbstractEngine]) -> type[AbstractEngine]:
        if engine_id in _ENGINES:
            raise ValueError(f"Engine '{engine_id}' is already registered")
        _ENGINES[engine_id] = cls
        return cls

    return decorator


def get_engine_class(engine_id: str) -> type[AbstractEngine]:
    """Return the engine class for the given engine_id."""
    if engine_id not in _ENGINES:
        available = ", ".join(sorted(_ENGINES.keys()))
        raise KeyError(f"Unknown engine '{engine_id}'. Available: {available}")
    return _ENGINES[engine_id]


def create_engine(engine_id: str, config: EngineConfig) -> AbstractEngine:
    """Instantiate an engine adapter with the given config."""
    cls = get_engine_class(engine_id)
    return cls(config)  # type: ignore[call-arg]


def list_engines() -> list[str]:
    """Return all registered engine identifiers."""
    return sorted(_ENGINES.keys())
