"""Bundled fixture assets for validation tests (images, audio, etc.)."""

from __future__ import annotations

from importlib import resources


def load_bytes(relpath: str) -> bytes:
    """Return the bytes of a resource file under tests/resources/.

    Example: ``load_bytes("vision/shapes.png")``.
    """
    parts = relpath.split("/")
    ref = resources.files(__name__).joinpath(*parts)
    return ref.read_bytes()
