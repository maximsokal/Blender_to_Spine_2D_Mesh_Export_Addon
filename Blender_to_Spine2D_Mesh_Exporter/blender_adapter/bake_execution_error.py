"""Shared error contract for semantic object-bake execution."""

from __future__ import annotations


class BakeExecutionError(RuntimeError):
    """Raised when a planned Blender bake fails before atomic output commit."""


__all__ = ["BakeExecutionError"]
