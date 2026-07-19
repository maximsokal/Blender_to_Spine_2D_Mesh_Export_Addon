"""Shared shader-graph analysis error contract."""

from __future__ import annotations


class MaterialGraphAnalysisError(RuntimeError):
    """Raised when a Blender node tree cannot be inspected deterministically."""


__all__ = ["MaterialGraphAnalysisError"]
