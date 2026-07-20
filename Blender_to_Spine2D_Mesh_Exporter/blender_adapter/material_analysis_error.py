"""Shared errors for Blender material analysis."""

class MaterialAnalysisError(RuntimeError):
    """Raised when Blender material data cannot be inspected safely."""


__all__ = ["MaterialAnalysisError"]
