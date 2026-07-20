"""Shared error contract for Blender scene-bake analysis and parity."""


class SceneBakeAnalysisError(RuntimeError):
    """Raised when required Blender scene state cannot be inspected safely."""


__all__ = ["SceneBakeAnalysisError"]
