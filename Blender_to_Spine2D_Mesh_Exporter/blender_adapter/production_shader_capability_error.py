"""Shared failures for live production shader-capability validation."""


class ProductionShaderCapabilityError(RuntimeError):
    """Raised when live Blender materials and immutable analysis disagree."""


__all__ = ["ProductionShaderCapabilityError"]
