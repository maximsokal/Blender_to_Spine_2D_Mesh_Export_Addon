"""Pure availability and validation contracts for multi-object shared pivots.

Shared-pivot export is intentionally narrower than generic multi-object export. It is a
Normal / UV Segments feature for signed world-axis projection only, and it only has
meaning when at least two Mesh objects participate in one export transaction.

Keeping this rule Blender-independent gives the UI, request builder, preparation
pipeline, and tests one authoritative capability contract instead of duplicating subtle
visibility/activation conditions.
"""

from __future__ import annotations

from math import isfinite
from typing import Tuple

from ..domain.baking import A1TextureExportMode
from ..domain.projection import A1ProjectionDirection


A1SharedPivotWorld = Tuple[float, float, float]


def supports_a1_shared_pivot(
    texture_export_mode: A1TextureExportMode,
    projection_direction: A1ProjectionDirection,
    object_count: int,
) -> bool:
    """Return whether one export request may use a shared selection pivot."""

    if not isinstance(texture_export_mode, A1TextureExportMode):
        raise TypeError("texture_export_mode must be A1TextureExportMode")
    if not isinstance(projection_direction, A1ProjectionDirection):
        raise TypeError("projection_direction must be A1ProjectionDirection")
    if isinstance(object_count, bool) or not isinstance(object_count, int):
        raise TypeError("object_count must be int")
    if object_count < 0:
        raise ValueError("object_count cannot be negative")

    return bool(
        object_count > 1
        and texture_export_mode is A1TextureExportMode.NORMAL_UV_SEGMENTS
        and projection_direction.axis_aligned
    )


def validate_a1_shared_pivot_world(value: object) -> A1SharedPivotWorld:
    """Return one canonical finite world-space pivot tuple or fail closed."""

    if not isinstance(value, tuple) or len(value) != 3:
        raise TypeError("shared_pivot_world must be a three-value tuple")

    resolved: list[float] = []
    for index, component in enumerate(value):
        if isinstance(component, bool) or not isinstance(component, (int, float)):
            raise TypeError(
                f"shared_pivot_world[{index}] must be a finite number"
            )
        numeric = float(component)
        if not isfinite(numeric):
            raise ValueError(
                f"shared_pivot_world[{index}] must be finite"
            )
        resolved.append(0.0 if numeric == 0.0 else numeric)

    return resolved[0], resolved[1], resolved[2]


__all__ = [
    "A1SharedPivotWorld",
    "supports_a1_shared_pivot",
    "validate_a1_shared_pivot_world",
]
