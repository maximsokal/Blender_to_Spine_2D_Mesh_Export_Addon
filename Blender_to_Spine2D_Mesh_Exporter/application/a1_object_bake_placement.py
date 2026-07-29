"""Resolve object-bake main-bone placement without Blender runtime dependencies."""

from __future__ import annotations

from ..domain.geometry import MeshSnapshot
from .a1_numeric_contracts import require_finite_number
from .a1_single_object import (
    A1MeshBounds,
    A1SingleObjectExportSettings,
    calculate_a1_main_position_pixels,
)


def _validate_cached_bounds(bounds: A1MeshBounds) -> None:
    """Validate an optional cached bounds value retained for API compatibility.

    Object-bake pivot placement no longer uses the geometry midpoint. The bounds argument
    remains accepted because callers may already cache and pass it, but malformed cached
    data must still fail closed instead of being silently ignored.
    """

    if not isinstance(bounds, A1MeshBounds):
        raise TypeError("bounds must be A1MeshBounds or None")

    minimum_x = require_finite_number(bounds.minimum_x, "bounds.minimum_x")
    maximum_x = require_finite_number(bounds.maximum_x, "bounds.maximum_x")
    minimum_y = require_finite_number(bounds.minimum_y, "bounds.minimum_y")
    maximum_y = require_finite_number(bounds.maximum_y, "bounds.maximum_y")
    center_x = require_finite_number(bounds.center_x, "bounds.center_x")
    center_y = require_finite_number(bounds.center_y, "bounds.center_y")

    if minimum_x > maximum_x:
        raise ValueError("bounds.minimum_x cannot exceed bounds.maximum_x")
    if minimum_y > maximum_y:
        raise ValueError("bounds.minimum_y cannot exceed bounds.maximum_y")

    expected_center_x = require_finite_number(
        (minimum_x + maximum_x) / 2.0,
        "bounds.expected_center_x",
    )
    expected_center_y = require_finite_number(
        (minimum_y + maximum_y) / 2.0,
        "bounds.expected_center_y",
    )
    if center_x != expected_center_x:
        raise ValueError(
            "bounds.center_x must be the midpoint of minimum_x and maximum_x"
        )
    if center_y != expected_center_y:
        raise ValueError(
            "bounds.center_y must be the midpoint of minimum_y and maximum_y"
        )


def calculate_a1_object_bake_main_position_pixels(
    snapshot: MeshSnapshot,
    settings: A1SingleObjectExportSettings,
    *,
    bounds: A1MeshBounds | None = None,
) -> tuple[float, float]:
    """Return the Spine main-bone position for the Blender Object Origin.

    Mesh coordinates in :class:`MeshSnapshot` are object-local. Their local ``(0, 0)`` is
    therefore the authored Blender Object Origin and must stay the deformation pivot in
    Spine. Attachment projection stores vertices relative to that origin; this function
    contributes only the optional Blender world translation converted to Spine pixels.

    When ``use_world_location_for_main_bone`` is disabled, connected composition owns the
    anchor-relative world translation and this function returns a neutral local origin.
    """

    if not isinstance(snapshot, MeshSnapshot):
        raise TypeError("snapshot must be MeshSnapshot")
    if not isinstance(settings, A1SingleObjectExportSettings):
        raise TypeError("settings must be A1SingleObjectExportSettings")
    if bounds is not None:
        _validate_cached_bounds(bounds)

    world_position_pixels = calculate_a1_main_position_pixels(snapshot, settings)
    if world_position_pixels is None:
        return (0.0, 0.0)

    return (
        require_finite_number(
            world_position_pixels[0],
            "object_origin_main_x",
        ),
        require_finite_number(
            world_position_pixels[1],
            "object_origin_main_y",
        ),
    )


__all__ = ["calculate_a1_object_bake_main_position_pixels"]
