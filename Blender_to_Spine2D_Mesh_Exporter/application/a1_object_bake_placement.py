"""Resolve object-bake main-bone placement without Blender runtime dependencies."""

from __future__ import annotations

from math import isclose

from ..domain.geometry import MeshSnapshot
from ..domain.spine.legacy_rig_scale import calculate_uniform_scale
from .a1_numeric_contracts import require_finite_number
from .a1_single_object import (
    A1MeshBounds,
    A1SingleObjectExportSettings,
    calculate_a1_main_position_pixels,
    calculate_a1_mesh_bounds,
)


def _validated_bounds_center(bounds: A1MeshBounds) -> tuple[float, float]:
    """Validate a cached bounds object before it affects exported coordinates."""

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

    # Divide before adding so two large same-sign finite endpoints cannot overflow.
    expected_center_x = require_finite_number(
        minimum_x / 2.0 + maximum_x / 2.0,
        "bounds.expected_center_x",
    )
    expected_center_y = require_finite_number(
        minimum_y / 2.0 + maximum_y / 2.0,
        "bounds.expected_center_y",
    )
    if not isclose(center_x, expected_center_x, rel_tol=1.0e-12, abs_tol=1.0e-12):
        raise ValueError(
            "bounds.center_x must be the midpoint of minimum_x and maximum_x"
        )
    if not isclose(center_y, expected_center_y, rel_tol=1.0e-12, abs_tol=1.0e-12):
        raise ValueError(
            "bounds.center_y must be the midpoint of minimum_y and maximum_y"
        )
    return center_x, center_y


def calculate_a1_object_bake_main_position_pixels(
    snapshot: MeshSnapshot,
    settings: A1SingleObjectExportSettings,
    *,
    bounds: A1MeshBounds | None = None,
) -> tuple[float, float]:
    """Return the main-bone XY that preserves geometry relative to Object Origin.

    Object-bake attachment vertices are centered around the normalized source XY
    bounding-box midpoint. The inverse center translation therefore belongs on the
    object's main bone. Blender mesh Y is inverted during attachment projection, so
    the matching main-bone offset is ``(center_x, -center_y) * uniform_scale``.

    When ``use_world_location_for_main_bone`` is disabled, the result contains only
    that document-local geometry offset. Connected composition can then add the
    anchor-relative Blender Object translation without losing the authored origin.
    """

    if not isinstance(snapshot, MeshSnapshot):
        raise TypeError("snapshot must be MeshSnapshot")
    if not isinstance(settings, A1SingleObjectExportSettings):
        raise TypeError("settings must be A1SingleObjectExportSettings")

    resolved_bounds = calculate_a1_mesh_bounds(snapshot) if bounds is None else bounds
    center_x, center_y = _validated_bounds_center(resolved_bounds)

    world_position_pixels = calculate_a1_main_position_pixels(snapshot, settings)
    if world_position_pixels is None:
        base_x, base_y = 0.0, 0.0
    else:
        base_x = require_finite_number(
            world_position_pixels[0],
            "world_position_pixels[0]",
        )
        base_y = require_finite_number(
            world_position_pixels[1],
            "world_position_pixels[1]",
        )

    uniform_scale = calculate_uniform_scale(
        settings.export.texture_width,
        settings.export.texture_height,
        settings.rig_scale_mode,
    )
    main_x = require_finite_number(
        base_x + center_x * uniform_scale,
        "object_bake_main_x",
    )
    main_y = require_finite_number(
        base_y - center_y * uniform_scale,
        "object_bake_main_y",
    )
    return main_x, main_y


__all__ = ["calculate_a1_object_bake_main_position_pixels"]
