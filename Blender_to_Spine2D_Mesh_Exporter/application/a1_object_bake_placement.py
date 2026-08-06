"""Resolve object-bake main-bone placement without Blender runtime dependencies."""

from __future__ import annotations

from ..domain.geometry import MeshSnapshot
from ..domain.spine import A1RigSetupPoseMode, calculate_uniform_scale
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


def _preprojected_origin_pixels(
    snapshot: MeshSnapshot,
    settings: A1SingleObjectExportSettings,
) -> tuple[float, float]:
    """Return projected Object Origin even when connected placement is localized.

    ``PREPROJECTED_SCREEN`` deliberately places ``main`` at camera-space zero and stores
    the projected Blender Object Origin on the internal base layer. That base position is
    required for every object, including connected exports that normally disable world
    translation on the ordinary Object Root main bone.
    """

    if len(snapshot.world_matrix) != 16:
        raise ValueError("snapshot.world_matrix must contain 16 values")
    uniform_scale = calculate_uniform_scale(
        settings.export.texture_width,
        settings.export.texture_height,
        settings.rig_scale_mode,
    )
    return (
        require_finite_number(
            float(snapshot.world_matrix[3]) * uniform_scale,
            "preprojected_object_origin_x",
        ),
        require_finite_number(
            float(snapshot.world_matrix[7]) * uniform_scale,
            "preprojected_object_origin_y",
        ),
    )


def calculate_a1_object_bake_main_position_pixels(
    snapshot: MeshSnapshot,
    settings: A1SingleObjectExportSettings,
    *,
    bounds: A1MeshBounds | None = None,
) -> tuple[float, float]:
    """Return the Spine placement point owned by the selected object-bake rig.

    Ordinary model-space rigs keep Blender Object Origin on ``main``. When
    ``use_world_location_for_main_bone`` is disabled, connected composition owns the
    anchor-relative translation and the ordinary Object Root route returns ``(0, 0)``.

    Camera Root uses ``PREPROJECTED_SCREEN``: ``main`` is camera-space zero and the
    projected Blender Object Origin is stored below it on the rigid object base. That
    projected position must therefore be returned even for connected composition.
    """

    if not isinstance(snapshot, MeshSnapshot):
        raise TypeError("snapshot must be MeshSnapshot")
    if not isinstance(settings, A1SingleObjectExportSettings):
        raise TypeError("settings must be A1SingleObjectExportSettings")
    if bounds is not None:
        _validate_cached_bounds(bounds)

    if settings.rig_setup_pose_mode is A1RigSetupPoseMode.PREPROJECTED_SCREEN:
        return _preprojected_origin_pixels(snapshot, settings)

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
