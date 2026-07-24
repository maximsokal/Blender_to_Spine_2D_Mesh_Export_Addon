"""Resolve object-bake main-bone placement without Blender runtime dependencies."""

from __future__ import annotations

from ..domain.geometry import MeshSnapshot
from ..domain.spine.legacy_rig_scale import calculate_uniform_scale
from .a1_single_object import (
    A1MeshBounds,
    A1SingleObjectExportSettings,
    calculate_a1_main_position_pixels,
    calculate_a1_mesh_bounds,
)


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
    if not isinstance(resolved_bounds, A1MeshBounds):
        raise TypeError("bounds must be A1MeshBounds or None")

    world_position_pixels = calculate_a1_main_position_pixels(snapshot, settings)
    base_x, base_y = (
        (0.0, 0.0)
        if world_position_pixels is None
        else (float(world_position_pixels[0]), float(world_position_pixels[1]))
    )
    uniform_scale = calculate_uniform_scale(
        settings.export.texture_width,
        settings.export.texture_height,
        settings.rig_scale_mode,
    )
    return (
        base_x + float(resolved_bounds.center_x) * uniform_scale,
        base_y - float(resolved_bounds.center_y) * uniform_scale,
    )


__all__ = ["calculate_a1_object_bake_main_position_pixels"]
