"""Dispatch post-render finalization without conflating flat and depth topology."""

from __future__ import annotations

from ..domain.baking import A1TextureExportMode, CameraProjectionLayout
from .a1_depth_projection_finalization import (
    finalize_prepared_depth_camera_projection,
)
from .a1_object_preparation import PreparedA1Object
from .a1_projection_finalization import finalize_prepared_camera_projection


def finalize_prepared_rendered_projection(
    prepared: PreparedA1Object,
    layout: CameraProjectionLayout | None,
) -> PreparedA1Object:
    """Finalize the selected rendered-camera representation exactly once."""

    if not isinstance(prepared, PreparedA1Object):
        raise TypeError("prepared must be PreparedA1Object")
    mode = prepared.settings.bake_execution.texture_export_mode
    if mode is A1TextureExportMode.DEPTH_CAMERA_PROJECTION:
        return finalize_prepared_depth_camera_projection(prepared, layout)
    return finalize_prepared_camera_projection(prepared, layout)


__all__ = ["finalize_prepared_rendered_projection"]
