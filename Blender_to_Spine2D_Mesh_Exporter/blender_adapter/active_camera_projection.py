"""Resolve Blender 5.2 active-camera matrices for Normal / UV Segments.

This adapter performs Blender-specific evaluation only. Projection of immutable points and
Mesh snapshots remains in Blender-independent domain modules.
"""

from __future__ import annotations

from math import isfinite
from typing import Any

from ..domain.camera_projection import (
    A1CameraProjectionFrame,
    A1CameraProjectionKind,
)
from .scene_context_contract import (
    BlenderSceneContextError,
    require_depsgraph_scene_consistency,
)


class A1ActiveCameraProjectionError(ValueError):
    """Raised when the current Scene cannot provide a supported active camera frame."""


def _matrix_tuple(matrix: Any, field_name: str) -> tuple[float, ...]:
    try:
        values = tuple(
            float(matrix[row][column])
            for row in range(4)
            for column in range(4)
        )
    except Exception as exc:
        raise A1ActiveCameraProjectionError(
            f"Unable to read {field_name} as a 4x4 matrix"
        ) from exc
    if len(values) != 16 or not all(isfinite(value) for value in values):
        raise A1ActiveCameraProjectionError(
            f"{field_name} must contain sixteen finite values"
        )
    return values


def _resolved_scene_and_depsgraph(
    scene: Any | None,
    depsgraph: Any | None,
) -> tuple[Any, Any]:
    try:
        import bpy
    except Exception as exc:
        raise A1ActiveCameraProjectionError(
            "Blender bpy module is unavailable for active-camera projection"
        ) from exc

    resolved_scene = scene or getattr(bpy.context, "scene", None)
    if resolved_scene is None:
        raise A1ActiveCameraProjectionError(
            "Active Camera projection requires a Blender Scene"
        )
    resolved_depsgraph = depsgraph
    if resolved_depsgraph is None:
        try:
            resolved_depsgraph = bpy.context.evaluated_depsgraph_get()
        except Exception as exc:
            raise A1ActiveCameraProjectionError(
                "Unable to acquire Blender evaluated dependency graph"
            ) from exc
    if resolved_depsgraph is None:
        raise A1ActiveCameraProjectionError(
            "Blender returned no evaluated dependency graph"
        )
    try:
        require_depsgraph_scene_consistency(
            resolved_depsgraph,
            resolved_scene,
        )
    except BlenderSceneContextError as exc:
        raise A1ActiveCameraProjectionError(
            f"Active camera Scene and dependency graph disagree: {exc}"
        ) from exc
    return resolved_scene, resolved_depsgraph


def _rotation_only_view_matrix(evaluated_camera: Any) -> tuple[float, ...]:
    """Build camera world-to-local transform while ignoring object scale."""

    matrix_world = getattr(evaluated_camera, "matrix_world", None)
    if matrix_world is None:
        raise A1ActiveCameraProjectionError(
            "Evaluated active camera has no matrix_world"
        )
    try:
        location, rotation, _scale = matrix_world.decompose()
        rotation.normalize()
        camera_world = rotation.to_matrix().to_4x4()
        camera_world.translation = location
        view_matrix = camera_world.inverted()
    except Exception as exc:
        raise A1ActiveCameraProjectionError(
            "Unable to build an invertible active-camera view matrix"
        ) from exc
    return _matrix_tuple(view_matrix, "active camera view matrix")


def resolve_a1_active_camera_projection_frame(
    scene: Any | None,
    *,
    texture_width: int,
    texture_height: int,
    depsgraph: Any | None = None,
) -> A1CameraProjectionFrame:
    """Resolve one evaluated Perspective or Orthographic active camera frame.

    ``Object.calc_matrix_camera`` receives the export texture dimensions directly, so the
    frame aspect ratio follows the texture canvas rather than the current render window.
    Camera object scale is intentionally ignored, matching Blender camera-view behavior.
    """

    if isinstance(texture_width, bool) or not isinstance(texture_width, int):
        raise TypeError("texture_width must be int")
    if isinstance(texture_height, bool) or not isinstance(texture_height, int):
        raise TypeError("texture_height must be int")
    if texture_width <= 0 or texture_height <= 0:
        raise ValueError("texture dimensions must be positive")

    resolved_scene, resolved_depsgraph = _resolved_scene_and_depsgraph(
        scene,
        depsgraph,
    )
    camera = getattr(resolved_scene, "camera", None)
    if camera is None:
        raise A1ActiveCameraProjectionError(
            "Active Camera projection requires Scene.camera"
        )
    if str(getattr(camera, "type", "") or "") != "CAMERA":
        raise A1ActiveCameraProjectionError(
            "Scene.camera must reference a Blender CAMERA object"
        )
    try:
        scene_objects = tuple(resolved_scene.objects)
    except Exception as exc:
        raise A1ActiveCameraProjectionError(
            "Unable to inspect Scene objects for the active camera"
        ) from exc
    if camera not in scene_objects:
        raise A1ActiveCameraProjectionError(
            "Scene.camera is not linked to the projection Scene"
        )

    evaluated_get = getattr(camera, "evaluated_get", None)
    if not callable(evaluated_get):
        raise A1ActiveCameraProjectionError(
            "Active camera does not provide evaluated_get()"
        )
    try:
        evaluated_camera = evaluated_get(resolved_depsgraph)
    except Exception as exc:
        raise A1ActiveCameraProjectionError(
            "Unable to evaluate the active camera"
        ) from exc
    if evaluated_camera is None:
        raise A1ActiveCameraProjectionError(
            "Active camera evaluation returned None"
        )

    camera_data = getattr(evaluated_camera, "data", None)
    if camera_data is None:
        raise A1ActiveCameraProjectionError(
            "Evaluated active camera has no Camera data"
        )
    blender_type = str(getattr(camera_data, "type", "") or "")
    kind_by_blender_type = {
        "PERSP": A1CameraProjectionKind.PERSPECTIVE,
        "ORTHO": A1CameraProjectionKind.ORTHOGRAPHIC,
    }
    kind = kind_by_blender_type.get(blender_type)
    if kind is None:
        raise A1ActiveCameraProjectionError(
            "Active Camera projection supports Perspective and Orthographic cameras; "
            f"received {blender_type or '<empty>'!r}"
        )

    calc_matrix_camera = getattr(evaluated_camera, "calc_matrix_camera", None)
    if not callable(calc_matrix_camera):
        raise A1ActiveCameraProjectionError(
            "Evaluated active camera has no calc_matrix_camera()"
        )
    try:
        projection_matrix = calc_matrix_camera(
            resolved_depsgraph,
            x=texture_width,
            y=texture_height,
            scale_x=1.0,
            scale_y=1.0,
        )
    except Exception as exc:
        raise A1ActiveCameraProjectionError(
            "Unable to calculate active-camera projection matrix for export texture "
            f"{texture_width}x{texture_height}"
        ) from exc

    camera_id = str(
        getattr(camera, "name_full", None)
        or getattr(camera, "name", None)
        or ""
    ).strip()
    if not camera_id:
        raise A1ActiveCameraProjectionError("Active camera name is empty")

    return A1CameraProjectionFrame(
        camera_id=camera_id,
        kind=kind,
        texture_width=texture_width,
        texture_height=texture_height,
        clip_start=float(getattr(camera_data, "clip_start")),
        clip_end=float(getattr(camera_data, "clip_end")),
        view_matrix=_rotation_only_view_matrix(evaluated_camera),
        projection_matrix=_matrix_tuple(
            projection_matrix,
            "active camera projection matrix",
        ),
    )


__all__ = [
    "A1ActiveCameraProjectionError",
    "resolve_a1_active_camera_projection_frame",
]
