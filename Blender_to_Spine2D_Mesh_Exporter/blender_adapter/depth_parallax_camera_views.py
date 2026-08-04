# pylint: disable=import-error
"""Resolve fitted virtual camera views for Depth parallax reserve rendering.

The active camera remains the setup/front view. A positive horizon angle creates eight
virtual orbit views around the projected object origin. Every view uses the active camera
projection model and clipping planes, but receives an immutable camera-world override and
an optional lens/ortho fit scale so all evaluated source vertices remain inside its render.
No Blender datablock is created or mutated by this module.
"""

from __future__ import annotations

from math import isfinite, pi, sqrt
from typing import Any

from ..domain.camera_projection import A1CameraProjectionFrame
from ..domain.geometry import (
    DepthParallaxCameraView,
    DepthParallaxViewId,
    MeshSnapshot,
)
from ..domain.geometry.depth_camera_projection import (
    _translation_only_origin,
    _world_point,
)
from .active_camera_projection import (
    A1ActiveCameraProjectionError,
    _matrix_tuple,
    _resolved_scene_and_depsgraph,
)


_VIEW_COMPONENTS = (
    (DepthParallaxViewId.RIGHT, 1.0, 0.0),
    (DepthParallaxViewId.UP_RIGHT, 1.0 / sqrt(2.0), 1.0 / sqrt(2.0)),
    (DepthParallaxViewId.UP, 0.0, 1.0),
    (DepthParallaxViewId.UP_LEFT, -1.0 / sqrt(2.0), 1.0 / sqrt(2.0)),
    (DepthParallaxViewId.LEFT, -1.0, 0.0),
    (DepthParallaxViewId.DOWN_LEFT, -1.0 / sqrt(2.0), -1.0 / sqrt(2.0)),
    (DepthParallaxViewId.DOWN, 0.0, -1.0),
    (DepthParallaxViewId.DOWN_RIGHT, 1.0 / sqrt(2.0), -1.0 / sqrt(2.0)),
)


class DepthParallaxCameraViewError(ValueError):
    """Raised when virtual reserve cameras cannot be derived safely."""


def _load_mathutils() -> tuple[Any, Any, Any]:
    """Load Blender math types only when a positive reserve is actually resolved."""

    try:
        from mathutils import Matrix, Quaternion, Vector
    except Exception as exc:
        raise DepthParallaxCameraViewError(
            "Blender mathutils is unavailable for virtual parallax camera views"
        ) from exc
    return Matrix, Quaternion, Vector


def _active_camera_world_matrix(
    scene: Any,
    depsgraph: Any,
) -> tuple[Any, Any]:
    camera = getattr(scene, "camera", None)
    if camera is None or str(getattr(camera, "type", "")) != "CAMERA":
        raise DepthParallaxCameraViewError(
            "Depth parallax reserve requires an active CAMERA object"
        )
    evaluated_get = getattr(camera, "evaluated_get", None)
    if not callable(evaluated_get):
        raise DepthParallaxCameraViewError(
            "Active camera does not provide evaluated_get()"
        )
    try:
        evaluated = evaluated_get(depsgraph)
        location, rotation, _scale = evaluated.matrix_world.decompose()
        rotation.normalize()
        world = rotation.to_matrix().to_4x4()
        world.translation = location
    except Exception as exc:
        raise DepthParallaxCameraViewError(
            "Unable to resolve evaluated active-camera world matrix"
        ) from exc
    return camera, world


def _virtual_camera_world_matrix(
    camera_world: Any,
    pivot_world: Any,
    *,
    yaw_radians: float,
    pitch_radians: float,
) -> Any:
    _Matrix, Quaternion, Vector = _load_mathutils()
    location = camera_world.translation.copy()
    offset = location - pivot_world
    if offset.length_squared <= 1.0e-18:
        raise DepthParallaxCameraViewError(
            "Active camera coincides with the projected object origin"
        )

    rotation = camera_world.to_quaternion()
    rotation.normalize()
    up_world = rotation @ Vector((0.0, 1.0, 0.0))
    right_world = rotation @ Vector((1.0, 0.0, 0.0))

    yaw = Quaternion(up_world.normalized(), float(yaw_radians))
    right_after_yaw = (yaw @ right_world).normalized()
    pitch = Quaternion(right_after_yaw, float(pitch_radians))
    orbit = pitch @ yaw
    new_location = pivot_world + orbit @ offset

    direction = pivot_world - new_location
    if direction.length_squared <= 1.0e-18:
        raise DepthParallaxCameraViewError(
            "Virtual parallax camera collapsed onto the object origin"
        )
    try:
        new_rotation = direction.to_track_quat("-Z", "Y")
        world = new_rotation.to_matrix().to_4x4()
        world.translation = new_location
    except Exception as exc:
        raise DepthParallaxCameraViewError(
            "Unable to orient virtual parallax camera toward object origin"
        ) from exc
    return world


def _scaled_projection_matrix(
    values: tuple[float, ...],
    scale: float,
) -> tuple[float, ...]:
    if not isinstance(values, tuple) or len(values) != 16:
        raise TypeError("projection matrix must contain sixteen values")
    if not isfinite(scale) or scale <= 0.0 or scale > 1.0:
        raise ValueError("projection scale must be in (0, 1]")
    resolved = list(float(value) for value in values)
    for index in range(0, 4):
        resolved[index] *= scale
    for index in range(4, 8):
        resolved[index] *= scale
    return tuple(resolved)


def _fit_projection_scale(
    snapshot: MeshSnapshot,
    frame: A1CameraProjectionFrame,
    *,
    padding_fraction: float,
) -> float:
    origin = _translation_only_origin(snapshot.world_matrix)
    maximum = 0.0
    for vertex in snapshot.vertices:
        projected = frame.project_world_point(
            _world_point(origin, vertex.position),
            field_name=f"parallax_fit_vertex[{vertex.id.index}]",
        )
        ndc_x = abs(float(projected.u)) / (float(frame.texture_width) / 2.0)
        ndc_y = abs(float(projected.v)) / (float(frame.texture_height) / 2.0)
        maximum = max(maximum, ndc_x, ndc_y)
    if maximum <= padding_fraction:
        return 1.0
    scale = padding_fraction / maximum
    if not isfinite(scale) or scale <= 0.0:
        raise DepthParallaxCameraViewError(
            "Unable to fit evaluated source into virtual parallax camera"
        )
    return min(1.0, max(1.0e-6, float(scale)))


def resolve_depth_parallax_camera_views(
    scene: Any | None,
    source_snapshot: MeshSnapshot,
    front_frame: A1CameraProjectionFrame,
    *,
    horizon_angle_radians: float,
    depsgraph: Any | None = None,
    padding_fraction: float = 0.94,
) -> tuple[DepthParallaxCameraView, ...]:
    """Return eight fitted virtual views or an empty tuple for the zero-angle contract."""

    if not isinstance(source_snapshot, MeshSnapshot):
        raise TypeError("source_snapshot must be MeshSnapshot")
    if not isinstance(front_frame, A1CameraProjectionFrame):
        raise TypeError("front_frame must be A1CameraProjectionFrame")
    if (
        isinstance(horizon_angle_radians, bool)
        or not isinstance(horizon_angle_radians, (int, float))
    ):
        raise TypeError("horizon_angle_radians must be numeric")
    angle = float(horizon_angle_radians)
    if not isfinite(angle) or angle < 0.0 or angle >= pi / 2.0:
        raise ValueError("horizon_angle_radians must be finite in [0, pi/2)")
    if (
        isinstance(padding_fraction, bool)
        or not isinstance(padding_fraction, (int, float))
    ):
        raise TypeError("padding_fraction must be numeric")
    padding = float(padding_fraction)
    if not isfinite(padding) or padding <= 0.0 or padding >= 1.0:
        raise ValueError("padding_fraction must be finite in (0, 1)")
    if angle <= 1.0e-12:
        return ()

    _Matrix, _Quaternion, Vector = _load_mathutils()
    try:
        resolved_scene, resolved_depsgraph = _resolved_scene_and_depsgraph(
            scene,
            depsgraph,
        )
        _camera, camera_world = _active_camera_world_matrix(
            resolved_scene,
            resolved_depsgraph,
        )
    except A1ActiveCameraProjectionError as exc:
        raise DepthParallaxCameraViewError(str(exc)) from exc

    pivot_values = _translation_only_origin(source_snapshot.world_matrix)
    pivot = Vector(pivot_values)
    views = []
    for view_id, yaw_component, pitch_component in _VIEW_COMPONENTS:
        yaw = angle * yaw_component
        pitch = angle * pitch_component
        world = _virtual_camera_world_matrix(
            camera_world,
            pivot,
            yaw_radians=yaw,
            pitch_radians=pitch,
        )
        try:
            view_matrix = world.inverted()
        except Exception as exc:
            raise DepthParallaxCameraViewError(
                f"Virtual camera {view_id.value} matrix is not invertible"
            ) from exc
        provisional = A1CameraProjectionFrame(
            camera_id=f"{front_frame.camera_id}:PARALLAX:{view_id.value}",
            kind=front_frame.kind,
            texture_width=front_frame.texture_width,
            texture_height=front_frame.texture_height,
            clip_start=front_frame.clip_start,
            clip_end=front_frame.clip_end,
            view_matrix=_matrix_tuple(
                view_matrix,
                f"parallax {view_id.value} view matrix",
            ),
            projection_matrix=front_frame.projection_matrix,
        )
        lens_scale = _fit_projection_scale(
            source_snapshot,
            provisional,
            padding_fraction=padding,
        )
        frame = A1CameraProjectionFrame(
            camera_id=provisional.camera_id,
            kind=provisional.kind,
            texture_width=provisional.texture_width,
            texture_height=provisional.texture_height,
            clip_start=provisional.clip_start,
            clip_end=provisional.clip_end,
            view_matrix=provisional.view_matrix,
            projection_matrix=_scaled_projection_matrix(
                provisional.projection_matrix,
                lens_scale,
            ),
        )
        views.append(
            DepthParallaxCameraView(
                view_id=view_id,
                yaw_radians=yaw,
                pitch_radians=pitch,
                frame=frame,
                camera_world_matrix=_matrix_tuple(
                    world,
                    f"parallax {view_id.value} camera world matrix",
                ),
                lens_scale=lens_scale,
            )
        )
    return tuple(views)


__all__ = [
    "DepthParallaxCameraViewError",
    "resolve_depth_parallax_camera_views",
]
