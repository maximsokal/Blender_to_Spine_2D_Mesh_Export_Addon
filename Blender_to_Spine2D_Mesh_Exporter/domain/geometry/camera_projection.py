"""Project one normalized MeshSnapshot through an evaluated active-camera frame.

The source snapshot must already have object rotation, scale, mirror and shear baked into
its local geometry, leaving only world translation in ``world_matrix``. Every world vertex
is projected independently, so the current Perspective or Orthographic camera view is
retained in X/Y.

Active Camera inside Normal / UV Segments is an object-space presentation mode, not the
flat Camera Projection export mode. The projected Blender Object Origin therefore remains
the exported object pivot, while every vertex keeps camera-space depth relative to that
origin. This preserves the ordinary multi-depth rig and allows the side surfaces retained
by Normal segmentation to appear when the Spine controls rotate the object.

The existing object-bake attachment projector converts internal Mesh Y to Spine Y by
negating it. Camera-projected local Y is therefore stored with the opposite sign so the
final Spine setup position matches camera screen-up coordinates without changing the
legacy signed-axis paths.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import isfinite

from ..camera_projection import A1CameraProjectionFrame
from ..projection import A1ProjectedPoint
from .model import Matrix4x4, MeshSnapshot, Vector3
from .validator import MeshSnapshotValidator


class A1MeshCameraProjectionError(ValueError):
    """Raised when a normalized mesh cannot be projected through an active camera."""


@dataclass(frozen=True, slots=True)
class A1MeshCameraProjectionResult:
    """One immutable camera-oriented snapshot and its projected Object Origin."""

    snapshot: MeshSnapshot
    frame: A1CameraProjectionFrame
    projected_origin: A1ProjectedPoint
    uniform_scale: float

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot, MeshSnapshot):
            raise TypeError("snapshot must be MeshSnapshot")
        if not isinstance(self.frame, A1CameraProjectionFrame):
            raise TypeError("frame must be A1CameraProjectionFrame")
        if not isinstance(self.projected_origin, A1ProjectedPoint):
            raise TypeError("projected_origin must be A1ProjectedPoint")
        if (
            isinstance(self.uniform_scale, bool)
            or not isinstance(self.uniform_scale, (int, float))
            or not isfinite(float(self.uniform_scale))
            or float(self.uniform_scale) <= 0.0
        ):
            raise ValueError("uniform_scale must be a finite positive number")
        object.__setattr__(self, "uniform_scale", float(self.uniform_scale))


@dataclass(frozen=True, slots=True)
class _ProjectedVertex:
    position: Vector3
    projected_world: A1ProjectedPoint


def _translation_only_origin(matrix: Matrix4x4) -> Vector3:
    if not isinstance(matrix, tuple) or len(matrix) != 16:
        raise TypeError("snapshot.world_matrix must be a 16-value tuple")
    values = tuple(float(value) for value in matrix)
    if not all(isfinite(value) for value in values):
        raise A1MeshCameraProjectionError(
            "snapshot.world_matrix contains non-finite values"
        )

    tolerance = 1.0e-10
    expected = (
        1.0,
        0.0,
        0.0,
        values[3],
        0.0,
        1.0,
        0.0,
        values[7],
        0.0,
        0.0,
        1.0,
        values[11],
        0.0,
        0.0,
        0.0,
        1.0,
    )
    mismatches = tuple(
        (index, actual, required)
        for index, (actual, required) in enumerate(zip(values, expected, strict=True))
        if abs(actual - required) > tolerance
    )
    if mismatches:
        raise A1MeshCameraProjectionError(
            "snapshot.world_matrix must contain translation only; call "
            "normalize_mesh_snapshot_world_transform first; "
            f"mismatches={mismatches}"
        )
    return (
        0.0 if values[3] == 0.0 else values[3],
        0.0 if values[7] == 0.0 else values[7],
        0.0 if values[11] == 0.0 else values[11],
    )


def _world_point(origin: Vector3, local_position: Vector3) -> Vector3:
    return (
        float(origin[0]) + float(local_position[0]),
        float(origin[1]) + float(local_position[1]),
        float(origin[2]) + float(local_position[2]),
    )


def _projected_translation_matrix(
    origin: A1ProjectedPoint,
    uniform_scale: float,
) -> Matrix4x4:
    """Store the projected Blender Object Origin as the ordinary object pivot.

    Camera X/Y are expressed in export pixels, while Normal attachment construction
    expects geometry units and multiplies them by ``uniform_scale`` exactly once. Camera
    depth remains in Blender/camera units, matching signed-axis Normal depth ownership.
    """

    if not isinstance(origin, A1ProjectedPoint):
        raise TypeError("origin must be A1ProjectedPoint")
    if (
        isinstance(uniform_scale, bool)
        or not isinstance(uniform_scale, (int, float))
        or not isfinite(float(uniform_scale))
        or float(uniform_scale) <= 0.0
    ):
        raise ValueError("uniform_scale must be a finite positive number")

    resolved_scale = float(uniform_scale)
    return (
        1.0,
        0.0,
        0.0,
        origin.u / resolved_scale,
        0.0,
        1.0,
        0.0,
        origin.v / resolved_scale,
        0.0,
        0.0,
        1.0,
        origin.depth,
        0.0,
        0.0,
        0.0,
        1.0,
    )


def _attachment_space_normal(
    frame: A1CameraProjectionFrame,
    normal: Vector3,
    *,
    field_name: str,
) -> Vector3:
    """Rotate to camera space, then apply the oriented Y-reflection cofactor."""

    camera_normal = frame.transform_world_direction(
        normal,
        field_name=field_name,
    )
    return (
        -camera_normal[0],
        camera_normal[1],
        -camera_normal[2],
    )


def project_a1_mesh_snapshot_camera(
    snapshot: MeshSnapshot,
    frame: A1CameraProjectionFrame,
    *,
    uniform_scale: float,
) -> A1MeshCameraProjectionResult:
    """Project Normal geometry to the active-camera view around Object Origin.

    Perspective or Orthographic X/Y is baked into the setup pose, but this is not a flat
    camera attachment. Every vertex retains ``camera_depth - object_origin_depth`` as its
    local Z and the projected Object Origin is retained in ``world_matrix``. Consequently
    downstream Z grouping and rig construction remain the same object-pivot pipeline used
    by signed-axis Normal export.

    Geometry outside the camera frame and geometry that is edge-on in the current setup
    pose are both retained.
    """

    if not isinstance(snapshot, MeshSnapshot):
        raise TypeError("snapshot must be MeshSnapshot")
    if not isinstance(frame, A1CameraProjectionFrame):
        raise TypeError("frame must be A1CameraProjectionFrame")
    if (
        isinstance(uniform_scale, bool)
        or not isinstance(uniform_scale, (int, float))
        or not isfinite(float(uniform_scale))
        or float(uniform_scale) <= 0.0
    ):
        raise ValueError("uniform_scale must be a finite positive number")
    resolved_scale = float(uniform_scale)

    MeshSnapshotValidator().validate_or_raise(snapshot)
    if not snapshot.vertices:
        raise A1MeshCameraProjectionError(
            "camera projection requires at least one mesh vertex"
        )

    world_origin = _translation_only_origin(snapshot.world_matrix)
    projected_origin = frame.project_world_point(
        world_origin,
        field_name="object_origin",
    )

    projected_vertices: list[_ProjectedVertex] = []
    for vertex in snapshot.vertices:
        projected_world = frame.project_world_point(
            _world_point(world_origin, vertex.position),
            field_name=f"vertex[{vertex.id.index}]",
        )
        projected_vertices.append(
            _ProjectedVertex(
                position=(
                    (projected_world.u - projected_origin.u) / resolved_scale,
                    -(projected_world.v - projected_origin.v) / resolved_scale,
                    float(projected_world.depth - projected_origin.depth),
                ),
                projected_world=projected_world,
            )
        )

    vertices = tuple(
        replace(
            vertex,
            position=projected.position,
            normal=_attachment_space_normal(
                frame,
                vertex.normal,
                field_name=f"vertex[{vertex.id.index}].normal",
            ),
        )
        for vertex, projected in zip(
            snapshot.vertices,
            projected_vertices,
            strict=True,
        )
    )
    faces = tuple(
        replace(
            face,
            normal=_attachment_space_normal(
                frame,
                face.normal,
                field_name=f"face[{face.id.index}].normal",
            ),
        )
        for face in snapshot.faces
    )
    projected_snapshot = replace(
        snapshot,
        vertices=vertices,
        faces=faces,
        world_matrix=_projected_translation_matrix(
            projected_origin,
            resolved_scale,
        ),
    )
    MeshSnapshotValidator().validate_or_raise(projected_snapshot)

    return A1MeshCameraProjectionResult(
        snapshot=projected_snapshot,
        frame=frame,
        projected_origin=projected_origin,
        uniform_scale=resolved_scale,
    )


__all__ = [
    "A1MeshCameraProjectionError",
    "A1MeshCameraProjectionResult",
    "project_a1_mesh_snapshot_camera",
]
