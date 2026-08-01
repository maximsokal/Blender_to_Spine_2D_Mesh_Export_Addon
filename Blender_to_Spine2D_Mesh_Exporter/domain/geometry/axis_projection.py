"""Project one normalized MeshSnapshot into canonical signed-axis space.

The input snapshot must already have its object linear world transform baked into vertex
positions and normals by ``normalize_mesh_snapshot_world_transform``. Consequently its
``world_matrix`` owns translation only. This module then applies the approved right-handed
``U/V/D`` basis to both local geometry and Object Origin translation without mutating the
source snapshot or Blender data.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import isfinite

from ..projection import (
    A1ProjectedPoint,
    A1ProjectionDirection,
    resolve_a1_axis_projection_basis,
)
from .model import Matrix4x4, MeshSnapshot, Vector3
from .validator import MeshSnapshotValidator


class A1MeshAxisProjectionError(ValueError):
    """Raised when a snapshot cannot enter the signed-axis projection pipeline."""


@dataclass(frozen=True, slots=True)
class A1MeshAxisProjectionResult:
    """One immutable projected snapshot and its projected Blender Object Origin."""

    snapshot: MeshSnapshot
    direction: A1ProjectionDirection
    projected_origin: A1ProjectedPoint
    changed: bool

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot, MeshSnapshot):
            raise TypeError("snapshot must be MeshSnapshot")
        if not isinstance(self.direction, A1ProjectionDirection):
            raise TypeError("direction must be A1ProjectionDirection")
        if not self.direction.axis_aligned:
            raise ValueError("A1MeshAxisProjectionResult requires an axis direction")
        if not isinstance(self.projected_origin, A1ProjectedPoint):
            raise TypeError("projected_origin must be A1ProjectedPoint")
        if not isinstance(self.changed, bool):
            raise TypeError("changed must be bool")


def _translation_only_origin(matrix: Matrix4x4) -> Vector3:
    """Return translation while rejecting a non-normalized affine matrix."""

    if not isinstance(matrix, tuple) or len(matrix) != 16:
        raise TypeError("snapshot.world_matrix must be a 16-value tuple")
    values = tuple(float(value) for value in matrix)
    if not all(isfinite(value) for value in values):
        raise A1MeshAxisProjectionError("snapshot.world_matrix contains non-finite values")

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
        raise A1MeshAxisProjectionError(
            "snapshot.world_matrix must contain translation only; call "
            "normalize_mesh_snapshot_world_transform first; "
            f"mismatches={mismatches}"
        )
    return (
        0.0 if values[3] == 0.0 else values[3],
        0.0 if values[7] == 0.0 else values[7],
        0.0 if values[11] == 0.0 else values[11],
    )


def _translation_matrix(origin: A1ProjectedPoint) -> Matrix4x4:
    if not isinstance(origin, A1ProjectedPoint):
        raise TypeError("origin must be A1ProjectedPoint")
    return (
        1.0,
        0.0,
        0.0,
        origin.u,
        0.0,
        1.0,
        0.0,
        origin.v,
        0.0,
        0.0,
        1.0,
        origin.depth,
        0.0,
        0.0,
        0.0,
        1.0,
    )


def project_a1_mesh_snapshot_axis(
    snapshot: MeshSnapshot,
    direction: A1ProjectionDirection,
) -> A1MeshAxisProjectionResult:
    """Project normalized local geometry and Object Origin into canonical ``U/V/D``.

    ``+Z`` is the exact compatibility path and returns the original snapshot instance.
    Other signed axes create a validated replacement snapshot. All bases are orthonormal
    and right-handed, so topology, winding, UVs, source lineage, and normal orientation
    remain valid without any index remapping.
    """

    if not isinstance(snapshot, MeshSnapshot):
        raise TypeError("snapshot must be MeshSnapshot")
    if not isinstance(direction, A1ProjectionDirection):
        raise TypeError("direction must be A1ProjectionDirection")

    basis = resolve_a1_axis_projection_basis(direction)
    world_origin = _translation_only_origin(snapshot.world_matrix)
    projected_origin = basis.project_point(world_origin)

    if direction is A1ProjectionDirection.POSITIVE_Z:
        return A1MeshAxisProjectionResult(
            snapshot=snapshot,
            direction=direction,
            projected_origin=projected_origin,
            changed=False,
        )

    vertices = tuple(
        replace(
            vertex,
            position=basis.project_vector(vertex.position),
            normal=basis.project_vector(vertex.normal),
        )
        for vertex in snapshot.vertices
    )
    faces = tuple(
        replace(
            face,
            normal=basis.project_vector(face.normal),
        )
        for face in snapshot.faces
    )
    projected_snapshot = replace(
        snapshot,
        vertices=vertices,
        faces=faces,
        world_matrix=_translation_matrix(projected_origin),
    )
    MeshSnapshotValidator().validate_or_raise(projected_snapshot)

    return A1MeshAxisProjectionResult(
        snapshot=projected_snapshot,
        direction=direction,
        projected_origin=projected_origin,
        changed=True,
    )


__all__ = [
    "A1MeshAxisProjectionError",
    "A1MeshAxisProjectionResult",
    "project_a1_mesh_snapshot_axis",
]
