"""Project normalized Mesh snapshots into canonical signed-axis space.

The input snapshot must already have its object linear world transform baked into vertex
positions and normals by ``normalize_mesh_snapshot_world_transform``. Consequently its
``world_matrix`` owns translation only. This module applies the approved right-handed
``U/V/D`` basis to local geometry and Blender Object Origin translation without mutating
the source snapshot or Blender data. It also resolves deterministic world-space projected
depth bounds for later object-block draw-order planning.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import isfinite

from ..projection import (
    A1ProjectedPoint,
    A1ProjectionDirection,
    resolve_a1_axis_projection_basis,
)
from .ids import VertexId
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


@dataclass(frozen=True, slots=True)
class A1ProjectedSnapshotDepthRange:
    """World-space canonical depth bounds of one projected immutable snapshot.

    Canonical depth increases toward the selected observer. Therefore
    ``nearest_vertex_depth`` is the maximum world depth and
    ``farthest_vertex_depth`` is the minimum world depth. Vertex identities are retained
    for deterministic diagnostics and later camera-projection parity.
    """

    origin_depth: float
    nearest_vertex_id: VertexId
    nearest_vertex_depth: float
    farthest_vertex_id: VertexId
    farthest_vertex_depth: float

    def __post_init__(self) -> None:
        for field_name in (
            "origin_depth",
            "nearest_vertex_depth",
            "farthest_vertex_depth",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{field_name} must be a finite number")
            if not isfinite(float(value)):
                raise ValueError(f"{field_name} must be finite")
        if not isinstance(self.nearest_vertex_id, VertexId):
            raise TypeError("nearest_vertex_id must be VertexId")
        if not isinstance(self.farthest_vertex_id, VertexId):
            raise TypeError("farthest_vertex_id must be VertexId")
        if self.farthest_vertex_depth > self.nearest_vertex_depth:
            raise ValueError(
                "farthest_vertex_depth cannot exceed nearest_vertex_depth"
            )

    @property
    def depth_span(self) -> float:
        return float(self.nearest_vertex_depth - self.farthest_vertex_depth)


def _normalized_zero(value: float) -> float:
    resolved = float(value)
    return 0.0 if resolved == 0.0 else resolved


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
        _normalized_zero(values[3]),
        _normalized_zero(values[7]),
        _normalized_zero(values[11]),
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


def calculate_a1_projected_snapshot_depth_range(
    snapshot: MeshSnapshot,
) -> A1ProjectedSnapshotDepthRange:
    """Resolve nearest and farthest world depths from canonical projected geometry.

    The snapshot must already be normalized and projected. Its local vertex Z component
    is canonical depth relative to Blender Object Origin, while ``world_matrix[11]`` is
    projected Object Origin depth. Ties select the lowest stable ``VertexId`` so reports
    remain byte-deterministic.
    """

    if not isinstance(snapshot, MeshSnapshot):
        raise TypeError("snapshot must be MeshSnapshot")
    MeshSnapshotValidator().validate_or_raise(snapshot)
    if not snapshot.vertices:
        raise A1MeshAxisProjectionError(
            "projected snapshot must contain at least one vertex"
        )

    origin_depth = float(_translation_only_origin(snapshot.world_matrix)[2])
    records = tuple(
        (
            _normalized_zero(origin_depth + float(vertex.position[2])),
            vertex.id,
        )
        for vertex in snapshot.vertices
    )
    if not all(isfinite(depth) for depth, _ in records):
        raise A1MeshAxisProjectionError(
            "projected snapshot contains non-finite world depth"
        )

    nearest_depth, nearest_vertex_id = min(
        records,
        key=lambda item: (-item[0], item[1].index),
    )
    farthest_depth, farthest_vertex_id = min(
        records,
        key=lambda item: (item[0], item[1].index),
    )
    return A1ProjectedSnapshotDepthRange(
        origin_depth=_normalized_zero(origin_depth),
        nearest_vertex_id=nearest_vertex_id,
        nearest_vertex_depth=_normalized_zero(nearest_depth),
        farthest_vertex_id=farthest_vertex_id,
        farthest_vertex_depth=_normalized_zero(farthest_depth),
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
    "A1ProjectedSnapshotDepthRange",
    "calculate_a1_projected_snapshot_depth_range",
    "project_a1_mesh_snapshot_axis",
]
