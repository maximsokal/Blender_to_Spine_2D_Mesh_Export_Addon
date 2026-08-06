"""Preserve every prepared Normal / UV Segments region for Spine rig export.

A three-dimensional face can have zero area in the selected setup-pose X/Y projection
while still being valid, textured geometry. Side walls of a coin are the canonical
example: they may be edge-on for one signed-axis setup pose, but the generated Spine
rotation controls reveal them later. Removing such faces during document assembly loses
real source topology and makes different Normal projection directions export different
meshes.

The historical public function name is retained for API compatibility. It now validates
the projected region and returns it unchanged. Camera Projection and Depth Camera
Projection own separate visibility/hull pipelines and are not affected by this module.
"""

from __future__ import annotations

from math import isfinite

from ..domain.geometry import MeshSnapshot, MeshSnapshotValidator


class A1ProjectedRegionFilterError(ValueError):
    """Raised when a prepared Normal region is invalid for Spine serialization."""


def _require_finite_number(
    value: object,
    field_name: str,
    *,
    positive: bool = False,
) -> float:
    """Validate one numeric assembly parameter without accepting booleans."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a finite number")
    resolved = float(value)
    if not isfinite(resolved):
        raise ValueError(f"{field_name} must be finite")
    if positive and resolved <= 0.0:
        raise ValueError(f"{field_name} must be greater than zero")
    return resolved


def _validate_triangulated_region(snapshot: MeshSnapshot) -> None:
    """Require complete finite triangle topology before attachment projection."""

    MeshSnapshotValidator().validate_or_raise(snapshot)
    non_triangles = tuple(
        face.id.index
        for face in snapshot.faces
        if len(face.loop_ids) != 3
    )
    if non_triangles:
        raise A1ProjectedRegionFilterError(
            "Normal projected-region retention requires triangulated input; "
            f"non_triangle_faces={non_triangles}"
        )

    invalid_vertices = tuple(
        vertex.id.index
        for vertex in snapshot.vertices
        if not all(isfinite(float(component)) for component in vertex.position)
    )
    if invalid_vertices:
        raise A1ProjectedRegionFilterError(
            "Normal projected region contains non-finite vertex positions; "
            f"vertex_ids={invalid_vertices}"
        )


def split_xy_visible_region_snapshots(
    snapshot: MeshSnapshot,
    *,
    uniform_scale: float,
    center_x: float,
    center_y: float,
) -> tuple[MeshSnapshot, ...]:
    """Return the complete prepared region without deleting edge-on faces.

    ``uniform_scale`` and the attachment centre remain validated because they are part
    of the public assembly contract and malformed values must still fail closed. They
    intentionally do not influence topology ownership.

    A valid region always produces exactly one output snapshot and preserves object
    identity, vertices, edges, loops, faces, UVs, normals, and Source* lineage byte for
    byte. Faces that collapse to a line only in the current setup pose remain available
    to the generated Spine rig after X/Y rotation.
    """

    if not isinstance(snapshot, MeshSnapshot):
        raise TypeError("snapshot must be MeshSnapshot")

    _require_finite_number(
        uniform_scale,
        "uniform_scale",
        positive=True,
    )
    _require_finite_number(center_x, "center_x")
    _require_finite_number(center_y, "center_y")
    _validate_triangulated_region(snapshot)

    return (snapshot,)


__all__ = [
    "A1ProjectedRegionFilterError",
    "split_xy_visible_region_snapshots",
]
