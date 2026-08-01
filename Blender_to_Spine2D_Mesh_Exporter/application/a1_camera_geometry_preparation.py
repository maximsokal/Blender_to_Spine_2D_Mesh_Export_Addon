"""Project already-triangulated A1 geometry regions through an active camera.

Perspective projection is nonlinear in the retained ``(screen U, screen V, depth)``
space. A planar source polygon with four or more corners can therefore become
non-planar after projection even though every source face is valid Blender geometry.
The strict triangulator must run before that nonlinear transform. This module keeps
segmentation, decomposition and triangulation in normalized world geometry, then
projects only the immutable triangle-region snapshots used by downstream UV and
attachment stages.
"""

from __future__ import annotations

from dataclasses import replace
from math import isfinite

from ..domain.camera_projection import A1CameraProjectionFrame
from ..domain.geometry import (
    MeshSnapshotValidator,
    project_a1_mesh_snapshot_camera,
)
from .a1_geometry_preparation import A1GeometryPreparationResult


class A1CameraGeometryPreparationError(ValueError):
    """Raised when prepared regions cannot be projected without contract loss."""


def _require_uniform_scale(value: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError("uniform_scale must be a finite positive number")
    return float(value)


def _triangle_face_indices(snapshot) -> tuple[int, ...]:
    return tuple(
        face.id.index
        for face in snapshot.faces
        if len(face.loop_ids) != 3
    )


def project_a1_prepared_geometry_camera(
    geometry: A1GeometryPreparationResult,
    frame: A1CameraProjectionFrame,
    *,
    uniform_scale: float,
) -> A1GeometryPreparationResult:
    """Project immutable triangle regions after world-space geometry preparation.

    ``geometry.segmentation`` and ``geometry.decomposition`` remain the authoritative
    source-space plans. Only each region's already-triangulated snapshot is projected.
    This preserves exact face/loop/source lineage while avoiding a second planarity
    decision in nonlinear perspective ``U/V/depth`` space.
    """

    if not isinstance(geometry, A1GeometryPreparationResult):
        raise TypeError("geometry must be A1GeometryPreparationResult")
    if not isinstance(frame, A1CameraProjectionFrame):
        raise TypeError("frame must be A1CameraProjectionFrame")
    resolved_scale = _require_uniform_scale(uniform_scale)

    projected_regions = []
    reference_origin = None

    for region in geometry.regions:
        source_snapshot = region.triangulation.snapshot
        MeshSnapshotValidator().validate_or_raise(source_snapshot)

        non_triangles = _triangle_face_indices(source_snapshot)
        if non_triangles:
            raise A1CameraGeometryPreparationError(
                "Active Camera region projection requires pre-triangulated faces; "
                f"region={region.region_index}, non_triangle_faces={non_triangles}"
            )

        projection = project_a1_mesh_snapshot_camera(
            source_snapshot,
            frame,
            uniform_scale=resolved_scale,
        )
        projected_snapshot = projection.snapshot
        MeshSnapshotValidator().validate_or_raise(projected_snapshot)

        projected_non_triangles = _triangle_face_indices(projected_snapshot)
        if projected_non_triangles:
            raise A1CameraGeometryPreparationError(
                "Active Camera projection changed triangle topology; "
                f"region={region.region_index}, non_triangle_faces="
                f"{projected_non_triangles}"
            )

        if reference_origin is None:
            reference_origin = projection.projected_origin.canonical_position
        else:
            actual_origin = projection.projected_origin.canonical_position
            mismatches = tuple(
                (axis, actual, expected)
                for axis, (actual, expected) in enumerate(
                    zip(actual_origin, reference_origin, strict=True)
                )
                if abs(float(actual) - float(expected)) > 1.0e-10
            )
            if mismatches:
                raise A1CameraGeometryPreparationError(
                    "Prepared regions resolved different projected Object Origins; "
                    f"region={region.region_index}, mismatches={mismatches}"
                )

        projected_regions.append(
            replace(
                region,
                triangulation=replace(
                    region.triangulation,
                    snapshot=projected_snapshot,
                ),
            )
        )

    return replace(
        geometry,
        regions=tuple(projected_regions),
    )


__all__ = [
    "A1CameraGeometryPreparationError",
    "project_a1_prepared_geometry_camera",
]
