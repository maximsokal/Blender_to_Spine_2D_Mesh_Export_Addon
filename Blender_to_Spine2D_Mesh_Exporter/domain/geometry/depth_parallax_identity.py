"""Canonical working identity for generated Depth parallax union geometry.

Front relief triangles and reserve triangles originate from different topology domains:
front faces are generated screen-space triangles, while reserve faces retain evaluated
Blender polygon ownership for isolated rendering. Their historical integer indices may
therefore collide even though the derived faces are unrelated.

This module rebases the completed union to one dense local identity domain, then rebuilds
front and reserve subsets from that canonical union. Render ownership remains stored in
``front_face_indices``, ``reserve_face_indices``, and each reserve surface's
``source_face_indices``; downstream segmentation and UV correspondence receive unique
SourceFaceId and SourceLoopId values.
"""

from __future__ import annotations

from dataclasses import replace
import logging

from .depth_parallax import (
    DepthParallaxGeometryPackage,
    DepthParallaxReserveSurface,
    _subset_material,
)
from .evaluated_identity import rebase_mesh_snapshot_to_evaluated_identity
from .validator import MeshSnapshotValidator


logger = logging.getLogger(__name__)


def canonicalize_depth_parallax_package_identity(
    package: DepthParallaxGeometryPackage,
    *,
    uv_layer_name: str,
) -> DepthParallaxGeometryPackage:
    """Return one package whose union and subsets share unique working lineage.

    The incoming package has already completed source provenance checks and render-view
    assignment. Only generated working identities are changed; geometry, UV coordinates,
    materials, camera plans, and evaluated source-face ownership remain unchanged.
    """

    if not isinstance(package, DepthParallaxGeometryPackage):
        raise TypeError("package must be DepthParallaxGeometryPackage")
    if not isinstance(uv_layer_name, str) or not uv_layer_name.strip():
        raise ValueError("uv_layer_name must be a non-empty string")

    MeshSnapshotValidator().validate_or_raise(package.union_snapshot)
    rebase = rebase_mesh_snapshot_to_evaluated_identity(
        package.union_snapshot
    )
    union = rebase.snapshot

    source_face_ids = tuple(face.source_id for face in union.faces)
    source_loop_ids = tuple(loop.source_id for loop in union.loops)
    if len(source_face_ids) != len(set(source_face_ids)):
        raise ValueError(
            "Canonical parallax union still contains duplicate SourceFaceId values"
        )
    if len(source_loop_ids) != len(set(source_loop_ids)):
        raise ValueError(
            "Canonical parallax union still contains duplicate SourceLoopId values"
        )

    front = _subset_material(
        union,
        0,
        uv_layer_name=uv_layer_name,
        suffix="parallax-front-canonical",
    )
    reserve_surfaces = tuple(
        replace(
            surface,
            snapshot=_subset_material(
                union,
                surface.view.material_index,
                uv_layer_name=uv_layer_name,
                suffix=(
                    "parallax-"
                    f"{surface.view.view_id.value.lower()}-canonical"
                ),
            ),
        )
        for surface in package.reserve_surfaces
    )
    if not all(
        isinstance(surface, DepthParallaxReserveSurface)
        for surface in reserve_surfaces
    ):
        raise TypeError(
            "reserve_surfaces must contain DepthParallaxReserveSurface values"
        )

    result = replace(
        package,
        front_result=replace(package.front_result, snapshot=front),
        union_snapshot=union,
        front_snapshot=front,
        reserve_surfaces=reserve_surfaces,
    )
    MeshSnapshotValidator().validate_or_raise(result.union_snapshot)
    MeshSnapshotValidator().validate_or_raise(result.front_snapshot)
    for surface in result.reserve_surfaces:
        MeshSnapshotValidator().validate_or_raise(surface.snapshot)

    logger.info(
        "Canonicalized Depth parallax package identity for '%s': changed=%s "
        "faces=%d loops=%d reserve_surfaces=%d",
        union.source_object_id,
        rebase.changed,
        len(union.faces),
        len(union.loops),
        len(result.reserve_surfaces),
    )
    return result


__all__ = ["canonicalize_depth_parallax_package_identity"]
