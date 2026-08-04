"""Canonical working identity for generated Depth parallax union geometry.

Front relief triangles and reserve triangles originate from different topology domains:
front faces are generated screen-space triangles, while reserve faces retain evaluated
Blender polygon ownership for isolated rendering. Their historical integer indices may
therefore collide even though the derived faces are unrelated.

This module rebases the completed union to one dense local identity domain, then rebuilds
front and reserve subsets from that canonical union. Reserve render ownership is resolved
from the pre-rebase SourceFaceId provenance, so triangulated n-gons still isolate the
correct evaluated Blender polygons.
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
from .model import MeshSnapshot
from .validator import MeshSnapshotValidator


logger = logging.getLogger(__name__)


def _evaluated_render_face_indices(
    surface: DepthParallaxReserveSurface,
) -> tuple[int, ...]:
    """Resolve original evaluated polygon indices from reserve provenance.

    ``surface.source_face_indices`` may still contain triangulated working indices from
    horizon expansion. The reserve snapshot itself retains SourceFaceId values copied
    from evaluated Blender polygons, including legal repetition when one n-gon emitted
    several triangles. Those historical values are the only valid indices for BMesh face
    isolation on the temporary evaluated render proxy.
    """

    if not isinstance(surface, DepthParallaxReserveSurface):
        raise TypeError("surface must be DepthParallaxReserveSurface")
    MeshSnapshotValidator().validate_or_raise(surface.snapshot)
    resolved = tuple(
        sorted(
            {
                int(face.source_id.face_index)
                for face in surface.snapshot.faces
            }
        )
    )
    if not resolved:
        raise ValueError(
            f"Reserve view {surface.view.view_id.value} has no evaluated face ownership"
        )
    return resolved


def _canonical_reserve_surface(
    surface: DepthParallaxReserveSurface,
    union: MeshSnapshot,
    *,
    uv_layer_name: str,
) -> DepthParallaxReserveSurface:
    """Rebuild one reserve subset and preserve evaluated render ownership."""

    if not isinstance(surface, DepthParallaxReserveSurface):
        raise TypeError("surface must be DepthParallaxReserveSurface")
    if not isinstance(union, MeshSnapshot):
        raise TypeError("union must be MeshSnapshot")
    if not isinstance(uv_layer_name, str) or not uv_layer_name.strip():
        raise ValueError("uv_layer_name must be a non-empty string")

    render_face_indices = _evaluated_render_face_indices(surface)
    snapshot = _subset_material(
        union,
        surface.view.material_index,
        uv_layer_name=uv_layer_name,
        suffix=(
            "parallax-"
            f"{surface.view.view_id.value.lower()}-canonical"
        ),
    )
    return replace(
        surface,
        snapshot=snapshot,
        source_face_indices=render_face_indices,
    )


def canonicalize_depth_parallax_package_identity(
    package: DepthParallaxGeometryPackage,
    *,
    uv_layer_name: str,
) -> DepthParallaxGeometryPackage:
    """Return one package whose union and subsets share unique working lineage.

    The incoming package has already completed source provenance checks and render-view
    assignment. Generated working IDs are canonicalized, while reserve render ownership
    is restored to original evaluated polygon indices before the old lineage is replaced.
    """

    if not isinstance(package, DepthParallaxGeometryPackage):
        raise TypeError("package must be DepthParallaxGeometryPackage")
    if not isinstance(uv_layer_name, str) or not uv_layer_name.strip():
        raise ValueError("uv_layer_name must be a non-empty string")

    MeshSnapshotValidator().validate_or_raise(package.union_snapshot)
    evaluated_render_owners = tuple(
        _evaluated_render_face_indices(surface)
        for surface in package.reserve_surfaces
    )

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
        _canonical_reserve_surface(
            surface,
            union,
            uv_layer_name=uv_layer_name,
        )
        for surface in package.reserve_surfaces
    )
    actual_render_owners = tuple(
        surface.source_face_indices for surface in reserve_surfaces
    )
    if actual_render_owners != evaluated_render_owners:
        raise ValueError(
            "Canonical reserve surfaces changed evaluated render ownership; "
            f"expected={evaluated_render_owners}, actual={actual_render_owners}"
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
        "faces=%d loops=%d reserve_surfaces=%d render_owners=%s",
        union.source_object_id,
        rebase.changed,
        len(union.faces),
        len(union.loops),
        len(result.reserve_surfaces),
        actual_render_owners,
    )
    return result


__all__ = ["canonicalize_depth_parallax_package_identity"]
