"""Regressions captured from the real 0.90.0 mushrooms and flower-shop scenes."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace

import pytest

from Blender_to_Spine2D_Mesh_Exporter.application.a1_geometry_preparation import (
    A1GeometryPreparationError,
    _validate_prepared_coverage,
    prepare_a1_geometry_regions,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.geometry import (
    EdgeId,
    FaceId,
    LoopId,
    MeshEdge,
    MeshFace,
    MeshLoop,
    MeshSnapshot,
    MeshSnapshotValidator,
    MeshVertex,
    SourceEdgeId,
    SourceFaceId,
    SourceLoopId,
    SourceVertexId,
    TriangulationError,
    TriangulationSettings,
    VertexId,
    triangulate_snapshot,
)


_MUSHROOMS_OBJECT_ID = "Plane.008"
_FLOWER_SHOP_OBJECT_ID = "banco"
# One lifted corner on a unit quad yields the exact centroid-plane residue reported by
# the real evaluated Plane.008 polygon: approximately 0.00018954694198463555.
_MUSHROOMS_WARP_HEIGHT = 0.0007581877679385422
_CAPTURED_MAXIMUM_PLANE_DISTANCE = 0.00018954694198463555
_MATERIAL_WARP_HEIGHT = 0.01


def _quad_snapshot(
    object_id: str,
    *,
    warp_height: float = 0.0,
) -> MeshSnapshot:
    """Build one ordered Blender-style n-gon with complete boundary lineage."""

    if not isinstance(object_id, str) or not object_id.strip():
        raise ValueError("object_id must be a non-empty string")
    if isinstance(warp_height, bool) or not isinstance(warp_height, (int, float)):
        raise TypeError("warp_height must be numeric")

    positions = (
        (-0.5, -0.5, 0.0),
        (0.5, -0.5, 0.0),
        (0.5, 0.5, float(warp_height)),
        (-0.5, 0.5, 0.0),
    )
    edge_pairs = (
        (0, 1),
        (1, 2),
        (2, 3),
        (0, 3),
    )
    edge_id_by_pair = {
        pair: EdgeId(index) for index, pair in enumerate(edge_pairs)
    }

    vertices = tuple(
        MeshVertex(
            id=VertexId(index),
            source_id=SourceVertexId(object_id, index),
            position=position,
            normal=(0.0, 0.0, 1.0),
        )
        for index, position in enumerate(positions)
    )
    edges = tuple(
        MeshEdge(
            id=edge_id_by_pair[pair],
            source_id=SourceEdgeId(object_id, edge_id_by_pair[pair].index),
            vertex_ids=(VertexId(pair[0]), VertexId(pair[1])),
        )
        for pair in edge_pairs
    )

    face_vertices = (0, 1, 2, 3)
    loops = tuple(
        MeshLoop(
            id=LoopId(corner_index),
            source_id=SourceLoopId(object_id, 0, corner_index),
            vertex_id=VertexId(vertex_index),
            edge_id=edge_id_by_pair[
                tuple(
                    sorted(
                        (
                            vertex_index,
                            face_vertices[(corner_index + 1) % len(face_vertices)],
                        )
                    )
                )
            ],
        )
        for corner_index, vertex_index in enumerate(face_vertices)
    )
    face = MeshFace(
        id=FaceId(0),
        source_id=SourceFaceId(object_id, 0),
        loop_ids=tuple(loop.id for loop in loops),
        material_index=0,
        normal=(0.0, 0.0, 1.0),
        smooth=True,
    )
    snapshot = MeshSnapshot(
        snapshot_id=f"{object_id}:captured-regression",
        source_object_id=object_id,
        object_name=object_id,
        vertices=vertices,
        edges=edges,
        loops=loops,
        faces=(face,),
    )
    MeshSnapshotValidator().validate_or_raise(snapshot)
    return snapshot


def test_mushrooms_plane_008_relative_warp_triangulates_deterministically() -> None:
    source = _quad_snapshot(
        _MUSHROOMS_OBJECT_ID,
        warp_height=_MUSHROOMS_WARP_HEIGHT,
    )

    first = triangulate_snapshot(source)
    second = triangulate_snapshot(source)

    assert first == second
    assert len(first.snapshot.faces) == 2
    assert len(first.generated_edge_ids) == 1
    assert tuple(face.source_id for face in first.snapshot.faces) == (
        SourceFaceId(_MUSHROOMS_OBJECT_ID, 0),
        SourceFaceId(_MUSHROOMS_OBJECT_ID, 0),
    )


def test_default_planarity_window_rejects_percent_level_warp() -> None:
    settings = TriangulationSettings()
    assert settings.relative_planarity_tolerance == pytest.approx(2.5e-4)
    assert settings.normal_alignment_tolerance_degrees == pytest.approx(1.0)

    source = _quad_snapshot(
        _MUSHROOMS_OBJECT_ID,
        warp_height=_MATERIAL_WARP_HEIGHT,
    )
    with pytest.raises(TriangulationError, match="Polygon is not planar"):
        triangulate_snapshot(source, settings)


def test_mushrooms_plane_008_warp_remains_rejectable_under_explicit_strict_policy() -> None:
    source = _quad_snapshot(
        _MUSHROOMS_OBJECT_ID,
        warp_height=_MUSHROOMS_WARP_HEIGHT,
    )
    strict = TriangulationSettings(
        planarity_tolerance=1.0e-6,
        relative_planarity_tolerance=1.0e-5,
    )

    with pytest.raises(TriangulationError) as captured:
        triangulate_snapshot(source, strict)

    message = str(captured.value)
    assert "maximum plane distance" in message
    assert "effective tolerance" in message
    assert "relative=1e-05" in message
    # Keep the real-scene failure magnitude visible in the contract without relying on
    # an exact floating-point string representation.
    assert _CAPTURED_MAXIMUM_PLANE_DISTANCE > strict.planarity_tolerance


def test_flower_shop_banco_repeated_source_face_lineage_is_valid_coverage() -> None:
    source = _quad_snapshot(_FLOWER_SHOP_OBJECT_ID)
    triangulated = triangulate_snapshot(source).snapshot

    assert len(triangulated.faces) == 2
    assert len({face.id for face in triangulated.faces}) == 2
    assert len({face.source_id for face in triangulated.faces}) == 1

    prepared = prepare_a1_geometry_regions(triangulated)

    local_faces = tuple(
        face.id
        for region in prepared.regions
        for face in region.snapshot.faces
    )
    assert len(local_faces) == 2
    expected_lineage = Counter(face.source_id for face in triangulated.faces)
    prepared_lineage = Counter(
        source_face_id
        for region in prepared.regions
        for source_face_id in region.source_face_ids
    )
    assert prepared_lineage == expected_lineage
    assert prepared_lineage[SourceFaceId(_FLOWER_SHOP_OBJECT_ID, 0)] == 2


def test_flower_shop_coverage_still_rejects_lost_lineage_occurrence() -> None:
    triangulated = triangulate_snapshot(
        _quad_snapshot(_FLOWER_SHOP_OBJECT_ID)
    ).snapshot
    prepared = prepare_a1_geometry_regions(triangulated)
    first = prepared.regions[0]
    assert len(first.source_face_ids) >= 2
    damaged_first = replace(
        first,
        source_face_ids=first.source_face_ids[:-1],
    )
    damaged_regions = (damaged_first, *prepared.regions[1:])

    with pytest.raises(
        A1GeometryPreparationError,
        match="SourceFaceId lineage differs from decomposition plan",
    ):
        _validate_prepared_coverage(
            triangulated,
            prepared.decomposition,
            damaged_regions,
        )
