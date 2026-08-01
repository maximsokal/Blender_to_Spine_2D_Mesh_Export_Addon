"""Regress perspective n-gons that become non-planar in U/V/depth space."""

from __future__ import annotations

from math import isclose, sqrt

import pytest

from Blender_to_Spine2D_Mesh_Exporter.application import (
    prepare_a1_geometry_regions,
    project_a1_prepared_geometry_camera,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.camera_projection import (
    A1CameraProjectionFrame,
    A1CameraProjectionKind,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.geometry import (
    EdgeId,
    FaceId,
    LoopId,
    MeshEdge,
    MeshFace,
    MeshLoop,
    MeshSnapshot,
    MeshVertex,
    SourceEdgeId,
    SourceFaceId,
    SourceLoopId,
    SourceVertexId,
    TriangulationError,
    VertexId,
    project_a1_mesh_snapshot_camera,
    triangulate_snapshot,
)


_OBJECT_ID = "PerspectiveQuad"
_IDENTITY = (
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
)


def _translation_matrix(x: float, y: float, z: float) -> tuple[float, ...]:
    return (
        1.0,
        0.0,
        0.0,
        x,
        0.0,
        1.0,
        0.0,
        y,
        0.0,
        0.0,
        1.0,
        z,
        0.0,
        0.0,
        0.0,
        1.0,
    )


def _perspective_matrix(
    *,
    aspect_ratio: float,
    near: float = 0.1,
    far: float = 100.0,
) -> tuple[float, ...]:
    return (
        1.0 / aspect_ratio,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        (far + near) / (near - far),
        (2.0 * far * near) / (near - far),
        0.0,
        0.0,
        -1.0,
        0.0,
    )


def _frame() -> A1CameraProjectionFrame:
    width = 200
    height = 100
    return A1CameraProjectionFrame(
        camera_id="PerspectiveCamera",
        kind=A1CameraProjectionKind.PERSPECTIVE,
        texture_width=width,
        texture_height=height,
        clip_start=0.1,
        clip_end=100.0,
        view_matrix=_IDENTITY,
        projection_matrix=_perspective_matrix(
            aspect_ratio=float(width) / float(height)
        ),
    )


def _tilted_planar_quad() -> MeshSnapshot:
    # All four local points satisfy z = 0.25*x + 0.25*y + 0.5 exactly.
    # The source polygon is therefore planar, but perspective U/V plus retained
    # camera-local depth produces a non-planar four-corner snapshot.
    positions = (
        (-1.0, -1.0, 0.0),
        (1.0, -1.0, 0.5),
        (1.0, 1.0, 1.0),
        (-1.0, 1.0, 0.5),
    )
    magnitude = sqrt(1.0 + 0.25 * 0.25 + 0.25 * 0.25)
    normal = (-0.25 / magnitude, -0.25 / magnitude, 1.0 / magnitude)

    vertices = tuple(
        MeshVertex(
            id=VertexId(index),
            source_id=SourceVertexId(_OBJECT_ID, index),
            position=position,
            normal=normal,
        )
        for index, position in enumerate(positions)
    )
    edges = tuple(
        MeshEdge(
            id=EdgeId(index),
            source_id=SourceEdgeId(_OBJECT_ID, index),
            vertex_ids=(VertexId(index), VertexId((index + 1) % 4)),
        )
        for index in range(4)
    )
    loops = tuple(
        MeshLoop(
            id=LoopId(index),
            source_id=SourceLoopId(_OBJECT_ID, 0, index),
            vertex_id=VertexId(index),
            edge_id=EdgeId(index),
        )
        for index in range(4)
    )
    return MeshSnapshot(
        snapshot_id="perspective-planar-quad",
        source_object_id=_OBJECT_ID,
        object_name=_OBJECT_ID,
        vertices=vertices,
        edges=edges,
        loops=loops,
        faces=(
            MeshFace(
                id=FaceId(0),
                source_id=SourceFaceId(_OBJECT_ID, 0),
                loop_ids=tuple(loop.id for loop in loops),
                material_index=0,
                normal=normal,
            ),
        ),
        world_matrix=_translation_matrix(0.0, 0.0, -5.0),
    )


def test_perspective_projection_before_triangulation_reproduces_nonplanarity() -> None:
    source = _tilted_planar_quad()
    projected = project_a1_mesh_snapshot_camera(
        source,
        _frame(),
        uniform_scale=100.0,
    )

    with pytest.raises(TriangulationError, match="Polygon is not planar"):
        triangulate_snapshot(projected.snapshot)


def test_world_triangulation_then_camera_projection_preserves_regions_and_lineage() -> None:
    source = _tilted_planar_quad()
    frame = _frame()

    world_geometry = prepare_a1_geometry_regions(source)
    direct_projection = project_a1_mesh_snapshot_camera(
        source,
        frame,
        uniform_scale=100.0,
    )
    projected_geometry = project_a1_prepared_geometry_camera(
        world_geometry,
        frame,
        uniform_scale=100.0,
    )

    assert projected_geometry.source_snapshot_id == source.snapshot_id
    assert projected_geometry.segmentation is world_geometry.segmentation
    assert projected_geometry.decomposition is world_geometry.decomposition
    assert len(projected_geometry.regions) == 1

    world_region = world_geometry.regions[0]
    projected_region = projected_geometry.regions[0]
    assert len(world_region.snapshot.faces) == 2
    assert len(projected_region.snapshot.faces) == 2
    assert all(len(face.loop_ids) == 3 for face in projected_region.snapshot.faces)
    assert projected_region.source_face_ids == (SourceFaceId(_OBJECT_ID, 0),)
    assert projected_region.triangulation.faces == world_region.triangulation.faces
    assert (
        projected_region.triangulation.generated_edge_ids
        == world_region.triangulation.generated_edge_ids
    )

    expected_by_source = {
        vertex.source_id: vertex.position
        for vertex in direct_projection.snapshot.vertices
    }
    actual_by_source = {
        vertex.source_id: vertex.position
        for vertex in projected_region.snapshot.vertices
    }
    assert set(actual_by_source) == set(expected_by_source)
    for source_id, expected in expected_by_source.items():
        actual = actual_by_source[source_id]
        assert all(
            isclose(actual_value, expected_value, abs_tol=1.0e-12)
            for actual_value, expected_value in zip(actual, expected, strict=True)
        )

    assert projected_region.snapshot.world_matrix == direct_projection.snapshot.world_matrix
    assert tuple(vertex.position for vertex in source.vertices) == (
        (-1.0, -1.0, 0.0),
        (1.0, -1.0, 0.5),
        (1.0, 1.0, 1.0),
        (-1.0, 1.0, 0.5),
    )
