"""Regress single-pass dense parallax analysis without wall-clock assumptions."""

from __future__ import annotations

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
    MeshSnapshotValidator,
    MeshVertex,
    SourceFaceId,
    SourceLoopId,
    SourceVertexId,
    VertexId,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.geometry.depth_camera_projection import (
    _ProjectedTriangle,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.geometry.depth_camera_projection_visible_topology import (
    _ClipPoint,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.geometry.depth_parallax_optimized import (
    _VisibilityTriangle,
    _build_occupied_grid,
    _build_source_analysis,
)


_OBJECT_ID = "OptimizedParallax"
_IDENTITY = (
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
)
_TRANSLATED_IN_FRONT = (
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, -5.0,
    0.0, 0.0, 0.0, 1.0,
)


def _frame() -> A1CameraProjectionFrame:
    return A1CameraProjectionFrame(
        camera_id="Camera",
        kind=A1CameraProjectionKind.ORTHOGRAPHIC,
        texture_width=1024,
        texture_height=1024,
        clip_start=0.1,
        clip_end=100.0,
        view_matrix=_IDENTITY,
        projection_matrix=_IDENTITY,
    )


def test_occupied_grid_keeps_localized_dense_queries_local() -> None:
    triangles: dict[int, _VisibilityTriangle] = {}
    polygons: dict[int, tuple[_ClipPoint, ...]] = {}

    face_index = 0
    for row in range(20):
        for column in range(20):
            minimum_x = -10.0 + float(column)
            minimum_y = -10.0 + float(row)
            projected = _ProjectedTriangle(
                face_index=face_index,
                points=(
                    (minimum_x, minimum_y, -5.0),
                    (minimum_x + 0.8, minimum_y, -5.0),
                    (minimum_x, minimum_y + 0.8, -5.0),
                ),
            )
            triangles[face_index] = _VisibilityTriangle.from_projected(projected)
            polygons[face_index] = (
                _ClipPoint(minimum_x, minimum_y, -5.0, None),
                _ClipPoint(minimum_x + 0.8, minimum_y, -5.0, None),
                _ClipPoint(minimum_x, minimum_y + 0.8, -5.0, None),
            )
            face_index += 1

    grid = _build_occupied_grid(triangles, polygons)
    candidates = grid.candidates(
        -9.7,
        -9.7,
        expected_face_index=0,
    )

    assert grid.columns > 1
    assert grid.rows > 1
    assert candidates
    assert 0 in {triangle.face_index for triangle in candidates}
    assert len(candidates) < 64
    assert len(candidates) < len(triangles) // 4


def test_grid_boundary_probe_retains_expected_face_without_global_scan() -> None:
    projected = _ProjectedTriangle(
        face_index=7,
        points=(
            (-1.0, -1.0, -5.0),
            (1.0, -1.0, -5.0),
            (-1.0, 1.0, -5.0),
        ),
    )
    triangle = _VisibilityTriangle.from_projected(projected)
    polygon = (
        _ClipPoint(-1.0, -1.0, -5.0, None),
        _ClipPoint(1.0, -1.0, -5.0, None),
        _ClipPoint(-1.0, 1.0, -5.0, None),
    )
    grid = _build_occupied_grid({7: triangle}, {7: polygon})

    candidates = grid.candidates(
        1.0,
        -1.0,
        expected_face_index=7,
    )

    assert tuple(candidate.face_index for candidate in candidates) == (7,)


def _duplicated_lineage_disconnected_snapshot() -> MeshSnapshot:
    positions = (
        (-0.8, -0.3, 0.0),
        (-0.2, -0.3, 0.0),
        (-0.5, 0.3, 0.0),
        (0.2, -0.3, 0.0),
        (0.8, -0.3, 0.0),
        (0.5, 0.3, 0.0),
    )
    face_vertices = (
        (0, 1, 2),
        (3, 4, 5),
    )
    edge_pairs = tuple(
        sorted(
            {
                tuple(sorted((face[index], face[(index + 1) % 3])))
                for face in face_vertices
                for index in range(3)
            }
        )
    )
    edge_id_by_pair = {
        pair: EdgeId(index) for index, pair in enumerate(edge_pairs)
    }
    vertices = tuple(
        MeshVertex(
            id=VertexId(index),
            # Both disconnected triangles intentionally reuse lineage 0,1,2.
            source_id=SourceVertexId(_OBJECT_ID, index % 3),
            position=position,
            normal=(0.0, 0.0, 1.0),
        )
        for index, position in enumerate(positions)
    )
    edges = tuple(
        MeshEdge(
            id=edge_id_by_pair[pair],
            source_id=None,
            vertex_ids=(VertexId(pair[0]), VertexId(pair[1])),
        )
        for pair in edge_pairs
    )

    loops: list[MeshLoop] = []
    faces: list[MeshFace] = []
    for face_index, vertices_for_face in enumerate(face_vertices):
        loop_ids: list[LoopId] = []
        for corner_index, vertex_index in enumerate(vertices_for_face):
            following = vertices_for_face[(corner_index + 1) % 3]
            loop_id = LoopId(len(loops))
            loops.append(
                MeshLoop(
                    id=loop_id,
                    source_id=SourceLoopId(
                        _OBJECT_ID,
                        face_index,
                        corner_index,
                    ),
                    vertex_id=VertexId(vertex_index),
                    edge_id=edge_id_by_pair[
                        tuple(sorted((vertex_index, following)))
                    ],
                )
            )
            loop_ids.append(loop_id)
        faces.append(
            MeshFace(
                id=FaceId(face_index),
                source_id=SourceFaceId(_OBJECT_ID, face_index),
                loop_ids=tuple(loop_ids),
                material_index=0,
                normal=(0.0, 0.0, 1.0),
                smooth=False,
            )
        )

    snapshot = MeshSnapshot(
        snapshot_id="DuplicatedLineageDisconnected",
        source_object_id=_OBJECT_ID,
        object_name="Duplicated Lineage Disconnected",
        vertices=vertices,
        edges=edges,
        loops=tuple(loops),
        faces=tuple(faces),
        world_matrix=_TRANSLATED_IN_FRONT,
    )
    MeshSnapshotValidator().validate_or_raise(snapshot)
    return snapshot


def test_analysis_adjacency_uses_local_vertices_not_duplicated_lineage() -> None:
    analysis = _build_source_analysis(
        _duplicated_lineage_disconnected_snapshot(),
        _frame(),
    )

    assert set(analysis.geometry) == {0, 1}
    assert analysis.adjacency == {0: (), 1: ()}
    assert (
        analysis.geometry[0].source_vertex_ids
        == analysis.geometry[1].source_vertex_ids
    )
