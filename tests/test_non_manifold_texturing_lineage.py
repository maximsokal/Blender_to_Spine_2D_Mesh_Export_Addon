import pytest

from Blender_to_Spine2D_Mesh_Exporter.application.a1_texturing_layout import (
    A1TexturingLayoutError,
    _resolve_geometry_cut_edge_ids,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.geometry import (
    EdgeId,
    FaceId,
    LoopId,
    LoopUV,
    MeshEdge,
    MeshFace,
    MeshLoop,
    MeshSnapshot,
    MeshVertex,
    SourceEdgeId,
    SourceFaceId,
    SourceLoopId,
    SourceVertexId,
    VertexId,
    build_edge_to_faces,
    split_non_manifold_edges,
)

from test_geometry_domain import build_square_snapshot


def _build_three_face_non_manifold_edge() -> MeshSnapshot:
    source = "NonManifold"
    vertices = tuple(
        MeshVertex(
            VertexId(index),
            SourceVertexId(source, index),
            position,
            (0.0, 0.0, 1.0),
        )
        for index, position in enumerate(
            (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, -1.0, 0.0),
                (0.0, 0.0, 1.0),
            )
        )
    )
    edges = (
        MeshEdge(EdgeId(0), SourceEdgeId(source, 0), (VertexId(0), VertexId(1))),
        MeshEdge(EdgeId(1), SourceEdgeId(source, 1), (VertexId(1), VertexId(2))),
        MeshEdge(EdgeId(2), SourceEdgeId(source, 2), (VertexId(2), VertexId(0))),
        MeshEdge(EdgeId(3), SourceEdgeId(source, 3), (VertexId(0), VertexId(3))),
        MeshEdge(EdgeId(4), SourceEdgeId(source, 4), (VertexId(3), VertexId(1))),
        MeshEdge(EdgeId(5), SourceEdgeId(source, 5), (VertexId(1), VertexId(4))),
        MeshEdge(EdgeId(6), SourceEdgeId(source, 6), (VertexId(4), VertexId(0))),
    )
    face_specs = (
        ((0, 1, 2), (0, 1, 2), (0.0, 0.0, 1.0)),
        ((1, 0, 3), (0, 3, 4), (0.0, 0.0, -1.0)),
        ((0, 1, 4), (0, 5, 6), (0.0, -1.0, 0.0)),
    )
    loops = []
    faces = []
    for face_index, (vertex_indices, edge_indices, normal) in enumerate(face_specs):
        loop_ids = []
        for corner_index, (vertex_index, edge_index) in enumerate(
            zip(vertex_indices, edge_indices, strict=True)
        ):
            loop_id = LoopId(len(loops))
            loop_ids.append(loop_id)
            loops.append(
                MeshLoop(
                    loop_id,
                    SourceLoopId(source, face_index, corner_index),
                    VertexId(vertex_index),
                    EdgeId(edge_index),
                    (
                        LoopUV(
                            "UVMap",
                            (
                                float(corner_index == 1),
                                float(corner_index == 2),
                            ),
                        ),
                    ),
                )
            )
        faces.append(
            MeshFace(
                FaceId(face_index),
                SourceFaceId(source, face_index),
                tuple(loop_ids),
                0,
                normal,
            )
        )
    return MeshSnapshot(
        snapshot_id="non-manifold-three-face-edge",
        source_object_id=source,
        object_name=source,
        vertices=vertices,
        edges=edges,
        loops=tuple(loops),
        faces=tuple(faces),
        uv_layer_names=("UVMap",),
        active_uv_layer="UVMap",
    )


def test_non_manifold_repair_preserves_original_edge_ids_and_appends_copies():
    source = _build_three_face_non_manifold_edge()
    assert len(build_edge_to_faces(source)[EdgeId(0)]) == 3

    repaired, report = split_non_manifold_edges(source)

    assert report.changed
    assert report.split_edge_ids == (EdgeId(0),)
    assert report.input_edge_count == 7
    assert report.output_edge_count == 9
    assert tuple(edge.id for edge in repaired.edges[:7]) == tuple(
        edge.id for edge in source.edges
    )
    assert tuple(
        (
            edge.source_id,
            edge.vertex_ids,
            edge.seam,
            edge.sharp,
        )
        for edge in repaired.edges[:7]
    ) == tuple(
        (
            edge.source_id,
            edge.vertex_ids,
            edge.seam,
            edge.sharp,
        )
        for edge in source.edges
    )
    assert tuple(loop.edge_id for loop in repaired.loops if loop.id in (LoopId(0), LoopId(3), LoopId(6))) == (
        EdgeId(0),
        EdgeId(7),
        EdgeId(8),
    )
    assert all(
        len(face_ids) <= 2
        for face_ids in build_edge_to_faces(repaired).values()
    )


def test_texturing_cut_resolves_stale_local_id_through_source_edge_lineage():
    source = build_square_snapshot()

    resolved = _resolve_geometry_cut_edge_ids(
        source,
        ((EdgeId(999), SourceEdgeId("Cube", 2)),),
        label="Regression",
    )

    assert resolved == (EdgeId(2),)


def test_texturing_cut_rejects_unknown_generated_edge_without_lineage():
    source = build_square_snapshot()

    with pytest.raises(A1TexturingLayoutError, match="has no SourceEdgeId"):
        _resolve_geometry_cut_edge_ids(
            source,
            ((EdgeId(999), None),),
            label="Regression",
        )
