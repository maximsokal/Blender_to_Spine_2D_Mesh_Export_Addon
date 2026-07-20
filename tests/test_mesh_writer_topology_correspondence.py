from types import SimpleNamespace

import pytest

from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.mesh_writer import (
    MeshWriteError,
    build_mesh_topology_correspondence,
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
    VertexId,
)


def build_triangle_snapshot() -> MeshSnapshot:
    source = "Triangle"
    vertices = tuple(
        MeshVertex(
            id=VertexId(index),
            source_id=SourceVertexId(source, index),
            position=position,
            normal=(0.0, 0.0, 1.0),
        )
        for index, position in enumerate(
            (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
            )
        )
    )
    edges = tuple(
        MeshEdge(
            id=EdgeId(index),
            source_id=SourceEdgeId(source, index),
            vertex_ids=(VertexId(first), VertexId(second)),
        )
        for index, (first, second) in enumerate(((0, 1), (1, 2), (2, 0)))
    )
    loops = tuple(
        MeshLoop(
            id=LoopId(index),
            source_id=SourceLoopId(source, 0, index),
            vertex_id=VertexId(vertex_index),
            edge_id=EdgeId(edge_index),
        )
        for index, (vertex_index, edge_index) in enumerate(((0, 0), (1, 1), (2, 2)))
    )
    face = MeshFace(
        id=FaceId(0),
        source_id=SourceFaceId(source, 0),
        loop_ids=(LoopId(0), LoopId(1), LoopId(2)),
        material_index=0,
        normal=(0.0, 0.0, 1.0),
    )
    return MeshSnapshot(
        snapshot_id="triangle",
        source_object_id=source,
        object_name=source,
        vertices=vertices,
        edges=edges,
        loops=loops,
        faces=(face,),
    )


def fake_mesh(vertex_order):
    edge_index_by_pair = {
        frozenset((0, 1)): 0,
        frozenset((1, 2)): 1,
        frozenset((2, 0)): 2,
    }
    loops = []
    for index, vertex_index in enumerate(vertex_order):
        next_vertex = vertex_order[(index + 1) % len(vertex_order)]
        loops.append(
            SimpleNamespace(
                vertex_index=vertex_index,
                edge_index=edge_index_by_pair[frozenset((vertex_index, next_vertex))],
            )
        )
    return SimpleNamespace(
        vertices=tuple(SimpleNamespace(index=index) for index in range(3)),
        edges=(
            SimpleNamespace(vertices=(0, 1)),
            SimpleNamespace(vertices=(1, 2)),
            SimpleNamespace(vertices=(2, 0)),
        ),
        loops=tuple(loops),
        polygons=(
            SimpleNamespace(
                vertices=tuple(vertex_order),
                loop_start=0,
                loop_total=3,
            ),
        ),
    )


def test_correspondence_accepts_oriented_cyclic_corner_rotation():
    snapshot = build_triangle_snapshot()
    correspondence = build_mesh_topology_correspondence(
        snapshot,
        fake_mesh((1, 2, 0)),
        stage="test",
    )

    assert correspondence.polygon_index_for(FaceId(0)) == 0
    assert correspondence.mesh_loop_index_for(LoopId(0)) == 2
    assert correspondence.mesh_loop_index_for(LoopId(1)) == 0
    assert correspondence.mesh_loop_index_for(LoopId(2)) == 1


def test_correspondence_rejects_reversed_winding():
    with pytest.raises(MeshWriteError, match="no oriented Blender polygon match"):
        build_mesh_topology_correspondence(
            build_triangle_snapshot(),
            fake_mesh((0, 2, 1)),
            stage="test",
        )


def test_correspondence_rejects_loop_edge_mismatch():
    mesh = fake_mesh((0, 1, 2))
    mesh.loops[0].edge_index = 1

    with pytest.raises(MeshWriteError, match="edge mismatch"):
        build_mesh_topology_correspondence(
            build_triangle_snapshot(),
            mesh,
            stage="test",
        )


def test_correspondence_rejects_polygon_loop_vertex_disagreement():
    mesh = fake_mesh((0, 1, 2))
    mesh.loops[0].vertex_index = 2

    with pytest.raises(MeshWriteError, match="disagrees with its mesh loop order"):
        build_mesh_topology_correspondence(
            build_triangle_snapshot(),
            mesh,
            stage="test",
        )
