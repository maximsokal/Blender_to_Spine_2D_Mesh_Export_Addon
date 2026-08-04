"""Lineage regressions for Depth parallax union and material subsets."""

from __future__ import annotations

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
from Blender_to_Spine2D_Mesh_Exporter.domain.geometry.depth_parallax import (
    _FaceRecord,
    _snapshot_from_records,
    _subset_material,
)


_OBJECT_ID = "ParallaxLineageObject"
_UV_LAYER = "SpineBakeUV"


def _source_snapshot() -> MeshSnapshot:
    positions = (
        (0.0, 0.0, -5.0),
        (1.0, 0.0, -5.0),
        (0.0, 1.0, -5.0),
        (-1.0, 1.0, -5.0),
        # Deliberately coincident with vertex 0 but topologically unrelated.
        (0.0, 0.0, -5.0),
    )
    source_faces = (
        (2, (0, 1, 2)),
        (3, (4, 2, 3)),
    )
    edge_pairs = tuple(
        sorted(
            {
                tuple(sorted((face[index], face[(index + 1) % 3])))
                for _source_face_index, face in source_faces
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
            source_id=SourceVertexId(_OBJECT_ID, index),
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
    for local_face_index, (source_face_index, face) in enumerate(source_faces):
        loop_ids: list[LoopId] = []
        for corner_index, vertex_index in enumerate(face):
            following = face[(corner_index + 1) % 3]
            loop_id = LoopId(len(loops))
            loops.append(
                MeshLoop(
                    id=loop_id,
                    source_id=SourceLoopId(
                        _OBJECT_ID,
                        source_face_index,
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
                id=FaceId(local_face_index),
                source_id=SourceFaceId(_OBJECT_ID, source_face_index),
                loop_ids=tuple(loop_ids),
                material_index=0,
                normal=(0.0, 0.0, 1.0),
            )
        )

    snapshot = MeshSnapshot(
        snapshot_id="ParallaxLineageSource",
        source_object_id=_OBJECT_ID,
        object_name="Parallax Lineage Source",
        vertices=vertices,
        edges=edges,
        loops=tuple(loops),
        faces=tuple(faces),
    )
    MeshSnapshotValidator().validate_or_raise(snapshot)
    return snapshot


def _records(source: MeshSnapshot) -> tuple[_FaceRecord, ...]:
    vertices = source.vertex_by_id()
    return (
        _FaceRecord(
            material_index=0,
            source_face_index=2,
            source_vertex_ids=(
                vertices[VertexId(0)].source_id,
                vertices[VertexId(1)].source_id,
                vertices[VertexId(2)].source_id,
            ),
            positions=(
                vertices[VertexId(0)].position,
                vertices[VertexId(1)].position,
                vertices[VertexId(2)].position,
            ),
            uvs=((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),
        ),
        _FaceRecord(
            material_index=1,
            source_face_index=3,
            source_vertex_ids=(
                vertices[VertexId(4)].source_id,
                vertices[VertexId(2)].source_id,
                vertices[VertexId(3)].source_id,
            ),
            positions=(
                vertices[VertexId(4)].position,
                vertices[VertexId(2)].position,
                vertices[VertexId(3)].position,
            ),
            uvs=((0.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
        ),
    )


def _assert_face_loop_lineage(snapshot: MeshSnapshot) -> None:
    loops = snapshot.loop_by_id()
    for face in snapshot.faces:
        for loop_id in face.loop_ids:
            assert (
                loops[loop_id].source_id.face_index
                == face.source_id.face_index
            )


def test_union_keeps_coincident_disconnected_vertices_and_coherent_face_lineage() -> None:
    source = _source_snapshot()
    union = _snapshot_from_records(
        source,
        _records(source),
        uv_layer_name=_UV_LAYER,
        snapshot_suffix="union",
        preserve_source_vertex_ids=False,
    )

    MeshSnapshotValidator().validate_or_raise(union)
    assert len(union.vertices) == 5
    assert len({vertex.source_id for vertex in union.vertices}) == 5
    coincident = tuple(
        vertex for vertex in union.vertices if vertex.position == (0.0, 0.0, -5.0)
    )
    assert len(coincident) == 2
    assert coincident[0].source_id != coincident[1].source_id
    _assert_face_loop_lineage(union)


def test_material_subset_preserves_exact_union_vertex_lineage() -> None:
    source = _source_snapshot()
    union = _snapshot_from_records(
        source,
        _records(source),
        uv_layer_name=_UV_LAYER,
        snapshot_suffix="union",
        preserve_source_vertex_ids=False,
    )
    reserve = _subset_material(
        union,
        1,
        uv_layer_name=_UV_LAYER,
        suffix="reserve",
    )

    MeshSnapshotValidator().validate_or_raise(reserve)
    union_by_source = {vertex.source_id: vertex for vertex in union.vertices}
    assert len(reserve.vertices) == 3
    for vertex in reserve.vertices:
        assert vertex.source_id in union_by_source
        assert vertex.position == union_by_source[vertex.source_id].position
    _assert_face_loop_lineage(reserve)
