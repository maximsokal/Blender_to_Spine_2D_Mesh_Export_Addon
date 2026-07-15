from dataclasses import replace

import pytest

from Blender_to_Spine2D_Mesh_Exporter.domain.geometry import (
    ConflictingSourceLoopUVError,
    EdgeId,
    FaceId,
    LoopId,
    LoopUV,
    MeshEdge,
    MeshFace,
    MeshLoop,
    MeshSnapshot,
    MeshSnapshotValidator,
    MeshValidationError,
    MeshVertex,
    MissingSourceLoopError,
    SourceEdgeId,
    SourceFaceId,
    SourceLoopId,
    SourceVertexId,
    VertexId,
    build_mesh_fingerprint,
    build_uv_correspondence,
    extract_face_subset,
    transfer_uv_by_source_loop,
)


def build_square_snapshot(snapshot_id="square", layer="UVMap"):
    source = "Cube"
    vertices = (
        MeshVertex(
            VertexId(0),
            SourceVertexId(source, 0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        MeshVertex(
            VertexId(1),
            SourceVertexId(source, 1),
            (1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        MeshVertex(
            VertexId(2),
            SourceVertexId(source, 2),
            (1.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        MeshVertex(
            VertexId(3),
            SourceVertexId(source, 3),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
    )
    edges = (
        MeshEdge(EdgeId(0), SourceEdgeId(source, 0), (VertexId(0), VertexId(1))),
        MeshEdge(EdgeId(1), SourceEdgeId(source, 1), (VertexId(1), VertexId(2))),
        MeshEdge(EdgeId(2), SourceEdgeId(source, 2), (VertexId(2), VertexId(0))),
        MeshEdge(EdgeId(3), SourceEdgeId(source, 3), (VertexId(2), VertexId(3))),
        MeshEdge(EdgeId(4), SourceEdgeId(source, 4), (VertexId(3), VertexId(0))),
    )
    uv = {
        (0, 0): (0.0, 0.0),
        (0, 1): (1.0, 0.0),
        (0, 2): (1.0, 1.0),
        (1, 0): (0.0, 0.0),
        (1, 1): (1.0, 1.0),
        (1, 2): (0.0, 1.0),
    }
    loops = (
        MeshLoop(
            LoopId(0),
            SourceLoopId(source, 0, 0),
            VertexId(0),
            EdgeId(0),
            (LoopUV(layer, uv[(0, 0)]),),
        ),
        MeshLoop(
            LoopId(1),
            SourceLoopId(source, 0, 1),
            VertexId(1),
            EdgeId(1),
            (LoopUV(layer, uv[(0, 1)]),),
        ),
        MeshLoop(
            LoopId(2),
            SourceLoopId(source, 0, 2),
            VertexId(2),
            EdgeId(2),
            (LoopUV(layer, uv[(0, 2)]),),
        ),
        MeshLoop(
            LoopId(3),
            SourceLoopId(source, 1, 0),
            VertexId(0),
            EdgeId(2),
            (LoopUV(layer, uv[(1, 0)]),),
        ),
        MeshLoop(
            LoopId(4),
            SourceLoopId(source, 1, 1),
            VertexId(2),
            EdgeId(3),
            (LoopUV(layer, uv[(1, 1)]),),
        ),
        MeshLoop(
            LoopId(5),
            SourceLoopId(source, 1, 2),
            VertexId(3),
            EdgeId(4),
            (LoopUV(layer, uv[(1, 2)]),),
        ),
    )
    faces = (
        MeshFace(
            FaceId(0),
            SourceFaceId(source, 0),
            (LoopId(0), LoopId(1), LoopId(2)),
            0,
            (0.0, 0.0, 1.0),
        ),
        MeshFace(
            FaceId(1),
            SourceFaceId(source, 1),
            (LoopId(3), LoopId(4), LoopId(5)),
            0,
            (0.0, 0.0, 1.0),
        ),
    )
    return MeshSnapshot(
        snapshot_id=snapshot_id,
        source_object_id=source,
        object_name="Cube",
        vertices=vertices,
        edges=edges,
        loops=loops,
        faces=faces,
        uv_layer_names=(layer,),
        active_uv_layer=layer,
    )


def test_snapshot_validator_accepts_valid_topology():
    snapshot = build_square_snapshot()
    assert MeshSnapshotValidator().validate(snapshot) == ()


def test_snapshot_validator_rejects_wrong_loop_edge_connectivity():
    snapshot = build_square_snapshot()
    broken_loop = replace(snapshot.loops[0], edge_id=EdgeId(1))
    broken = replace(snapshot, loops=(broken_loop,) + snapshot.loops[1:])

    issues = MeshSnapshotValidator().validate(broken)
    assert any(issue.code == "FACE_EDGE_CONNECTIVITY" for issue in issues)
    with pytest.raises(MeshValidationError):
        MeshSnapshotValidator().validate_or_raise(broken)


def test_extract_face_subset_reindexes_locally_and_preserves_source_ids():
    snapshot = build_square_snapshot()
    segment = extract_face_subset(
        snapshot,
        (FaceId(1),),
        snapshot_id="segment-1",
        object_name="Cube_Segment_001",
    )

    assert tuple(vertex.id.index for vertex in segment.vertices) == (0, 1, 2)
    assert tuple(edge.id.index for edge in segment.edges) == (0, 1, 2)
    assert tuple(loop.id.index for loop in segment.loops) == (0, 1, 2)
    assert segment.faces[0].source_id == SourceFaceId("Cube", 1)
    assert tuple(loop.source_id for loop in segment.loops) == (
        SourceLoopId("Cube", 1, 0),
        SourceLoopId("Cube", 1, 1),
        SourceLoopId("Cube", 1, 2),
    )
    MeshSnapshotValidator().validate_or_raise(segment)


def test_uv_transfer_uses_source_loop_ids_not_position_or_local_order():
    source = build_square_snapshot(layer="BakedUV")
    target = extract_face_subset(
        source,
        (FaceId(1),),
        snapshot_id="segment",
    )
    moved_vertices = tuple(
        replace(
            vertex,
            position=(
                vertex.position[0] + 100.0,
                vertex.position[1] - 50.0,
                7.0,
            ),
        )
        for vertex in target.vertices
    )
    target = replace(
        target,
        vertices=moved_vertices,
        loops=tuple(reversed(target.loops)),
    )
    old_to_new = {loop.id: LoopId(index) for index, loop in enumerate(target.loops)}
    target = replace(
        target,
        loops=tuple(
            replace(loop, id=old_to_new[loop.id], uvs=()) for loop in target.loops
        ),
        faces=(
            replace(
                target.faces[0],
                loop_ids=tuple(
                    old_to_new[loop_id] for loop_id in target.faces[0].loop_ids
                ),
            ),
        ),
        uv_layer_names=(),
        active_uv_layer=None,
    )
    MeshSnapshotValidator().validate_or_raise(target)

    updated, report = transfer_uv_by_source_loop(
        source,
        target,
        source_layer_name="BakedUV",
        target_layer_name="TransferredUV",
    )

    source_lookup = build_uv_correspondence(source, "BakedUV").as_dict()
    assert report.complete
    assert report.updated_loop_count == 3
    for loop in updated.loops:
        assert loop.uv("TransferredUV") == source_lookup[loop.source_id]


def test_repeated_source_loop_id_is_allowed_when_uv_matches():
    snapshot = build_square_snapshot()
    repeated = replace(
        snapshot.loops[1],
        id=LoopId(6),
        source_id=snapshot.loops[0].source_id,
        vertex_id=VertexId(0),
        edge_id=EdgeId(0),
        uvs=snapshot.loops[0].uvs,
    )
    derived = replace(snapshot, loops=snapshot.loops + (repeated,))
    correspondence = build_uv_correspondence(derived, "UVMap")
    assert correspondence.as_dict()[SourceLoopId("Cube", 0, 0)] == (0.0, 0.0)


def test_repeated_source_loop_id_with_conflicting_uv_is_rejected():
    snapshot = build_square_snapshot()
    conflicting = replace(
        snapshot.loops[0],
        id=LoopId(6),
        uvs=(LoopUV("UVMap", (0.5, 0.5)),),
    )
    derived = replace(snapshot, loops=snapshot.loops + (conflicting,))
    with pytest.raises(ConflictingSourceLoopUVError):
        build_uv_correspondence(derived, "UVMap")


def test_missing_correspondence_can_report_or_raise():
    source = build_square_snapshot()
    target = extract_face_subset(source, (FaceId(1),), snapshot_id="segment")
    missing_loop = replace(
        target.loops[0],
        source_id=SourceLoopId("Cube", 1, 99),
    )
    target = replace(target, loops=(missing_loop,) + target.loops[1:])

    with pytest.raises(MissingSourceLoopError):
        transfer_uv_by_source_loop(
            source,
            target,
            source_layer_name="UVMap",
            target_layer_name="Transferred",
        )

    updated, report = transfer_uv_by_source_loop(
        source,
        target,
        source_layer_name="UVMap",
        target_layer_name="Transferred",
        require_complete=False,
    )
    assert not report.complete
    assert report.missing_source_loop_ids == (SourceLoopId("Cube", 1, 99),)
    assert updated.loops[0].uv("Transferred") == updated.loops[0].uv("UVMap")


def test_mesh_fingerprint_ignores_uv_but_tracks_topology_and_lineage():
    snapshot = build_square_snapshot()
    changed_uvs = tuple(
        replace(loop, uvs=(LoopUV("UVMap", (0.25, 0.75)),))
        for loop in snapshot.loops
    )
    changed_uv_snapshot = replace(snapshot, loops=changed_uvs)
    assert build_mesh_fingerprint(snapshot) == build_mesh_fingerprint(
        changed_uv_snapshot
    )

    changed_position = replace(
        snapshot,
        vertices=(
            replace(snapshot.vertices[0], position=(0.1, 0.0, 0.0)),
        )
        + snapshot.vertices[1:],
    )
    assert build_mesh_fingerprint(snapshot) != build_mesh_fingerprint(changed_position)
