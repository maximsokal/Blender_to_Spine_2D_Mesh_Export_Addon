from math import cos, radians, sin

from Blender_to_Spine2D_Mesh_Exporter.domain.geometry import (
    DecompositionReason,
    EdgeId,
    FaceId,
    LoopId,
    MeshEdge,
    MeshFace,
    MeshLoop,
    MeshSnapshot,
    MeshVertex,
    SegmentationSettings,
    SourceEdgeId,
    SourceFaceId,
    SourceLoopId,
    SourceVertexId,
    VertexId,
    decompose_complex_segments,
    is_simple_disk,
    materialize_decomposed_snapshots,
    segment_mesh,
    segment_mesh_a1,
)


def _build_snapshot(name, positions, face_vertices, face_normals=None):
    source = name
    vertices = tuple(
        MeshVertex(
            id=VertexId(index),
            source_id=SourceVertexId(source, index),
            position=tuple(float(component) for component in position),
            normal=(0.0, 0.0, 1.0),
        )
        for index, position in enumerate(positions)
    )

    edge_index_by_vertices = {}
    edge_vertices = []
    loops = []
    faces = []
    next_loop_index = 0
    resolved_normals = face_normals or [(0.0, 0.0, 1.0)] * len(face_vertices)

    for face_index, polygon in enumerate(face_vertices):
        loop_ids = []
        for corner_index, vertex_index in enumerate(polygon):
            next_vertex_index = polygon[(corner_index + 1) % len(polygon)]
            edge_key = tuple(sorted((vertex_index, next_vertex_index)))
            edge_index = edge_index_by_vertices.get(edge_key)
            if edge_index is None:
                edge_index = len(edge_vertices)
                edge_index_by_vertices[edge_key] = edge_index
                edge_vertices.append(edge_key)
            loop_id = LoopId(next_loop_index)
            loops.append(
                MeshLoop(
                    id=loop_id,
                    source_id=SourceLoopId(source, face_index, corner_index),
                    vertex_id=VertexId(vertex_index),
                    edge_id=EdgeId(edge_index),
                    uvs=(),
                )
            )
            loop_ids.append(loop_id)
            next_loop_index += 1
        faces.append(
            MeshFace(
                id=FaceId(face_index),
                source_id=SourceFaceId(source, face_index),
                loop_ids=tuple(loop_ids),
                material_index=0,
                normal=tuple(float(component) for component in resolved_normals[face_index]),
            )
        )

    edges = tuple(
        MeshEdge(
            id=EdgeId(edge_index),
            source_id=SourceEdgeId(source, edge_index),
            vertex_ids=(VertexId(first), VertexId(second)),
        )
        for edge_index, (first, second) in enumerate(edge_vertices)
    )
    return MeshSnapshot(
        snapshot_id=name,
        source_object_id=source,
        object_name=name,
        vertices=vertices,
        edges=edges,
        loops=tuple(loops),
        faces=tuple(faces),
    )


def _normal_at_degrees(angle):
    value = radians(angle)
    return (0.0, sin(value), cos(value))


def build_three_quad_strip():
    positions = tuple(
        (float(x), float(y), 0.0)
        for y in range(2)
        for x in range(4)
    )
    faces = (
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
    )
    return _build_snapshot(
        "normal-strip",
        positions,
        faces,
        (_normal_at_degrees(0), _normal_at_degrees(20), _normal_at_degrees(40)),
    )


def build_quad_ring():
    positions = tuple(
        (float(x), float(y), 0.0)
        for y in range(4)
        for x in range(4)
    )
    faces = []
    for y in range(3):
        for x in range(3):
            if (x, y) == (1, 1):
                continue
            bottom_left = y * 4 + x
            faces.append(
                (
                    bottom_left,
                    bottom_left + 1,
                    bottom_left + 5,
                    bottom_left + 4,
                )
            )
    return _build_snapshot("quad-ring", positions, tuple(faces))


def test_a1_uses_seed_normal_instead_of_pairwise_normal_drift():
    snapshot = build_three_quad_strip()
    settings = SegmentationSettings(
        angle_limit_degrees=30.0,
        split_uv_boundaries=False,
    )

    pairwise = segment_mesh(snapshot, settings)
    a1 = segment_mesh_a1(snapshot, settings)

    assert tuple(segment.face_ids for segment in pairwise.segments) == (
        (FaceId(0), FaceId(1), FaceId(2)),
    )
    assert tuple(segment.face_ids for segment in a1.segments) == (
        (FaceId(0), FaceId(1)),
        (FaceId(2),),
    )
    assert [face_id for segment in a1.segments for face_id in segment.face_ids] == [
        FaceId(0),
        FaceId(1),
        FaceId(2),
    ]
    assert any(
        boundary.reasons[0].value == "ANGLE"
        for boundary in a1.boundary_edges
        if set(boundary.linked_face_ids) == {FaceId(1), FaceId(2)}
    )


def test_a1_segmentation_is_repeatable_and_disjoint():
    snapshot = build_three_quad_strip()
    settings = SegmentationSettings(
        angle_limit_degrees=30.0,
        split_uv_boundaries=False,
    )
    first = segment_mesh_a1(snapshot, settings)
    second = segment_mesh_a1(snapshot, settings)
    assert first == second
    covered = [face_id for segment in first.segments for face_id in segment.face_ids]
    assert len(covered) == len(set(covered)) == len(snapshot.faces)


def test_ring_is_decomposed_into_complete_manifold_disks():
    snapshot = build_quad_ring()
    segmentation = segment_mesh_a1(
        snapshot,
        SegmentationSettings(split_uv_boundaries=False),
    )
    assert len(segmentation.segments) == 1
    assert segmentation.segments[0].topology.boundary_component_count == 2
    assert segmentation.segments[0].topology.euler_characteristic == 0

    plan = decompose_complex_segments(snapshot, segmentation)

    assert len(plan.regions) > 1
    assert all(is_simple_disk(region.topology) for region in plan.regions)
    covered = [face_id for region in plan.regions for face_id in region.face_ids]
    assert len(covered) == len(set(covered)) == len(snapshot.faces)
    assert set(covered) == {face.id for face in snapshot.faces}
    assert len(plan.cuts) >= 1
    assert plan.diagnostics[0].reasons == (
        DecompositionReason.MULTIPLE_BOUNDARIES,
        DecompositionReason.NON_DISK_EULER,
    )


def test_ring_decomposition_is_repeatable_and_materializable():
    snapshot = build_quad_ring()
    segmentation = segment_mesh_a1(
        snapshot,
        SegmentationSettings(split_uv_boundaries=False),
    )
    first = decompose_complex_segments(snapshot, segmentation)
    second = decompose_complex_segments(snapshot, segmentation)
    assert first == second

    region_snapshots = materialize_decomposed_snapshots(snapshot, first)
    assert len(region_snapshots) == len(first.regions)
    assert tuple(item.object_name for item in region_snapshots) == tuple(
        f"quad-ring_Region_{index:03d}" for index in range(len(first.regions))
    )
    assert all(item.faces for item in region_snapshots)
