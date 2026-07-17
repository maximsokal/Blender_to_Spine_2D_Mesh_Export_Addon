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
    segment_mesh_a1,
)


def _build_snapshot(name, vertex_count, polygons):
    vertices = tuple(
        MeshVertex(
            id=VertexId(index),
            source_id=SourceVertexId(name, index),
            position=(float(index), 0.0, 0.0),
            normal=(0.0, 0.0, 1.0),
        )
        for index in range(vertex_count)
    )
    edge_index_by_vertices = {}
    edges = []
    loops = []
    faces = []
    next_loop_index = 0

    for face_index, polygon in enumerate(polygons):
        loop_ids = []
        for corner_index, vertex_index in enumerate(polygon):
            next_vertex_index = polygon[(corner_index + 1) % len(polygon)]
            edge_key = tuple(sorted((vertex_index, next_vertex_index)))
            edge_index = edge_index_by_vertices.get(edge_key)
            if edge_index is None:
                edge_index = len(edges)
                edge_index_by_vertices[edge_key] = edge_index
                edges.append(
                    MeshEdge(
                        id=EdgeId(edge_index),
                        source_id=SourceEdgeId(name, edge_index),
                        vertex_ids=(VertexId(edge_key[0]), VertexId(edge_key[1])),
                    )
                )
            loop_id = LoopId(next_loop_index)
            loops.append(
                MeshLoop(
                    id=loop_id,
                    source_id=SourceLoopId(name, face_index, corner_index),
                    vertex_id=VertexId(vertex_index),
                    edge_id=EdgeId(edge_index),
                )
            )
            loop_ids.append(loop_id)
            next_loop_index += 1
        faces.append(
            MeshFace(
                id=FaceId(face_index),
                source_id=SourceFaceId(name, face_index),
                loop_ids=tuple(loop_ids),
                material_index=0,
                normal=(0.0, 0.0, 1.0),
            )
        )

    return MeshSnapshot(
        snapshot_id=name,
        source_object_id=name,
        object_name=name,
        vertices=vertices,
        edges=tuple(edges),
        loops=tuple(loops),
        faces=tuple(faces),
    )


def _build_cube():
    return _build_snapshot(
        "closed-cube",
        8,
        (
            (0, 1, 2, 3),
            (4, 7, 6, 5),
            (0, 4, 5, 1),
            (1, 5, 6, 2),
            (2, 6, 7, 3),
            (4, 0, 3, 7),
        ),
    )


def _build_torus(columns=6, rows=5):
    def vertex(column, row):
        return (column % columns) * rows + (row % rows)

    polygons = tuple(
        (
            vertex(column, row),
            vertex(column + 1, row),
            vertex(column + 1, row + 1),
            vertex(column, row + 1),
        )
        for column in range(columns)
        for row in range(rows)
    )
    return _build_snapshot(
        "periodic-torus",
        columns * rows,
        polygons,
    )


def _decompose(snapshot):
    segmentation = segment_mesh_a1(
        snapshot,
        SegmentationSettings(
            split_by_angle=False,
            split_materials=False,
            split_uv_boundaries=False,
        ),
    )
    return decompose_complex_segments(snapshot, segmentation)


def _face_partition(plan):
    return tuple(
        tuple(face_id.index for face_id in region.face_ids)
        for region in plan.regions
    )


def _assert_complete_disks(snapshot, plan):
    covered = [
        face_id
        for region in plan.regions
        for face_id in region.face_ids
    ]
    assert len(covered) == len(set(covered)) == len(snapshot.faces)
    assert set(covered) == {face.id for face in snapshot.faces}
    assert all(is_simple_disk(region.topology) for region in plan.regions)
    assert plan.diagnostics[0].reasons[0] is DecompositionReason.CLOSED_SURFACE


def test_closed_cube_preserves_existing_deterministic_partition():
    snapshot = _build_cube()

    first = _decompose(snapshot)
    second = _decompose(snapshot)

    assert first == second
    assert _face_partition(first) == ((0, 1, 2, 3, 4), (5,))
    _assert_complete_disks(snapshot, first)


def test_periodic_torus_preserves_existing_deterministic_partition():
    snapshot = _build_torus()

    first = _decompose(snapshot)
    second = _decompose(snapshot)

    assert first == second
    assert _face_partition(first) == (
        (0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 15, 16, 17, 18, 20, 21, 22, 23),
        (4, 9, 14, 19, 24),
        (25, 26, 27, 28),
        (29,),
    )
    _assert_complete_disks(snapshot, first)
