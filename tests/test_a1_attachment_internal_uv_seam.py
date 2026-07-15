from Blender_to_Spine2D_Mesh_Exporter.application import (
    A1AttachmentProjectionSettings,
    A1VertexZBinding,
    project_triangulated_disk_attachment,
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
    MeshSnapshotValidator,
    MeshVertex,
    SourceEdgeId,
    SourceFaceId,
    SourceLoopId,
    SourceVertexId,
    VertexId,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine import (
    LegacyRigBuildRequest,
    LegacyZGroup,
    SpineValidator,
    build_legacy_mesh_attachment,
    build_legacy_rig,
)


def build_center_fan_with_uv_split() -> MeshSnapshot:
    source = "Fan"
    positions = (
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.5, 0.5, 0.0),
    )
    vertices = tuple(
        MeshVertex(
            id=VertexId(index),
            source_id=SourceVertexId(source, index),
            position=position,
            normal=(0.0, 0.0, 1.0),
        )
        for index, position in enumerate(positions)
    )
    edge_vertices = (
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (0, 4),
        (1, 4),
        (2, 4),
        (3, 4),
    )
    edges = tuple(
        MeshEdge(
            id=EdgeId(index),
            source_id=SourceEdgeId(source, index),
            vertex_ids=(VertexId(first), VertexId(second)),
        )
        for index, (first, second) in enumerate(edge_vertices)
    )
    face_vertices = (
        (0, 1, 4),
        (1, 2, 4),
        (2, 3, 4),
        (3, 0, 4),
    )
    face_edges = (
        (0, 5, 4),
        (1, 6, 5),
        (2, 7, 6),
        (3, 4, 7),
    )
    uv_by_vertex = {
        0: (0.0, 0.0),
        1: (1.0, 0.0),
        2: (1.0, 1.0),
        3: (0.0, 1.0),
        4: (0.5, 0.5),
    }

    loops = []
    faces = []
    loop_index = 0
    for face_index, (vertex_indices, edge_indices) in enumerate(
        zip(face_vertices, face_edges)
    ):
        face_loop_ids = []
        for corner_index, (vertex_index, edge_index) in enumerate(
            zip(vertex_indices, edge_indices)
        ):
            loop_id = LoopId(loop_index)
            face_loop_ids.append(loop_id)
            coordinate = uv_by_vertex[vertex_index]
            if face_index == 2 and vertex_index == 4:
                coordinate = (0.6, 0.6)
            loops.append(
                MeshLoop(
                    id=loop_id,
                    source_id=SourceLoopId(source, face_index, corner_index),
                    vertex_id=VertexId(vertex_index),
                    edge_id=EdgeId(edge_index),
                    uvs=(LoopUV("UVMap", coordinate),),
                )
            )
            loop_index += 1
        faces.append(
            MeshFace(
                id=FaceId(face_index),
                source_id=SourceFaceId(source, face_index),
                loop_ids=tuple(face_loop_ids),
                material_index=0,
                normal=(0.0, 0.0, 1.0),
            )
        )

    snapshot = MeshSnapshot(
        snapshot_id="center-fan-split",
        source_object_id=source,
        object_name=source,
        vertices=vertices,
        edges=edges,
        loops=tuple(loops),
        faces=tuple(faces),
        uv_layer_names=("UVMap",),
        active_uv_layer="UVMap",
    )
    MeshSnapshotValidator().validate_or_raise(snapshot)
    return snapshot


def build_rig():
    return build_legacy_rig(
        LegacyRigBuildRequest(
            prefix="Fan",
            texture_width=100,
            texture_height=100,
            z_groups=(LegacyZGroup(0.0, height_real_pixels=0.0),),
        )
    )


def build_settings():
    return A1AttachmentProjectionSettings(
        slot_name="Fan_Segment_0",
        attachment_name="Fan_Segment_0",
        vertex_prefix="Fan_Segment_0",
        image_path="images/Fan_Baked",
        uv_layer_name="UVMap",
        attachment_width=100.0,
        attachment_height=100.0,
        center_x=0.5,
        center_y=0.5,
        z_bindings=tuple(
            A1VertexZBinding(VertexId(index), 1) for index in range(5)
        ),
    )


def test_internal_uv_split_duplicates_vertex_after_physical_hull():
    rig = build_rig()
    projection = project_triangulated_disk_attachment(
        build_center_fan_with_uv_split(),
        rig,
        build_settings(),
    )

    assert projection.hull_vertex_ids == (
        VertexId(0),
        VertexId(1),
        VertexId(2),
        VertexId(3),
    )
    assert projection.ordered_vertex_ids == (
        VertexId(0),
        VertexId(1),
        VertexId(2),
        VertexId(3),
        VertexId(4),
        VertexId(4),
    )
    assert projection.request.hull == 4
    assert projection.attachment_indices_for(VertexId(4)) == (4, 5)
    assert projection.request.triangles == (
        0,
        1,
        4,
        1,
        2,
        4,
        2,
        3,
        5,
        3,
        0,
        4,
    )
    assert tuple(vertex.uv for vertex in projection.request.vertices[-2:]) == (
        (0.5, 0.5),
        (0.6, 0.6),
    )
    assert (
        projection.request.vertices[4].bone_position_pixels
        == projection.request.vertices[5].bone_position_pixels
    )

    attachment = build_legacy_mesh_attachment(rig, projection.request)
    assert len(attachment.vertex_bones) == 6
    assert attachment.attachment.hull == 4
    assert SpineValidator().validate(attachment.document) == ()
