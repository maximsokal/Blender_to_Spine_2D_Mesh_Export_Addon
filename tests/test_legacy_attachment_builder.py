import pytest

from Blender_to_Spine2D_Mesh_Exporter.domain.spine import (
    LegacyAttachmentSequence,
    LegacyAttachmentVertex,
    LegacyMeshAttachmentBuildError,
    LegacyMeshAttachmentRequest,
    LegacyRigBuildRequest,
    LegacyZGroup,
    SpineSerializer,
    SpineValidator,
    build_legacy_mesh_attachment,
    build_legacy_rig,
    decode_weighted_vertices,
)


def make_rig():
    return build_legacy_rig(
        LegacyRigBuildRequest(
            prefix="Triangle",
            texture_width=100,
            texture_height=100,
            z_groups=(LegacyZGroup(0.0, height_real_pixels=0.0),),
        )
    )


def make_request(*, sequence=None):
    return LegacyMeshAttachmentRequest(
        slot_name="Triangle_Segment_0",
        attachment_name="Triangle_Segment_0",
        vertex_prefix="Triangle_Segment_0",
        image_path="images/Triangle_Baked",
        width=100,
        height=100,
        vertices=(
            LegacyAttachmentVertex(0, (0.0, 0.0), (-50.0, 50.0), 1),
            LegacyAttachmentVertex(1, (1.0, 0.0), (50.0, 50.0), 1),
            LegacyAttachmentVertex(2, (0.5, 1.0), (0.0, -50.0), 1),
        ),
        triangles=(0, 1, 2),
        hull=3,
        # Rewrite topology uses logical attachment vertex indices internally.
        edges=(0, 1, 1, 2, 2, 0),
        sequence=sequence,
    )


def test_vertex_bones_use_explicit_z_group_and_pixel_positions():
    rig = make_rig()
    result = build_legacy_mesh_attachment(rig, make_request())

    assert tuple(bone.name for bone in result.vertex_bones) == (
        "Triangle_Segment_0_vertex_0",
        "Triangle_Segment_0_vertex_1",
        "Triangle_Segment_0_vertex_2",
    )
    assert tuple(bone.parent for bone in result.vertex_bones) == (
        "Triangle_1",
        "Triangle_1",
        "Triangle_1",
    )
    assert tuple((bone.x, bone.y) for bone in result.vertex_bones) == (
        (-50.0, 50.0),
        (50.0, 50.0),
        (0.0, -50.0),
    )


def test_weighted_stream_references_final_appended_bone_indices():
    rig = make_rig()
    result = build_legacy_mesh_attachment(rig, make_request())
    decoded = decode_weighted_vertices(
        result.attachment.vertices,
        expected_vertex_count=3,
    )

    first_index = len(rig.bones)
    assert tuple(
        vertex.influences[0].bone_index for vertex in decoded
    ) == (first_index, first_index + 1, first_index + 2)
    assert all(
        vertex.influences[0].x == 0.0
        and vertex.influences[0].y == 0.0
        and vertex.influences[0].weight == 1.0
        for vertex in decoded
    )


def test_attachment_slot_skin_and_document_are_validated_together():
    result = build_legacy_mesh_attachment(make_rig(), make_request())

    assert result.slot.name == "Triangle_Segment_0"
    assert result.slot.bone == "Triangle"
    assert result.slot.attachment == "Triangle_Segment_0"
    assert result.skin.name == "default"
    assert (
        result.skin.attachments["Triangle_Segment_0"]["Triangle_Segment_0"]
        is result.attachment
    )
    assert result.attachment.path == "images/Triangle_Baked"
    assert result.attachment.uvs == (0.0, 0.0, 1.0, 0.0, 0.5, 1.0)
    assert result.attachment.triangles == (0, 1, 2)
    assert result.attachment.edges == (0, 1, 1, 2, 2, 0)
    assert result.document.animations == {"animation": {}}
    assert SpineValidator().validate(result.document) == ()


def test_serializer_emits_spine_42_mesh_edge_coordinate_offsets():
    result = build_legacy_mesh_attachment(make_rig(), make_request())
    data = SpineSerializer().to_dict(result.document)
    attachment = data["skins"][0]["attachments"]["Triangle_Segment_0"][
        "Triangle_Segment_0"
    ]

    assert attachment == {
        "type": "mesh",
        "uvs": [0.0, 0.0, 1.0, 0.0, 0.5, 1.0],
        "triangles": [0, 1, 2],
        "vertices": list(result.attachment.vertices),
        "hull": 3,
        "path": "images/Triangle_Baked",
        # Spine JSON addresses interleaved x/y positions: vertex N -> N * 2.
        "edges": [0, 2, 2, 4, 4, 0],
        "width": 100.0,
        "height": 100.0,
    }
    assert all(value % 2 == 0 for value in attachment["edges"])


def test_sequence_path_and_mapping_match_spine_42_contract():
    sequence = LegacyAttachmentSequence(count=3, start=7, digits=4)
    result = build_legacy_mesh_attachment(
        make_rig(),
        make_request(sequence=sequence),
    )

    assert result.attachment.path == "images/Triangle_Baked_"
    assert result.attachment.sequence == {
        "count": 3,
        "start": 7,
        "digits": 4,
        "setup": 1,
    }


def test_single_frame_sequence_uses_zero_setup():
    sequence = LegacyAttachmentSequence(count=1, start=5)
    assert sequence.to_spine_mapping()["setup"] == 0


def test_unknown_z_group_index_fails_before_weighted_stream_is_exposed():
    request = make_request()
    broken_vertices = (
        request.vertices[0],
        request.vertices[1],
        LegacyAttachmentVertex(2, (0.5, 1.0), (0.0, -50.0), 99),
    )
    broken = LegacyMeshAttachmentRequest(
        slot_name=request.slot_name,
        attachment_name=request.attachment_name,
        vertex_prefix=request.vertex_prefix,
        image_path=request.image_path,
        width=request.width,
        height=request.height,
        vertices=broken_vertices,
        triangles=request.triangles,
        hull=request.hull,
        edges=request.edges,
    )

    with pytest.raises(LegacyMeshAttachmentBuildError):
        build_legacy_mesh_attachment(make_rig(), broken)


def test_triangle_and_internal_edge_indices_are_validated_at_request_boundary():
    base = make_request()
    with pytest.raises(ValueError):
        LegacyMeshAttachmentRequest(
            slot_name=base.slot_name,
            attachment_name=base.attachment_name,
            vertex_prefix=base.vertex_prefix,
            image_path=base.image_path,
            width=base.width,
            height=base.height,
            vertices=base.vertices,
            triangles=(0, 1, 9),
            hull=3,
        )
    with pytest.raises(ValueError):
        LegacyMeshAttachmentRequest(
            slot_name=base.slot_name,
            attachment_name=base.attachment_name,
            vertex_prefix=base.vertex_prefix,
            image_path=base.image_path,
            width=base.width,
            height=base.height,
            vertices=base.vertices,
            triangles=base.triangles,
            hull=3,
            edges=(0, 1, 2),
        )
