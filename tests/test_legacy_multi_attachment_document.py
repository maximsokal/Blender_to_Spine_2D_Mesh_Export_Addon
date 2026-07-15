import pytest

from Blender_to_Spine2D_Mesh_Exporter.domain.spine import (
    LegacyAttachmentVertex,
    LegacyMeshAttachmentBuildError,
    LegacyMeshAttachmentRequest,
    LegacyRigBuildRequest,
    LegacyZGroup,
    SpineSerializer,
    SpineValidator,
    build_legacy_mesh_document,
    build_legacy_rig,
    decode_weighted_vertices,
)


def make_rig():
    return build_legacy_rig(
        LegacyRigBuildRequest(
            prefix="Object",
            texture_width=100,
            texture_height=100,
            z_groups=(LegacyZGroup(0.0, height_real_pixels=0.0),),
        )
    )


def make_request(segment_index, *, skin_name="default"):
    name = f"Object_Segment_{segment_index}"
    offset = float(segment_index * 100)
    return LegacyMeshAttachmentRequest(
        slot_name=name,
        attachment_name=name,
        vertex_prefix=name,
        image_path="images/Object_Baked",
        width=100,
        height=100,
        vertices=(
            LegacyAttachmentVertex(0, (0.0, 0.0), (offset, 0.0), 1),
            LegacyAttachmentVertex(1, (1.0, 0.0), (offset + 50.0, 0.0), 1),
            LegacyAttachmentVertex(2, (0.0, 1.0), (offset, 50.0), 1),
        ),
        triangles=(0, 1, 2),
        hull=3,
        edges=(0, 1, 1, 2, 2, 0),
        skin_name=skin_name,
    )


def influence_indices(component):
    decoded = decode_weighted_vertices(
        component.attachment.vertices,
        expected_vertex_count=len(component.request.vertices),
    )
    return tuple(vertex.influences[0].bone_index for vertex in decoded)


def test_multiple_components_receive_non_overlapping_final_bone_ranges():
    rig = make_rig()
    result = build_legacy_mesh_document(
        rig,
        (make_request(0), make_request(1)),
    )

    first, second = result.components
    base = len(rig.bones)
    assert first.vertex_bone_start_index == base
    assert second.vertex_bone_start_index == base + 3
    assert influence_indices(first) == (base, base + 1, base + 2)
    assert influence_indices(second) == (base + 3, base + 4, base + 5)
    assert len(result.document.bones) == base + 6
    assert SpineValidator().validate(result.document) == ()


def test_slots_and_skin_attachments_preserve_request_order():
    result = build_legacy_mesh_document(
        make_rig(),
        (make_request(0), make_request(1)),
    )

    assert tuple(slot.name for slot in result.document.slots) == (
        "Object_Segment_0",
        "Object_Segment_1",
    )
    assert tuple(result.skins[0].attachments) == (
        "Object_Segment_0",
        "Object_Segment_1",
    )
    assert tuple(component.request for component in result.components) == result.requests


def test_serializer_contains_both_attachments_without_json_merge():
    result = build_legacy_mesh_document(
        make_rig(),
        (make_request(0), make_request(1)),
    )
    data = SpineSerializer().to_dict(result.document)

    assert len(data["slots"]) == 2
    attachments = data["skins"][0]["attachments"]
    assert set(attachments) == {"Object_Segment_0", "Object_Segment_1"}
    assert attachments["Object_Segment_0"]["Object_Segment_0"]["type"] == "mesh"
    assert attachments["Object_Segment_1"]["Object_Segment_1"]["type"] == "mesh"


def test_multiple_skin_names_are_grouped_without_reordering_first_occurrence():
    result = build_legacy_mesh_document(
        make_rig(),
        (
            make_request(0, skin_name="default"),
            make_request(1, skin_name="alternate"),
        ),
    )

    assert tuple(skin.name for skin in result.skins) == ("default", "alternate")
    assert tuple(result.skins[0].attachments) == ("Object_Segment_0",)
    assert tuple(result.skins[1].attachments) == ("Object_Segment_1",)


def test_duplicate_slot_or_vertex_prefix_is_rejected_before_composition():
    first = make_request(0)
    duplicate_slot = LegacyMeshAttachmentRequest(
        slot_name=first.slot_name,
        attachment_name="OtherAttachment",
        vertex_prefix="OtherPrefix",
        image_path=first.image_path,
        width=first.width,
        height=first.height,
        vertices=first.vertices,
        triangles=first.triangles,
        hull=first.hull,
        edges=first.edges,
    )
    with pytest.raises(LegacyMeshAttachmentBuildError, match="Duplicate slot_name"):
        build_legacy_mesh_document(make_rig(), (first, duplicate_slot))

    duplicate_prefix = LegacyMeshAttachmentRequest(
        slot_name="OtherSlot",
        attachment_name="OtherAttachment",
        vertex_prefix=first.vertex_prefix,
        image_path=first.image_path,
        width=first.width,
        height=first.height,
        vertices=first.vertices,
        triangles=first.triangles,
        hull=first.hull,
        edges=first.edges,
    )
    with pytest.raises(LegacyMeshAttachmentBuildError, match="Duplicate vertex_prefix"):
        build_legacy_mesh_document(make_rig(), (first, duplicate_prefix))


def test_default_skeleton_bounds_use_largest_attachment_dimensions():
    first = make_request(0)
    second_base = make_request(1)
    second = LegacyMeshAttachmentRequest(
        slot_name=second_base.slot_name,
        attachment_name=second_base.attachment_name,
        vertex_prefix=second_base.vertex_prefix,
        image_path=second_base.image_path,
        width=240,
        height=180,
        vertices=second_base.vertices,
        triangles=second_base.triangles,
        hull=second_base.hull,
        edges=second_base.edges,
    )
    result = build_legacy_mesh_document(make_rig(), (first, second))

    assert result.document.skeleton["width"] == 240.0
    assert result.document.skeleton["height"] == 180.0
