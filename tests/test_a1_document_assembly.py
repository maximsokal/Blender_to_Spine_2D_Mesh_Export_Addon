import pytest

from Blender_to_Spine2D_Mesh_Exporter.application import (
    A1DocumentAssemblyError,
    A1DocumentAssemblySettings,
    A1ZGroupHeightOverride,
    assemble_a1_document,
    build_a1_z_group_assignment,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.geometry import FaceId, extract_face_subset
from Blender_to_Spine2D_Mesh_Exporter.domain.spine import (
    LegacyAttachmentSequence,
    LegacyRigBuildRequest,
    SpineSerializer,
    SpineValidator,
    build_legacy_rig,
    decode_weighted_vertices,
)

from test_geometry_domain import build_square_snapshot


def build_inputs():
    source = build_square_snapshot()
    z_plan = build_a1_z_group_assignment(
        source,
        height_overrides=(A1ZGroupHeightOverride(0.0, 0.0),),
    )
    rig = build_legacy_rig(
        LegacyRigBuildRequest(
            prefix="Cube",
            texture_width=100,
            texture_height=100,
            z_groups=z_plan.groups,
        )
    )
    first = extract_face_subset(
        source,
        (FaceId(0),),
        snapshot_id="Cube:region-0",
        object_name="Cube_Region_0",
    )
    second = extract_face_subset(
        source,
        (FaceId(1),),
        snapshot_id="Cube:region-1",
        object_name="Cube_Region_1",
    )
    settings = A1DocumentAssemblySettings(
        prefix="Cube",
        uv_layer_name="UVMap",
        image_path="images/Cube_Baked",
        attachment_width=100.0,
        attachment_height=100.0,
        center_x=0.5,
        center_y=0.5,
    )
    return source, z_plan, rig, (first, second), settings


def test_two_regions_assemble_into_one_valid_spine_document():
    _, z_plan, rig, regions, settings = build_inputs()
    result = assemble_a1_document(rig, z_plan, regions, settings)

    assert tuple(slot.name for slot in result.document.slots) == (
        "Cube_Segment_0",
        "Cube_Segment_1",
    )
    assert len(result.projections) == 2
    assert len(result.document_build.components) == 2
    assert SpineValidator().validate(result.document) == ()


def test_document_assembly_shares_weighted_bones_for_region_boundary_vertices():
    _, z_plan, rig, regions, settings = build_inputs()
    result = assemble_a1_document(rig, z_plan, regions, settings)
    first, second = result.document_build.components
    base = len(rig.bones)

    first_stream = decode_weighted_vertices(
        first.attachment.vertices,
        expected_vertex_count=len(first.request.vertices),
    )
    second_stream = decode_weighted_vertices(
        second.attachment.vertices,
        expected_vertex_count=len(second.request.vertices),
    )

    # Face 0 owns source vertices 0, 1, 2. Face 1 owns 0, 2, 3, so its
    # first two attachment vertices must reuse the canonical bones from face 0.
    assert tuple(item.influences[0].bone_index for item in first_stream) == (
        base,
        base + 1,
        base + 2,
    )
    assert tuple(item.influences[0].bone_index for item in second_stream) == (
        base,
        base + 2,
        base + 3,
    )
    assert len(result.document.bones) == base + 4
    assert second.vertex_bone_start_index == base
    assert tuple(bone.name for bone in second.vertex_bones[:2]) == (
        "Cube_Segment_0_vertex_0",
        "Cube_Segment_0_vertex_2",
    )


def test_serialization_contains_both_slots_and_attachments_without_merge():
    _, z_plan, rig, regions, settings = build_inputs()
    result = assemble_a1_document(rig, z_plan, regions, settings)
    data = SpineSerializer().to_dict(result.document)

    assert [slot["name"] for slot in data["slots"]] == [
        "Cube_Segment_0",
        "Cube_Segment_1",
    ]
    attachments = data["skins"][0]["attachments"]
    assert set(attachments) == {"Cube_Segment_0", "Cube_Segment_1"}


def test_sequence_assembly_adds_one_timeline_per_segment_attachment():
    _, z_plan, rig, regions, settings = build_inputs()
    sequence_settings = A1DocumentAssemblySettings(
        prefix=settings.prefix,
        uv_layer_name=settings.uv_layer_name,
        image_path="images/Cube_Baked_",
        attachment_width=settings.attachment_width,
        attachment_height=settings.attachment_height,
        center_x=settings.center_x,
        center_y=settings.center_y,
        sequence=LegacyAttachmentSequence(
            count=3,
            start=0,
            digits=4,
        ),
    )

    result = assemble_a1_document(
        rig,
        z_plan,
        regions,
        sequence_settings,
    )
    data = SpineSerializer().to_dict(result.document)

    attachment_timelines = data["animations"]["animation"]["attachments"]["default"]
    assert set(attachment_timelines) == {
        "Cube_Segment_0",
        "Cube_Segment_1",
    }
    for slot_name in ("Cube_Segment_0", "Cube_Segment_1"):
        assert attachment_timelines[slot_name][slot_name]["sequence"] == [
            {"mode": "loop", "delay": 0.0333},
            {"time": 0.0333, "mode": "loop", "index": 1},
            {"time": 0.0666, "mode": "loop", "index": 2},
        ]


def test_segment_index_base_changes_names_without_changing_geometry():
    _, z_plan, rig, regions, settings = build_inputs()
    shifted = A1DocumentAssemblySettings(
        prefix=settings.prefix,
        uv_layer_name=settings.uv_layer_name,
        image_path=settings.image_path,
        attachment_width=settings.attachment_width,
        attachment_height=settings.attachment_height,
        center_x=settings.center_x,
        center_y=settings.center_y,
        segment_index_base=5,
    )
    result = assemble_a1_document(rig, z_plan, regions, shifted)

    assert tuple(slot.name for slot in result.document.slots) == (
        "Cube_Segment_5",
        "Cube_Segment_6",
    )


def test_prefix_mismatch_is_rejected_before_projection():
    _, z_plan, rig, regions, settings = build_inputs()
    wrong = A1DocumentAssemblySettings(
        prefix="Other",
        uv_layer_name=settings.uv_layer_name,
        image_path=settings.image_path,
        attachment_width=settings.attachment_width,
        attachment_height=settings.attachment_height,
        center_x=settings.center_x,
        center_y=settings.center_y,
    )

    with pytest.raises(A1DocumentAssemblyError, match="does not match rig prefix"):
        assemble_a1_document(rig, z_plan, regions, wrong)


def test_rig_and_z_assignment_plan_must_be_identical():
    source, _, rig, regions, settings = build_inputs()
    different_z_plan = build_a1_z_group_assignment(source)

    with pytest.raises(A1DocumentAssemblyError, match="Z groups do not match"):
        assemble_a1_document(rig, different_z_plan, regions, settings)
