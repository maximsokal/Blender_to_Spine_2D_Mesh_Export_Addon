"""Production routing and post-assembly finalization contracts for Spine 3.8."""

from __future__ import annotations

from Blender_to_Spine2D_Mesh_Exporter.application.a1_document_assembly import (
    A1DocumentAssemblyResult,
    A1DocumentAssemblySettings,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.a1_document_preparation import (
    finalize_a1_document_assembly_for_target,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.legacy_attachment_builder import (
    LegacyAttachmentVertex,
    LegacyMeshAttachmentRequest,
    build_legacy_mesh_document,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.legacy_rig_contracts import (
    LegacyRigBuildRequest,
    LegacyZGroup,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.model import MeshAttachment
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.rig_builder import build_rig
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.rig_profiles import (
    A1RigProfile,
    A1RigSetupPoseMode,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.version_target import SpineJsonTarget
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.weighted_vertices import (
    decode_weighted_vertices,
)


def _rig_request() -> LegacyRigBuildRequest:
    return LegacyRigBuildRequest(
        prefix="Cone",
        texture_width=256,
        texture_height=256,
        z_groups=(
            LegacyZGroup(z_value=0.0, height_real_pixels=0.0),
            LegacyZGroup(z_value=1.0, height_real_pixels=128.0),
        ),
        main_position_pixels=(0.0, 0.0),
        setup_pose_mode=A1RigSetupPoseMode.NORMALIZED_SINGLE,
    )


def _assembly(profile: A1RigProfile) -> A1DocumentAssemblyResult:
    rig = build_rig(
        _rig_request(),
        profile,
        spine_target=SpineJsonTarget.SPINE_3_8,
    )
    request = LegacyMeshAttachmentRequest(
        slot_name="Cone_Segment_0",
        attachment_name="Cone_Segment_0",
        vertex_prefix="Cone_Segment_0",
        image_path="images/Cone_Baked",
        width=256,
        height=256,
        vertices=(
            LegacyAttachmentVertex(
                index=0,
                uv=(0.0, 0.0),
                bone_position_pixels=(-32.0, -32.0),
                z_group_index=rig.info.z_groups[0].index,
            ),
            LegacyAttachmentVertex(
                index=1,
                uv=(1.0, 0.0),
                bone_position_pixels=(32.0, -32.0),
                z_group_index=rig.info.z_groups[0].index,
            ),
            LegacyAttachmentVertex(
                index=2,
                uv=(0.5, 1.0),
                bone_position_pixels=(0.0, 32.0),
                z_group_index=rig.info.z_groups[0].index,
            ),
        ),
        triangles=(0, 1, 2),
        hull=3,
    )
    document_build = build_legacy_mesh_document(
        rig,
        (request,),
        skeleton_metadata={
            "spine": SpineJsonTarget.SPINE_3_8.exact_version,
            "width": 256,
            "height": 256,
        },
    )
    return A1DocumentAssemblyResult(
        settings=A1DocumentAssemblySettings(
            prefix="Cone",
            uv_layer_name="SpineBakeUV",
            image_path="images/Cone_Baked",
            attachment_width=256,
            attachment_height=256,
            center_x=0.0,
            center_y=0.0,
        ),
        rig=rig,
        z_groups=object(),
        projections=(),
        document_build=document_build,
    )


def test_spine38_build_rig_routes_both_profiles_without_fallback() -> None:
    three_axis = build_rig(
        _rig_request(),
        A1RigProfile.THREE_AXIS_ROTATION,
        spine_target="3.8.99",
    )
    two_axis = build_rig(
        _rig_request(),
        A1RigProfile.TWO_AXIS_ROTATION_SCALE,
        spine_target=SpineJsonTarget.SPINE_3_8,
    )

    assert three_axis.profile.profile_id == A1RigProfile.THREE_AXIS_ROTATION.value
    assert two_axis.profile.profile_id == A1RigProfile.TWO_AXIS_ROTATION_SCALE.value
    assert three_axis.info.control_bone_names[2] == "Cone_rotation_Z"
    assert two_axis.info.control_bone_names[2] == "Cone_scale"


def test_spine38_three_axis_keeps_canonical_assembled_topology() -> None:
    assembly = _assembly(A1RigProfile.THREE_AXIS_ROTATION)

    finalized = finalize_a1_document_assembly_for_target(
        assembly,
        spine_target=SpineJsonTarget.SPINE_3_8,
        prefix="Cone",
    )

    assert finalized is assembly
    assert finalized.document is assembly.document
    assert not any(
        bone.name.endswith("_spine41_bridge") for bone in finalized.document.bones
    )


def test_spine38_two_axis_reuses_bridge_and_remaps_weighted_indices() -> None:
    assembly = _assembly(A1RigProfile.TWO_AXIS_ROTATION_SCALE)
    source_component = assembly.document_build.components[0]
    source_vertices = decode_weighted_vertices(
        source_component.attachment.vertices,
        expected_vertex_count=3,
    )

    finalized = finalize_a1_document_assembly_for_target(
        assembly,
        spine_target="3.8.99",
        prefix="Cone",
    )

    assert finalized is not assembly
    assert finalized.rig is assembly.rig
    assert len(finalized.document.bones) == (
        len(assembly.document.bones) + len(assembly.rig.info.z_groups)
    )
    final_component = finalized.document_build.components[0]
    assert final_component.vertex_bone_start_index == (
        source_component.vertex_bone_start_index + len(assembly.rig.info.z_groups)
    )
    assert isinstance(final_component.attachment, MeshAttachment)

    final_vertices = decode_weighted_vertices(
        final_component.attachment.vertices,
        expected_vertex_count=3,
    )
    assert tuple(
        vertex.influences[0].bone_index for vertex in final_vertices
    ) == tuple(
        vertex.influences[0].bone_index + len(assembly.rig.info.z_groups)
        for vertex in source_vertices
    )
    assert any(
        bone.name.endswith("_spine41_bridge") for bone in finalized.document.bones
    )
