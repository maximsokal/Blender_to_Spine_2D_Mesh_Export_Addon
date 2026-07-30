"""Spine 4.0 post-assembly bridge and weighted-index finalization regressions."""

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
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.spine41_setup_safety import (
    find_spine41_unsafe_world_constraints,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.version_target import SpineJsonTarget
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.weighted_vertices import (
    decode_weighted_vertices,
)


def _assembly() -> A1DocumentAssemblyResult:
    rig = build_rig(
        LegacyRigBuildRequest(
            prefix="Cone",
            texture_width=256,
            texture_height=256,
            z_groups=(
                LegacyZGroup(z_value=0.0, height_real_pixels=0.0),
                LegacyZGroup(z_value=1.0, height_real_pixels=128.0),
            ),
            main_position_pixels=(0.0, 0.0),
            setup_pose_mode=A1RigSetupPoseMode.NORMALIZED_SINGLE,
        ),
        A1RigProfile.TWO_AXIS_ROTATION_SCALE,
        spine_target=SpineJsonTarget.SPINE_4_0,
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
            "spine": "4.0.64",
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


def test_spine40_finalization_reuses_proven_legacy_scale_bridge_topology() -> None:
    assembly = _assembly()
    canonical = assembly.document
    profile = assembly.rig.profile

    finalized = finalize_a1_document_assembly_for_target(
        assembly,
        spine_target=SpineJsonTarget.SPINE_4_0,
        prefix="Cone",
    )

    assert finalized is not assembly
    assert finalized.rig is assembly.rig
    assert assembly.document is canonical
    assert len(finalized.document.bones) == (
        len(canonical.bones) + len(assembly.rig.info.z_groups)
    )

    source_transform = {item.name: item for item in canonical.transform}
    final_transform = {item.name: item for item in finalized.document.transform}
    scale_name = profile.scale_constraint("Cone")
    depth_name = profile.scale_depth_constraint("Cone")

    assert final_transform[scale_name].extras.get("local") is None
    assert final_transform[scale_name].extras["relative"] is True
    assert final_transform[scale_name].bones == tuple(
        profile.scale_rotate_x_bone("Cone")
        if name == profile.rotate_x_bone("Cone")
        else name
        for name in source_transform[scale_name].bones
    )
    assert final_transform[depth_name].bones == assembly.rig.info.sub_bone_scale_names
    assert find_spine41_unsafe_world_constraints(finalized.document) == ()

    final_bones = {bone.name: bone for bone in finalized.document.bones}
    for wrapper_name in assembly.rig.info.sub_bone_scale_names:
        bridge_name = f"{wrapper_name}_spine41_bridge"
        assert final_bones[bridge_name].parent == profile.rotate_x_bone("Cone")
        assert final_bones[wrapper_name].parent == bridge_name


def test_spine40_finalization_remaps_weighted_indices_and_builder_metadata() -> None:
    assembly = _assembly()
    source_component = assembly.document_build.components[0]
    source_attachment = source_component.attachment
    source_vertices = decode_weighted_vertices(
        source_attachment.vertices,
        expected_vertex_count=3,
    )

    finalized = finalize_a1_document_assembly_for_target(
        assembly,
        spine_target="4.0.64",
        prefix="Cone",
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

    final_skin_attachment = finalized.document.skins[0].attachments[
        "Cone_Segment_0"
    ]["Cone_Segment_0"]
    build_skin_attachment = finalized.document_build.skins[0].attachments[
        "Cone_Segment_0"
    ]["Cone_Segment_0"]
    assert final_component.attachment is final_skin_attachment
    assert build_skin_attachment is final_skin_attachment


def test_spine40_finalization_is_idempotent() -> None:
    first = finalize_a1_document_assembly_for_target(
        _assembly(),
        spine_target="4.0.64",
        prefix="Cone",
    )
    second = finalize_a1_document_assembly_for_target(
        first,
        spine_target="4.0.64",
        prefix="Cone",
    )

    assert second.document == first.document
    assert second.document_build.components == first.document_build.components
    assert second.document_build.skins == first.document_build.skins
    assert second.rig is first.rig
