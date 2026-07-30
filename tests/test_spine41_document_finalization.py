"""Regression contracts for post-assembly Spine 4.1 scale topology."""

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
    LegacyMeshDocumentBuildResult,
    build_legacy_mesh_document,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.legacy_rig_contracts import (
    LegacyRigBuildRequest,
    LegacyZGroup,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.model import MeshAttachment, SpineDocument
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.rig_builder import build_rig
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.rig_profiles import (
    A1RigProfile,
    A1RigSetupPoseMode,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.spine41_setup_safety import (
    calculate_spine41_setup_matrices,
    find_spine41_unsafe_world_constraints,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.version_target import SpineJsonTarget
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.weighted_vertices import (
    decode_weighted_vertices,
)


def _rig():
    return build_rig(
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
        spine_target=SpineJsonTarget.SPINE_4_1,
    )


def _settings() -> A1DocumentAssemblySettings:
    return A1DocumentAssemblySettings(
        prefix="Cone",
        uv_layer_name="SpineBakeUV",
        image_path="images/Cone_Baked",
        attachment_width=256,
        attachment_height=256,
        center_x=0.0,
        center_y=0.0,
    )


def _empty_assembly() -> A1DocumentAssemblyResult:
    rig = _rig()
    document = SpineDocument(
        skeleton={"spine": "4.1.24"},
        bones=rig.bones,
        slots=(),
        skins=(),
        ik=rig.ik,
        transform=rig.transform,
    )
    document_build = LegacyMeshDocumentBuildResult(
        rig=rig,
        requests=(),
        components=(),
        skins=(),
        document=document,
    )
    return A1DocumentAssemblyResult(
        settings=_settings(),
        rig=rig,
        z_groups=object(),
        projections=(),
        document_build=document_build,
    )


def _mesh_assembly() -> A1DocumentAssemblyResult:
    rig = _rig()
    z_group_index = rig.info.z_groups[0].index
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
                z_group_index=z_group_index,
            ),
            LegacyAttachmentVertex(
                index=1,
                uv=(1.0, 0.0),
                bone_position_pixels=(32.0, -32.0),
                z_group_index=z_group_index,
            ),
            LegacyAttachmentVertex(
                index=2,
                uv=(0.5, 1.0),
                bone_position_pixels=(0.0, 32.0),
                z_group_index=z_group_index,
            ),
        ),
        triangles=(0, 1, 2),
        hull=3,
    )
    document_build = build_legacy_mesh_document(
        rig,
        (request,),
        skeleton_metadata={
            "spine": "4.1.24",
            "width": 256,
            "height": 256,
        },
    )
    return A1DocumentAssemblyResult(
        settings=_settings(),
        rig=rig,
        z_groups=object(),
        projections=(),
        document_build=document_build,
    )


def test_spine_four_two_finalization_preserves_the_canonical_assembly_identity() -> None:
    assembly = _empty_assembly()

    finalized = finalize_a1_document_assembly_for_target(
        assembly,
        spine_target=SpineJsonTarget.SPINE_4_2,
        prefix="Cone",
    )

    assert finalized is assembly


def test_spine_four_one_finalization_preserves_scale_semantics_with_bridges() -> None:
    assembly = _empty_assembly()
    canonical_rig = assembly.rig
    canonical_document = assembly.document
    profile = canonical_rig.profile

    finalized = finalize_a1_document_assembly_for_target(
        assembly,
        spine_target=SpineJsonTarget.SPINE_4_1,
        prefix="Cone",
    )

    assert finalized is not assembly
    assert finalized.rig is canonical_rig
    assert finalized.document_build.rig is canonical_rig
    assert assembly.document is canonical_document
    assert assembly.document.transform == canonical_rig.transform
    assert finalized.document.ik == canonical_document.ik

    source_by_name = {item.name: item for item in canonical_document.transform}
    final_by_name = {item.name: item for item in finalized.document.transform}
    scale_name = profile.scale_constraint("Cone")
    depth_name = profile.scale_depth_constraint("Cone")

    assert tuple(final_by_name) == tuple(source_by_name)
    assert final_by_name[scale_name].extras.get("local") is None
    assert final_by_name[scale_name].extras["relative"] is True
    assert final_by_name[scale_name].bones == tuple(
        profile.scale_rotate_x_bone("Cone")
        if name == profile.rotate_x_bone("Cone")
        else name
        for name in source_by_name[scale_name].bones
    )
    assert final_by_name[depth_name] == source_by_name[depth_name]
    assert final_by_name[depth_name].bones == canonical_rig.info.sub_bone_scale_names

    unchanged = set(source_by_name) - {scale_name}
    assert all(final_by_name[name] == source_by_name[name] for name in unchanged)

    source_bones = {bone.name: bone for bone in canonical_document.bones}
    final_bones = {bone.name: bone for bone in finalized.document.bones}
    assert len(final_bones) == len(source_bones) + len(canonical_rig.info.z_groups)

    for wrapper_name in canonical_rig.info.sub_bone_scale_names:
        source_wrapper = source_bones[wrapper_name]
        final_wrapper = final_bones[wrapper_name]
        bridge_name = f"{wrapper_name}_spine41_bridge"
        bridge = final_bones[bridge_name]

        assert bridge.parent == profile.rotate_x_bone("Cone")
        assert bridge.x == source_wrapper.x
        assert bridge.y == source_wrapper.y
        assert bridge.extras == {"inherit": "onlyTranslation"}
        assert final_wrapper.parent == bridge_name
        assert float(final_wrapper.x or 0.0) == 0.0
        assert float(final_wrapper.y or 0.0) == 0.0
        assert final_wrapper.rotation == source_wrapper.rotation
        assert final_wrapper.extras == source_wrapper.extras

    source_matrices = calculate_spine41_setup_matrices(canonical_document.bones)
    final_matrices = calculate_spine41_setup_matrices(finalized.document.bones)
    for name in (
        *canonical_rig.info.sub_bone_scale_names,
        *canonical_rig.info.sub_bone_names,
    ):
        assert final_matrices[name] == source_matrices[name]

    assert find_spine41_unsafe_world_constraints(finalized.document) == ()


def test_spine_four_one_finalization_remaps_weighted_indices_and_builder_metadata() -> None:
    assembly = _mesh_assembly()
    source_component = assembly.document_build.components[0]
    source_attachment = source_component.attachment
    source_vertices = decode_weighted_vertices(
        source_attachment.vertices,
        expected_vertex_count=3,
    )

    finalized = finalize_a1_document_assembly_for_target(
        assembly,
        spine_target="4.1.24",
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


def test_spine_four_one_finalization_is_idempotent() -> None:
    first = finalize_a1_document_assembly_for_target(
        _mesh_assembly(),
        spine_target="4.1.24",
        prefix="Cone",
    )

    second = finalize_a1_document_assembly_for_target(
        first,
        spine_target="4.1.24",
        prefix="Cone",
    )

    assert second.document == first.document
    assert second.document_build.components == first.document_build.components
    assert second.document_build.skins == first.document_build.skins
    assert second.rig is first.rig


def test_spine_four_one_finalization_rejects_prefix_drift() -> None:
    assembly = _empty_assembly()

    try:
        finalize_a1_document_assembly_for_target(
            assembly,
            spine_target=SpineJsonTarget.SPINE_4_1,
            prefix="Other",
        )
    except ValueError as exc:
        assert "does not match rig prefix" in str(exc)
    else:
        raise AssertionError("prefix drift was accepted")
