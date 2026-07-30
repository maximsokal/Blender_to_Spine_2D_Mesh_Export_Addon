"""Regression contracts for post-assembly Spine 4.1 rig adaptation."""

from __future__ import annotations

from Blender_to_Spine2D_Mesh_Exporter.application.a1_document_assembly import (
    A1DocumentAssemblyResult,
    A1DocumentAssemblySettings,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.a1_document_preparation import (
    finalize_a1_document_assembly_for_target,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.legacy_attachment_builder import (
    LegacyMeshDocumentBuildResult,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.legacy_rig_contracts import (
    LegacyRigBuildRequest,
    LegacyZGroup,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.model import SpineDocument
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.rig_builder import build_rig
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.rig_profiles import (
    A1RigProfile,
    A1RigSetupPoseMode,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.spine41_setup_safety import (
    find_spine41_unsafe_world_constraints,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.version_target import SpineJsonTarget


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


def _assembly() -> A1DocumentAssemblyResult:
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


def test_spine_four_two_finalization_preserves_the_canonical_assembly_identity() -> None:
    assembly = _assembly()

    finalized = finalize_a1_document_assembly_for_target(
        assembly,
        spine_target=SpineJsonTarget.SPINE_4_2,
        prefix="Cone",
    )

    assert finalized is assembly


def test_spine_four_one_finalization_changes_only_the_immutable_document() -> None:
    assembly = _assembly()
    canonical_rig = assembly.rig
    canonical_document = assembly.document

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
    assert finalized.document.bones == canonical_document.bones
    assert finalized.document.ik == canonical_document.ik

    source_by_name = {item.name: item for item in canonical_document.transform}
    final_by_name = {item.name: item for item in finalized.document.transform}
    scale_name = canonical_rig.profile.scale_constraint("Cone")
    depth_name = canonical_rig.profile.scale_depth_constraint("Cone")

    assert tuple(final_by_name) == tuple(source_by_name)
    assert final_by_name[scale_name].extras["local"] is True
    assert final_by_name[scale_name].extras["relative"] is True
    assert final_by_name[depth_name].bones == canonical_rig.info.sub_bone_names

    unchanged = set(source_by_name) - {scale_name, depth_name}
    assert all(final_by_name[name] == source_by_name[name] for name in unchanged)
    assert find_spine41_unsafe_world_constraints(finalized.document) == ()


def test_spine_four_one_finalization_is_idempotent() -> None:
    first = finalize_a1_document_assembly_for_target(
        _assembly(),
        spine_target="4.1.24",
        prefix="Cone",
    )

    second = finalize_a1_document_assembly_for_target(
        first,
        spine_target="4.1.24",
        prefix="Cone",
    )

    assert second.document == first.document
    assert second.rig is first.rig


def test_spine_four_one_finalization_rejects_prefix_drift() -> None:
    assembly = _assembly()

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
