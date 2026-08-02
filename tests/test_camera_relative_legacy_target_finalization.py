"""Regression contracts for camera-relative rigs on Spine 3.8-4.1 targets."""

from __future__ import annotations

import pytest

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
    LegacyZGroupOriginMode,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.rig_builder import build_rig
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.rig_profiles import (
    A1CameraLayerProjectionKind,
    A1RigProfile,
    A1RigSetupPoseMode,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.spine41_setup_safety import (
    find_spine41_unsafe_world_constraints,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.version_target import (
    SpineJsonTarget,
)


_PREFIX = "CameraLayer"


def _assembly(target: SpineJsonTarget) -> A1DocumentAssemblyResult:
    rig = build_rig(
        LegacyRigBuildRequest(
            prefix=_PREFIX,
            texture_width=128,
            texture_height=128,
            z_groups=(LegacyZGroup(-4.5),),
            main_position_pixels=(19.0, -11.0),
            setup_pose_mode=A1RigSetupPoseMode.PREPROJECTED_SCREEN,
            z_group_origin_mode=LegacyZGroupOriginMode.OBJECT_ORIGIN,
            camera_layer_projection_kind=(
                A1CameraLayerProjectionKind.PERSPECTIVE
            ),
        ),
        A1RigProfile.TWO_AXIS_ROTATION_SCALE,
        spine_target=target,
    )
    z_group_index = rig.info.z_groups[0].index
    attachment_request = LegacyMeshAttachmentRequest(
        slot_name=f"{_PREFIX}_Segment_0",
        attachment_name=f"{_PREFIX}_Segment_0",
        vertex_prefix=f"{_PREFIX}_Segment_0",
        image_path=f"images/{_PREFIX}_Baked",
        width=128,
        height=128,
        vertices=(
            LegacyAttachmentVertex(
                index=0,
                uv=(0.0, 0.0),
                bone_position_pixels=(-24.0, -20.0),
                z_group_index=z_group_index,
            ),
            LegacyAttachmentVertex(
                index=1,
                uv=(1.0, 0.0),
                bone_position_pixels=(24.0, -20.0),
                z_group_index=z_group_index,
            ),
            LegacyAttachmentVertex(
                index=2,
                uv=(0.5, 1.0),
                bone_position_pixels=(0.0, 20.0),
                z_group_index=z_group_index,
            ),
        ),
        triangles=(0, 1, 2),
        hull=3,
    )
    document_build = build_legacy_mesh_document(
        rig,
        (attachment_request,),
        skeleton_metadata={
            "spine": target.exact_version,
            "width": 128,
            "height": 128,
        },
    )
    return A1DocumentAssemblyResult(
        settings=A1DocumentAssemblySettings(
            prefix=_PREFIX,
            uv_layer_name="SpineBakeUV",
            image_path=f"images/{_PREFIX}_Baked",
            attachment_width=128,
            attachment_height=128,
            center_x=0.0,
            center_y=0.0,
        ),
        rig=rig,
        z_groups=object(),
        projections=(),
        document_build=document_build,
    )


def _transform_by_name(document) -> dict[str, object]:
    return {constraint.name: constraint for constraint in document.transform}


def _ik_by_name(document) -> dict[str, object]:
    return {constraint.name: constraint for constraint in document.ik}


@pytest.mark.parametrize(
    "target",
    (
        SpineJsonTarget.SPINE_4_0,
        SpineJsonTarget.SPINE_4_1,
    ),
)
def test_camera_relative_spine40_and_41_keep_scale_on_object_base(
    target: SpineJsonTarget,
) -> None:
    source = _assembly(target)
    profile = source.rig.profile
    source_component = source.document_build.components[0]

    finalized = finalize_a1_document_assembly_for_target(
        source,
        spine_target=target,
        prefix=_PREFIX,
    )

    transforms = _transform_by_name(finalized.document)
    scale = transforms[profile.scale_constraint(_PREFIX)]
    assert scale.bones == (profile.base_bone(_PREFIX),)
    assert find_spine41_unsafe_world_constraints(finalized.document) == ()
    assert len(finalized.document.bones) == len(source.document.bones) + 1
    assert finalized.document_build.components[0].vertex_bone_start_index == (
        source_component.vertex_bone_start_index + 1
    )

    repeated = finalize_a1_document_assembly_for_target(
        finalized,
        spine_target=target,
        prefix=_PREFIX,
    )
    assert repeated.document == finalized.document
    assert repeated.document_build.components == finalized.document_build.components


def test_camera_relative_spine38_uses_orbit_then_object_scale_schedule() -> None:
    source = _assembly(SpineJsonTarget.SPINE_3_8)
    profile = source.rig.profile

    finalized = finalize_a1_document_assembly_for_target(
        source,
        spine_target=SpineJsonTarget.SPINE_3_8,
        prefix=_PREFIX,
    )

    transforms = _transform_by_name(finalized.document)
    ik = _ik_by_name(finalized.document)
    position_name = f"{_PREFIX}_scale_spine38_position"
    assert position_name not in transforms

    rotation_x = transforms[profile.rotation_x_constraint(_PREFIX)]
    depth = transforms[profile.scale_depth_constraint(_PREFIX)]
    rotation_y = transforms[profile.rotation_y_constraint(_PREFIX)]
    scale = transforms[profile.scale_constraint(_PREFIX)]
    scale_ik = ik[profile.scale_ik_constraint(_PREFIX)]
    base_order = rotation_x.order

    assert (
        rotation_x.order,
        scale_ik.order,
        depth.order,
        rotation_y.order,
        scale.order,
    ) == tuple(range(base_order, base_order + 5))
    assert scale.bones == (profile.base_bone(_PREFIX),)
    assert find_spine41_unsafe_world_constraints(finalized.document) == ()

    repeated = finalize_a1_document_assembly_for_target(
        finalized,
        spine_target=SpineJsonTarget.SPINE_3_8,
        prefix=_PREFIX,
    )
    assert repeated.document == finalized.document
    assert repeated.document_build.components == finalized.document_build.components
