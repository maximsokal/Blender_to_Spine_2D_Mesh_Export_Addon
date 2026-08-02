"""Attachment-parent regression for rigid camera-relative object layers."""

from __future__ import annotations

from Blender_to_Spine2D_Mesh_Exporter.domain.spine import (
    A1CameraLayerProjectionKind,
    A1RigProfile,
    A1RigSetupPoseMode,
    LegacyAttachmentVertex,
    LegacyMeshAttachmentRequest,
    LegacyRigBuildRequest,
    LegacyZGroup,
    LegacyZGroupOriginMode,
    SpineJsonTarget,
    build_legacy_mesh_attachment,
    build_rig,
)


def _camera_rig():
    return build_rig(
        LegacyRigBuildRequest(
            prefix="CameraLayer",
            texture_width=128,
            texture_height=128,
            z_groups=(LegacyZGroup(-6.0),),
            main_position_pixels=(24.0, -13.0),
            setup_pose_mode=A1RigSetupPoseMode.PREPROJECTED_SCREEN,
            z_group_origin_mode=LegacyZGroupOriginMode.OBJECT_ORIGIN,
            camera_layer_projection_kind=(
                A1CameraLayerProjectionKind.PERSPECTIVE
            ),
        ),
        A1RigProfile.TWO_AXIS_ROTATION_SCALE,
        spine_target=SpineJsonTarget.SPINE_4_2,
    )


def _request(rig) -> LegacyMeshAttachmentRequest:
    group_index = rig.info.z_groups[0].index
    return LegacyMeshAttachmentRequest(
        slot_name="CameraLayer_Segment_0",
        attachment_name="CameraLayer_Segment_0",
        vertex_prefix="CameraLayer_Segment_0",
        image_path="images/CameraLayer_Baked",
        width=128.0,
        height=128.0,
        vertices=(
            LegacyAttachmentVertex(
                index=0,
                uv=(0.0, 0.0),
                bone_position_pixels=(-10.0, -8.0),
                z_group_index=group_index,
            ),
            LegacyAttachmentVertex(
                index=1,
                uv=(1.0, 0.0),
                bone_position_pixels=(10.0, -8.0),
                z_group_index=group_index,
            ),
            LegacyAttachmentVertex(
                index=2,
                uv=(0.5, 1.0),
                bone_position_pixels=(0.0, 12.0),
                z_group_index=group_index,
            ),
        ),
        triangles=(0, 1, 2),
        hull=3,
        edges=(0, 1, 1, 2, 2, 0),
    )


def test_camera_layer_vertex_bones_are_object_local_below_base() -> None:
    rig = _camera_rig()

    result = build_legacy_mesh_attachment(rig, _request(rig))

    assert {bone.parent for bone in result.vertex_bones} == {
        rig.info.base_bone_name
    }
    base = next(bone for bone in rig.bones if bone.name == rig.info.base_bone_name)
    assert base.parent == rig.info.z_groups[0].bone_name


def test_camera_layer_scale_targets_base_not_orbital_bones() -> None:
    rig = _camera_rig()

    scale_constraint = next(
        constraint for constraint in rig.transform if constraint.order == 2
    )
    rotation_x = next(
        constraint for constraint in rig.transform if constraint.order == 0
    )
    rotation_y = next(
        constraint for constraint in rig.transform if constraint.order == 4
    )

    assert scale_constraint.bones == (rig.info.base_bone_name,)
    assert rig.info.main_rotation_bone_name in rotation_x.bones
    assert rotation_y.bones == (rig.info.z_groups[0].bone_name,)
