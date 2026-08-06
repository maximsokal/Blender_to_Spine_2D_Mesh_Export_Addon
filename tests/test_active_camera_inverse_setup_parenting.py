from __future__ import annotations

from Blender_to_Spine2D_Mesh_Exporter.domain.spine import (
    A1RigProfile,
    A1RigSetupPoseMode,
    SpineJsonTarget,
    build_legacy_mesh_attachment,
    build_rig,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.legacy_attachment_builder import (
    LegacyAttachmentVertex,
    LegacyMeshAttachmentRequest,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.legacy_rig_contracts import (
    LegacyRigBuildRequest,
    LegacyZGroup,
    LegacyZGroupOriginMode,
    UniformScaleMode,
)


_PREFIX = "CameraInverseParent"
_Z_GROUPS = (
    LegacyZGroup(z_value=-1.0, height_real_pixels=-64.0),
    LegacyZGroup(z_value=0.0, height_real_pixels=0.0),
    LegacyZGroup(z_value=1.0, height_real_pixels=64.0),
)


def _rig(setup_pose_mode: A1RigSetupPoseMode):
    return build_rig(
        LegacyRigBuildRequest(
            prefix=_PREFIX,
            texture_width=256,
            texture_height=256,
            z_groups=_Z_GROUPS,
            main_position_pixels=(25.0, -10.0),
            scale_mode=UniformScaleMode.AVERAGE,
            setup_pose_mode=setup_pose_mode,
            z_group_origin_mode=LegacyZGroupOriginMode.OBJECT_ORIGIN,
            camera_layer_projection_kind=None,
        ),
        A1RigProfile.TWO_AXIS_ROTATION_SCALE,
        spine_target=SpineJsonTarget.SPINE_4_2,
    )


def _attachment_request() -> LegacyMeshAttachmentRequest:
    return LegacyMeshAttachmentRequest(
        slot_name=f"{_PREFIX}_Segment_0",
        attachment_name=f"{_PREFIX}_Segment_0",
        vertex_prefix=f"{_PREFIX}_Segment_0",
        image_path="images/test",
        width=64.0,
        height=64.0,
        vertices=(
            LegacyAttachmentVertex(
                index=0,
                uv=(0.0, 0.0),
                bone_position_pixels=(-5.0, -2.0),
                z_group_index=1,
            ),
            LegacyAttachmentVertex(
                index=1,
                uv=(1.0, 0.0),
                bone_position_pixels=(5.0, -2.0),
                z_group_index=2,
            ),
            LegacyAttachmentVertex(
                index=2,
                uv=(0.5, 1.0),
                bone_position_pixels=(0.0, 6.0),
                z_group_index=3,
            ),
        ),
        triangles=(0, 1, 2),
        hull=3,
    )


def test_camera_view_vertices_use_inverse_setup_parents() -> None:
    rig = _rig(A1RigSetupPoseMode.CAMERA_VIEW_NORMAL)
    built = build_legacy_mesh_attachment(rig, _attachment_request())

    expected = tuple(
        rig.profile.z_camera_setup_bone(_PREFIX, group.index)
        for group in rig.info.z_groups
    )
    assert tuple(bone.parent for bone in built.vertex_bones) == expected

    bones = {bone.name: bone for bone in rig.bones}
    for group, parent_name in zip(rig.info.z_groups, expected, strict=True):
        compensation = bones[parent_name]
        assert compensation.parent == group.bone_name
        assert compensation.y == round(-float(group.y_offset_pixels), 2)


def test_signed_axis_vertices_keep_historical_depth_parents() -> None:
    rig = _rig(A1RigSetupPoseMode.PRESERVE_COMPOSITION)
    built = build_legacy_mesh_attachment(rig, _attachment_request())

    assert tuple(bone.parent for bone in built.vertex_bones) == tuple(
        group.bone_name for group in rig.info.z_groups
    )
    bone_names = {bone.name for bone in rig.bones}
    assert all(
        rig.profile.z_camera_setup_bone(_PREFIX, group.index) not in bone_names
        for group in rig.info.z_groups
    )
