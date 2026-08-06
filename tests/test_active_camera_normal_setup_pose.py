from __future__ import annotations

from Blender_to_Spine2D_Mesh_Exporter.domain.spine import (
    A1RigProfile,
    A1RigSetupPoseMode,
    SpineJsonTarget,
    build_rig,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.legacy_rig_contracts import (
    LegacyRigBuildRequest,
    LegacyZGroup,
    LegacyZGroupOriginMode,
    UniformScaleMode,
)


_PREFIX = "ActiveCameraNormal"
_Z_GROUPS = (
    LegacyZGroup(z_value=-1.0, height_real_pixels=-128.0),
    LegacyZGroup(z_value=0.0, height_real_pixels=0.0),
    LegacyZGroup(z_value=1.5, height_real_pixels=192.0),
)
_MAIN_POSITION = (125.0, -75.0)


def _request(
    setup_pose_mode: A1RigSetupPoseMode,
) -> LegacyRigBuildRequest:
    if not isinstance(setup_pose_mode, A1RigSetupPoseMode):
        raise TypeError("setup_pose_mode must be A1RigSetupPoseMode")
    return LegacyRigBuildRequest(
        prefix=_PREFIX,
        texture_width=512,
        texture_height=512,
        z_groups=_Z_GROUPS,
        main_position_pixels=_MAIN_POSITION,
        scale_mode=UniformScaleMode.AVERAGE,
        setup_pose_mode=setup_pose_mode,
        z_group_origin_mode=LegacyZGroupOriginMode.OBJECT_ORIGIN,
        camera_layer_projection_kind=None,
    )


def _transform_by_name(rig):
    return {constraint.name: constraint for constraint in rig.transform}


def _bone_by_name(rig):
    return {bone.name: bone for bone in rig.bones}


def _minimum_depth_y(rig) -> float:
    return min(float(group.y_offset_pixels) for group in rig.info.z_groups)


def _assert_camera_setup_inverse_bones(rig) -> None:
    bones = _bone_by_name(rig)
    for group in rig.info.z_groups:
        compensation_name = rig.profile.z_camera_setup_bone(
            rig.info.prefix,
            group.index,
        )
        compensation = bones[compensation_name]
        assert compensation.parent == group.bone_name
        assert compensation.x is None
        assert compensation.y == round(-float(group.y_offset_pixels), 2)


def _assert_no_camera_setup_inverse_bones(rig) -> None:
    bone_names = {bone.name for bone in rig.bones}
    for group in rig.info.z_groups:
        assert (
            rig.profile.z_camera_setup_bone(rig.info.prefix, group.index)
            not in bone_names
        )


def test_two_axis_active_camera_normal_has_neutral_full_rank_setup_and_inverses() -> None:
    rig = build_rig(
        _request(A1RigSetupPoseMode.CAMERA_VIEW_NORMAL),
        A1RigProfile.TWO_AXIS_ROTATION_SCALE,
        spine_target=SpineJsonTarget.SPINE_4_2,
    )
    constraints = _transform_by_name(rig)
    profile = rig.profile

    rotation_x = constraints[profile.rotation_x_constraint(_PREFIX)]
    rotation_y = constraints[profile.rotation_y_constraint(_PREFIX)]
    object_scale = constraints[profile.scale_constraint(_PREFIX)]
    depth_scale = constraints[profile.scale_depth_constraint(_PREFIX)]

    assert rig.request.setup_pose_mode is A1RigSetupPoseMode.CAMERA_VIEW_NORMAL
    assert rig.request.main_position_pixels == _MAIN_POSITION
    assert rig.request.z_group_origin_mode is LegacyZGroupOriginMode.OBJECT_ORIGIN
    assert rig.request.camera_layer_projection_kind is None
    assert len(rig.info.z_groups) == len(_Z_GROUPS)

    assert rotation_x.extras["rotation"] == 0.0
    assert rotation_y.extras["rotation"] == 0.0
    assert depth_scale.extras["x"] == 0.0
    assert depth_scale.extras["scaleX"] == 0.0
    assert rotation_x.extras.get("mixRotate", 1) == 1
    assert rotation_y.extras.get("mixRotate", 1) == 1

    expected_scale_bones = (
        rig.info.main_rotation_bone_name,
        *tuple(reversed(rig.info.sub_bone_names)),
    )
    assert object_scale.bones == expected_scale_bones
    assert depth_scale.bones == rig.info.sub_bone_scale_names

    bones = _bone_by_name(rig)
    assert bones[rig.info.base_bone_name].parent == rig.info.main_bone_name
    for group in rig.info.z_groups:
        assert bones[group.scale_bone_name].parent == rig.info.main_rotation_bone_name
        assert bones[group.bone_name].parent == group.scale_bone_name
    _assert_camera_setup_inverse_bones(rig)


def test_two_axis_depth_camera_surface_keeps_neutral_depth_setup_without_inverses() -> None:
    rig = build_rig(
        _request(A1RigSetupPoseMode.CAMERA_DEPTH_SURFACE),
        A1RigProfile.TWO_AXIS_ROTATION_SCALE,
        spine_target=SpineJsonTarget.SPINE_4_2,
    )
    constraints = _transform_by_name(rig)
    profile = rig.profile

    rotation_x = constraints[profile.rotation_x_constraint(_PREFIX)]
    rotation_y = constraints[profile.rotation_y_constraint(_PREFIX)]
    depth_scale = constraints[profile.scale_depth_constraint(_PREFIX)]

    assert rotation_x.extras["rotation"] == 0.0
    assert rotation_y.extras["rotation"] == 0.0
    assert depth_scale.extras["x"] == 0.0
    assert depth_scale.extras["scaleX"] == 0.0
    _assert_no_camera_setup_inverse_bones(rig)


def test_two_axis_signed_axis_normal_keeps_historical_setup_offsets() -> None:
    rig = build_rig(
        _request(A1RigSetupPoseMode.PRESERVE_COMPOSITION),
        A1RigProfile.TWO_AXIS_ROTATION_SCALE,
        spine_target=SpineJsonTarget.SPINE_4_2,
    )
    constraints = _transform_by_name(rig)
    profile = rig.profile

    assert (
        constraints[profile.rotation_x_constraint(_PREFIX)].extras["rotation"]
        == profile.rotation_x_setup_degrees
    )
    assert (
        constraints[profile.rotation_y_constraint(_PREFIX)].extras["rotation"]
        == profile.rotation_y_setup_degrees
    )
    depth_scale = constraints[profile.scale_depth_constraint(_PREFIX)]
    assert depth_scale.extras["x"] == _minimum_depth_y(rig)
    assert depth_scale.extras["scaleX"] == -1
    _assert_no_camera_setup_inverse_bones(rig)


def test_three_axis_active_camera_normal_has_neutral_full_rank_setup_and_inverses() -> None:
    rig = build_rig(
        _request(A1RigSetupPoseMode.CAMERA_VIEW_NORMAL),
        A1RigProfile.THREE_AXIS_ROTATION,
        spine_target=SpineJsonTarget.SPINE_4_2,
    )
    constraints = _transform_by_name(rig)
    profile = rig.profile

    rotation_x = constraints[profile.rotation_x_constraint(_PREFIX)]
    depth_scale = constraints[profile.scale_constraint(_PREFIX)]

    assert rig.request.setup_pose_mode is A1RigSetupPoseMode.CAMERA_VIEW_NORMAL
    assert len(rig.info.z_groups) == len(_Z_GROUPS)
    assert rotation_x.extras["rotation"] == 0.0
    assert depth_scale.extras["scaleX"] == 0.0
    assert rotation_x.extras.get("mixRotate", 1) == 1
    assert depth_scale.extras.get("mixScaleX", 1) == 1
    _assert_camera_setup_inverse_bones(rig)


def test_three_axis_depth_camera_surface_keeps_neutral_depth_setup_without_inverses() -> None:
    rig = build_rig(
        _request(A1RigSetupPoseMode.CAMERA_DEPTH_SURFACE),
        A1RigProfile.THREE_AXIS_ROTATION,
        spine_target=SpineJsonTarget.SPINE_4_2,
    )
    constraints = _transform_by_name(rig)
    profile = rig.profile

    rotation_x = constraints[profile.rotation_x_constraint(_PREFIX)]
    depth_scale = constraints[profile.scale_constraint(_PREFIX)]

    assert rotation_x.extras["rotation"] == 0.0
    assert depth_scale.extras["scaleX"] == 0.0
    _assert_no_camera_setup_inverse_bones(rig)


def test_three_axis_signed_axis_normal_keeps_historical_setup_offsets() -> None:
    rig = build_rig(
        _request(A1RigSetupPoseMode.PRESERVE_COMPOSITION),
        A1RigProfile.THREE_AXIS_ROTATION,
        spine_target=SpineJsonTarget.SPINE_4_2,
    )
    constraints = _transform_by_name(rig)
    profile = rig.profile

    assert constraints[profile.rotation_x_constraint(_PREFIX)].extras["rotation"] == 90
    assert constraints[profile.scale_constraint(_PREFIX)].extras["scaleX"] == -1
    _assert_no_camera_setup_inverse_bones(rig)
