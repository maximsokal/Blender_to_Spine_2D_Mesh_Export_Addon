from __future__ import annotations

from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.a1_depth_document_preparation import (
    _build_depth_rig_request,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.geometry import DepthProjectionBaseMode
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


_PREFIX = "DepthSurface"
_Z_GROUPS = (
    LegacyZGroup(z_value=10.0, height_real_pixels=1000.0),
    LegacyZGroup(z_value=12.0, height_real_pixels=1200.0),
)


def _depth_request() -> LegacyRigBuildRequest:
    return _build_depth_rig_request(
        prefix=_PREFIX,
        texture_width=512,
        texture_height=512,
        z_groups=_Z_GROUPS,
        main_position_pixels=(25.0, -40.0),
        scale_mode=UniformScaleMode.AVERAGE,
        base_mode=DepthProjectionBaseMode.FARTHEST_VISIBLE,
    )


def _preserve_request() -> LegacyRigBuildRequest:
    return LegacyRigBuildRequest(
        prefix=_PREFIX,
        texture_width=512,
        texture_height=512,
        z_groups=_Z_GROUPS,
        main_position_pixels=(25.0, -40.0),
        scale_mode=UniformScaleMode.AVERAGE,
        setup_pose_mode=A1RigSetupPoseMode.PRESERVE_COMPOSITION,
        z_group_origin_mode=LegacyZGroupOriginMode.OBJECT_ORIGIN,
    )


def _transform_by_name(rig):
    return {constraint.name: constraint for constraint in rig.transform}


def _bone_by_name(rig):
    return {bone.name: bone for bone in rig.bones}


def test_depth_rig_request_forces_neutral_multi_depth_setup_mode() -> None:
    request = _depth_request()

    assert request.setup_pose_mode is A1RigSetupPoseMode.CAMERA_DEPTH_SURFACE
    assert request.z_group_origin_mode is LegacyZGroupOriginMode.OBJECT_ORIGIN
    assert request.z_groups == _Z_GROUPS
    assert request.main_position_pixels == (25.0, -40.0)


def test_two_axis_depth_setup_has_zero_legacy_rotation_and_scale_offsets() -> None:
    rig = build_rig(
        _depth_request(),
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
    assert rotation_x.extras.get("mixRotate", 1) == 1
    assert rotation_y.extras.get("mixRotate", 1) == 1

    bones = _bone_by_name(rig)
    assert len(rig.info.z_groups) == 2
    for group in rig.info.z_groups:
        assert bones[group.scale_bone_name].parent == rig.info.main_rotation_bone_name
        assert bones[group.bone_name].parent == group.scale_bone_name


def test_two_axis_normal_export_retains_reference_setup_offsets() -> None:
    rig = build_rig(
        _preserve_request(),
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
    assert constraints[profile.scale_depth_constraint(_PREFIX)].extras["scaleX"] == -1


def test_three_axis_depth_setup_is_neutral_without_disabling_controls() -> None:
    rig = build_rig(
        _depth_request(),
        A1RigProfile.THREE_AXIS_ROTATION,
        spine_target=SpineJsonTarget.SPINE_4_2,
    )
    constraints = _transform_by_name(rig)
    profile = rig.profile

    rotation_x = constraints[profile.rotation_x_constraint(_PREFIX)]
    depth_scale = constraints[profile.scale_constraint(_PREFIX)]

    assert rotation_x.extras["rotation"] == 0.0
    assert depth_scale.extras["scaleX"] == 0.0
    assert rotation_x.extras.get("mixRotate", 1) == 1
    assert depth_scale.extras.get("mixScaleX", 1) == 1
    assert len(rig.info.z_groups) == 2
