"""Pure profile routing tests for rendered Camera Projection placement."""

from __future__ import annotations

from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.a1_projection_finalization import (
    _positioned_projection_rig,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine import (
    A1CameraLayerProjectionKind,
    A1RigProfile,
    A1RigSetupPoseMode,
    LegacyRigBuildRequest,
    LegacyZGroup,
    LegacyZGroupOriginMode,
    SpineJsonTarget,
    build_rig,
)


def _rig(profile: A1RigProfile):
    return build_rig(
        LegacyRigBuildRequest(
            prefix="Rendered",
            texture_width=128,
            texture_height=128,
            z_groups=(LegacyZGroup(-1.0), LegacyZGroup(2.0)),
            main_position_pixels=None,
            setup_pose_mode=A1RigSetupPoseMode.PRESERVE_COMPOSITION,
        ),
        profile,
        spine_target=SpineJsonTarget.SPINE_4_2,
    )


def _bone_by_name(rig, name: str):
    return next(bone for bone in rig.bones if bone.name == name)


def test_two_axis_rendered_projection_uses_camera_relative_rigid_setup() -> None:
    source = _rig(A1RigProfile.TWO_AXIS_ROTATION_SCALE)

    result = _positioned_projection_rig(
        source,
        (14.5, -6.25),
        -8.0,
        A1CameraLayerProjectionKind.PERSPECTIVE,
    )

    assert result is not source
    assert result.request.main_position_pixels == (14.5, -6.25)
    assert result.request.setup_pose_mode is A1RigSetupPoseMode.PREPROJECTED_SCREEN
    assert result.request.z_group_origin_mode is LegacyZGroupOriginMode.OBJECT_ORIGIN
    assert result.request.z_groups == (LegacyZGroup(-8.0),)
    assert result.request.camera_layer_projection_kind is (
        A1CameraLayerProjectionKind.PERSPECTIVE
    )
    assert len(result.info.z_groups) == 1

    main = _bone_by_name(result, result.info.main_bone_name)
    base = _bone_by_name(result, result.info.base_bone_name)
    assert (main.x, main.y) == (0.0, 0.0)
    assert (base.x, base.y) == (14.5, -6.25)

    assert source.request.main_position_pixels is None
    assert source.request.setup_pose_mode is A1RigSetupPoseMode.PRESERVE_COMPOSITION
    result.validate()


def test_orthographic_rendered_projection_disables_depth_scale() -> None:
    source = _rig(A1RigProfile.TWO_AXIS_ROTATION_SCALE)

    result = _positioned_projection_rig(
        source,
        (-3.0, 5.0),
        -4.5,
        A1CameraLayerProjectionKind.ORTHOGRAPHIC,
    )

    depth_constraint = next(
        constraint for constraint in result.transform if constraint.order == 3
    )
    assert result.request.camera_layer_projection_kind is (
        A1CameraLayerProjectionKind.ORTHOGRAPHIC
    )
    assert depth_constraint.extras["mixScaleX"] == 0
    result.validate()


def test_three_axis_rendered_projection_preserves_historical_setup_contract() -> None:
    source = _rig(A1RigProfile.THREE_AXIS_ROTATION)

    result = _positioned_projection_rig(
        source,
        (-9.0, 4.0),
        -5.0,
        A1CameraLayerProjectionKind.PERSPECTIVE,
    )

    assert result is not source
    assert result.request.main_position_pixels == (-9.0, 4.0)
    assert result.request.setup_pose_mode is A1RigSetupPoseMode.PRESERVE_COMPOSITION
    assert result.request.z_groups == source.request.z_groups
    assert result.request.camera_layer_projection_kind is None
    assert result.profile.profile_id == source.profile.profile_id
    result.validate()
