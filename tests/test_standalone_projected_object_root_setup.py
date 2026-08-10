"""Regression contracts for standalone signed-axis Normal/UV setup placement.

Signed-axis source geometry is already projected into canonical U/V/depth space before
rig construction. Standalone multi-object preparation therefore selects a dedicated
projected-axis setup baseline. The baseline must not reuse Active Camera hierarchy:
ordinary depth parents, IK, scale targets, and depth mapping remain intact so live X/Y
controls keep the historical deformable rig behavior.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from Blender_to_Spine2D_Mesh_Exporter.application import (
    A1MultiObjectMode,
    A1SingleObjectExportSettings,
    ExportSettings,
    resolve_a1_multi_object_preparation_settings,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.projection import A1ProjectionDirection
from Blender_to_Spine2D_Mesh_Exporter.domain.spine import (
    A1RigProfile,
    A1RigSetupPoseMode,
    LegacyRigBuildRequest,
    LegacyZGroup,
    build_rig,
)


def _settings(direction: A1ProjectionDirection) -> A1SingleObjectExportSettings:
    if not isinstance(direction, A1ProjectionDirection):
        raise TypeError("direction must be A1ProjectionDirection")
    return A1SingleObjectExportSettings(
        export=ExportSettings(
            texture_width=128,
            texture_height=128,
            output_directory=Path("standalone-projected-object-root-test-output"),
            rig_profile=A1RigProfile.TWO_AXIS_ROTATION_SCALE.value,
        ),
        prefix="Projected",
        projection_direction=direction,
        rig_setup_pose_mode=A1RigSetupPoseMode.PRESERVE_COMPOSITION,
    )


def _request(setup_pose_mode: A1RigSetupPoseMode) -> LegacyRigBuildRequest:
    if not isinstance(setup_pose_mode, A1RigSetupPoseMode):
        raise TypeError("setup_pose_mode must be A1RigSetupPoseMode")
    return LegacyRigBuildRequest(
        prefix="Projected",
        texture_width=128,
        texture_height=128,
        z_groups=(
            LegacyZGroup(-2.0, height_real_pixels=-16.0),
            LegacyZGroup(0.0, height_real_pixels=0.0),
            LegacyZGroup(3.0, height_real_pixels=24.0),
        ),
        main_position_pixels=(17.0, -11.0),
        setup_pose_mode=setup_pose_mode,
    )


def _bone_names(result) -> tuple[str, ...]:
    return tuple(bone.name for bone in result.bones)


@pytest.mark.parametrize(
    "direction",
    (
        A1ProjectionDirection.POSITIVE_X,
        A1ProjectionDirection.NEGATIVE_X,
        A1ProjectionDirection.POSITIVE_Y,
        A1ProjectionDirection.NEGATIVE_Y,
        A1ProjectionDirection.POSITIVE_Z,
        A1ProjectionDirection.NEGATIVE_Z,
    ),
)
def test_standalone_signed_axis_policy_selects_projected_axis_setup(
    direction: A1ProjectionDirection,
) -> None:
    source = _settings(direction)

    resolved = resolve_a1_multi_object_preparation_settings(
        source,
        A1MultiObjectMode.STANDALONE,
    )

    assert source.rig_setup_pose_mode is A1RigSetupPoseMode.PRESERVE_COMPOSITION
    assert resolved.rig_setup_pose_mode is A1RigSetupPoseMode.PROJECTED_AXIS_NORMAL
    assert resolved.projection_direction is direction
    assert resolved.use_world_location_for_main_bone is True


def test_projected_axis_setup_preserves_deformable_hierarchy_and_depth_mapping() -> None:
    projected = build_rig(
        _request(A1RigSetupPoseMode.PROJECTED_AXIS_NORMAL),
        A1RigProfile.TWO_AXIS_ROTATION_SCALE,
    )
    preserved = build_rig(
        _request(A1RigSetupPoseMode.PRESERVE_COMPOSITION),
        A1RigProfile.TWO_AXIS_ROTATION_SCALE,
    )

    main = next(bone for bone in projected.bones if bone.name == "Projected_main")
    assert (main.x, main.y) == (17.0, -11.0)

    # The projected-axis route must keep the exact ordinary bone/IK hierarchy. In
    # particular, no Active-Camera inverse setup layer may appear between depth and
    # generated vertex bones later in attachment assembly.
    assert projected.bones == preserved.bones
    assert projected.ik == preserved.ik
    assert _bone_names(projected) == _bone_names(preserved)

    camera_setup_names = {
        projected.profile.z_camera_setup_bone(projected.info.prefix, group.index)
        for group in projected.info.z_groups
    }
    assert camera_setup_names.isdisjoint(set(_bone_names(projected)))

    projected_x, projected_y, projected_scale, projected_depth = projected.transform
    preserved_x, preserved_y, preserved_scale, preserved_depth = preserved.transform

    # Only the X/Y setup calibration changes. Constraint targets/bones and every other
    # numeric mapping remain the historical deformable contract.
    assert projected_x.bones == preserved_x.bones
    assert projected_x.target == preserved_x.target
    assert projected_y.bones == preserved_y.bones
    assert projected_y.target == preserved_y.target
    assert projected_x.extras["rotation"] == 0.0
    assert projected_y.extras["rotation"] == 0.0
    assert preserved_x.extras["rotation"] == -134.67
    assert preserved_y.extras["rotation"] == -17.43

    for key, value in preserved_x.extras.items():
        if key != "rotation":
            assert projected_x.extras[key] == value
    for key, value in preserved_y.extras.items():
        if key != "rotation":
            assert projected_y.extras[key] == value

    assert projected_scale == preserved_scale
    assert projected_depth == preserved_depth
    assert projected_depth.extras["x"] == -16.0
    assert projected_depth.extras["scaleX"] == -1

    projected.validate()
    preserved.validate()


def test_preserve_composition_retains_historical_setup_for_other_contracts() -> None:
    result = build_rig(
        _request(A1RigSetupPoseMode.PRESERVE_COMPOSITION),
        A1RigProfile.TWO_AXIS_ROTATION_SCALE,
    )

    rotate_x, rotate_y, _scale, depth = result.transform
    assert rotate_x.extras["rotation"] == -134.67
    assert rotate_y.extras["rotation"] == -17.43
    assert depth.extras["x"] == -16.0
    assert depth.extras["scaleX"] == -1

    inverse_setup_bones = {
        result.profile.z_camera_setup_bone(result.info.prefix, group.index)
        for group in result.info.z_groups
    }
    assert inverse_setup_bones.isdisjoint({bone.name for bone in result.bones})
    result.validate()
