"""Regression contracts for camera-relative preprojected rig setup."""

from __future__ import annotations

import pytest

from Blender_to_Spine2D_Mesh_Exporter.domain.spine import (
    A1RigProfile,
    A1RigSetupPoseMode,
    LegacyRigBuildRequest,
    LegacyZGroup,
    LegacyZGroupOriginMode,
    SpineJsonTarget,
    build_rig,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.preprojected_setup import (
    ensure_preprojected_screen_rig,
)


def _request(
    setup_pose_mode: A1RigSetupPoseMode,
    *,
    z_groups: tuple[LegacyZGroup, ...] = (LegacyZGroup(-4.0),),
) -> LegacyRigBuildRequest:
    return LegacyRigBuildRequest(
        prefix="Projected",
        texture_width=160,
        texture_height=96,
        z_groups=z_groups,
        main_position_pixels=(13.0, -7.0),
        setup_pose_mode=setup_pose_mode,
        z_group_origin_mode=LegacyZGroupOriginMode.OBJECT_ORIGIN,
    )


def _two_axis(
    setup_pose_mode: A1RigSetupPoseMode,
    *,
    z_groups: tuple[LegacyZGroup, ...] = (LegacyZGroup(-4.0),),
):
    return build_rig(
        _request(setup_pose_mode, z_groups=z_groups),
        A1RigProfile.TWO_AXIS_ROTATION_SCALE,
        spine_target=SpineJsonTarget.SPINE_4_2,
    )


def _constraint_by_order(rig) -> dict[int, object]:
    return {constraint.order: constraint for constraint in rig.transform}


def _bone_by_name(rig, name: str):
    return next(bone for bone in rig.bones if bone.name == name)


def test_preprojected_setup_places_object_under_camera_zero() -> None:
    historical = _two_axis(A1RigSetupPoseMode.PRESERVE_COMPOSITION)
    screen = _two_axis(A1RigSetupPoseMode.PREPROJECTED_SCREEN)

    historical.validate()
    screen.validate()

    historical_main = _bone_by_name(historical, historical.info.main_bone_name)
    historical_base = _bone_by_name(historical, historical.info.base_bone_name)
    screen_main = _bone_by_name(screen, screen.info.main_bone_name)
    screen_base = _bone_by_name(screen, screen.info.base_bone_name)

    assert historical_main.x == 13.0
    assert historical_main.y == -7.0
    assert historical_base.x is None
    assert historical_base.y is None

    assert screen_main.x == 0.0
    assert screen_main.y == 0.0
    assert screen_base.x == 13.0
    assert screen_base.y == -7.0
    assert len(screen.info.z_groups) == 1


def test_preprojected_setup_preserves_live_constraint_topology() -> None:
    historical = _two_axis(A1RigSetupPoseMode.PRESERVE_COMPOSITION)
    screen = _two_axis(A1RigSetupPoseMode.PREPROJECTED_SCREEN)

    assert historical.ik == screen.ik
    assert tuple(item.order for item in screen.transform) == (0, 4, 2, 3)
    for historical_constraint, screen_constraint in zip(
        historical.transform,
        screen.transform,
        strict=True,
    ):
        assert historical_constraint.name == screen_constraint.name
        assert historical_constraint.order == screen_constraint.order
        assert historical_constraint.bones == screen_constraint.bones
        assert historical_constraint.target == screen_constraint.target


def test_preprojected_setup_neutralizes_only_initial_projection_offsets() -> None:
    historical = _constraint_by_order(
        _two_axis(A1RigSetupPoseMode.PRESERVE_COMPOSITION)
    )
    screen = _constraint_by_order(
        _two_axis(A1RigSetupPoseMode.PREPROJECTED_SCREEN)
    )

    assert historical[0].extras["rotation"] != 0.0
    assert historical[4].extras["rotation"] != 0.0
    assert historical[3].extras["scaleX"] == -1

    assert screen[0].extras["rotation"] == 0.0
    assert screen[4].extras["rotation"] == 0.0
    assert screen[3].extras["x"] == 0.0
    assert screen[3].extras["scaleX"] == 0.0

    assert screen[0].extras["scaleX"] == -1
    assert screen[0].extras["mixX"] == 0
    assert screen[0].extras["mixScaleX"] == 0
    assert screen[2].extras["relative"] is True
    assert screen[3].extras["rotation"] == -90
    assert screen[4].extras["relative"] is True


def test_ensure_preprojected_screen_rig_is_idempotent_and_immutable() -> None:
    source = _two_axis(A1RigSetupPoseMode.PRESERVE_COMPOSITION)
    source_request = source.request

    result = ensure_preprojected_screen_rig(source)

    assert result is not source
    assert source.request is source_request
    assert source.request.setup_pose_mode is A1RigSetupPoseMode.PRESERVE_COMPOSITION
    assert result.request.setup_pose_mode is A1RigSetupPoseMode.PREPROJECTED_SCREEN
    assert result.request.z_group_origin_mode is LegacyZGroupOriginMode.OBJECT_ORIGIN
    assert ensure_preprojected_screen_rig(result) is result


def test_helper_can_collapse_rendered_source_depths_to_camera_origin() -> None:
    source = _two_axis(
        A1RigSetupPoseMode.PRESERVE_COMPOSITION,
        z_groups=(LegacyZGroup(-1.0), LegacyZGroup(0.0), LegacyZGroup(2.0)),
    )

    result = ensure_preprojected_screen_rig(
        source,
        main_position_pixels=(21.0, 9.0),
        camera_depth=-6.5,
    )

    assert result.request.z_groups == (LegacyZGroup(-6.5),)
    assert result.request.main_position_pixels == (21.0, 9.0)
    assert len(result.info.z_groups) == 1
    assert result.info.z_groups[0].z_value == -6.5
    result.validate()


def test_helper_rejects_uncollapsed_vertex_depth_groups() -> None:
    source = _two_axis(
        A1RigSetupPoseMode.PRESERVE_COMPOSITION,
        z_groups=(LegacyZGroup(-1.0), LegacyZGroup(1.0)),
    )

    with pytest.raises(ValueError, match="exactly one Object-Origin depth group"):
        ensure_preprojected_screen_rig(source)


def test_preprojected_screen_rejects_unsupported_three_axis_profile() -> None:
    legacy = build_rig(
        _request(A1RigSetupPoseMode.PRESERVE_COMPOSITION),
        A1RigProfile.THREE_AXIS_ROTATION,
        spine_target=SpineJsonTarget.SPINE_4_2,
    )

    with pytest.raises(
        ValueError,
        match="requires TWO_AXIS_ROTATION_SCALE",
    ):
        ensure_preprojected_screen_rig(legacy)
