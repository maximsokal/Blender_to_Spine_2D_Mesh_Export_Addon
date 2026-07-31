"""Target-aware connected constraint scheduling contracts for Spine 4.1 research."""

from __future__ import annotations

import pytest

from Blender_to_Spine2D_Mesh_Exporter.domain.spine.connected_group_contracts import (
    ConnectedObjectPlacement,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.connected_group_schedule import (
    build_constraint_schedule,
    validate_constraint_schedule_for_target,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.legacy_profile import LegacyRigProfile
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.two_axis_scale_profile import (
    TwoAxisScaleRigProfile,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.version_target import SpineJsonTarget


def _placements() -> tuple[ConnectedObjectPlacement, ...]:
    return (
        ConnectedObjectPlacement(
            component_id="cone-a",
            prefix="Cone",
            relative_x=0.0,
            relative_y=0.0,
            relative_z=0.0,
            layer_index=0,
            main_bone_name="Cone_main",
            parent_layer_bone_name="group_layer_0",
        ),
        ConnectedObjectPlacement(
            component_id="cone-b",
            prefix="Cone.001",
            relative_x=1.0,
            relative_y=2.0,
            relative_z=0.0,
            layer_index=0,
            main_bone_name="Cone.001_main",
            parent_layer_bone_name="group_layer_0",
        ),
        ConnectedObjectPlacement(
            component_id="cone-c",
            prefix="Cone.002",
            relative_x=3.0,
            relative_y=4.0,
            relative_z=1.0,
            layer_index=1,
            main_bone_name="Cone.002_main",
            parent_layer_bone_name="group_layer_1",
        ),
    )


def test_spine_four_two_completes_global_rig_then_shares_same_layer_orders() -> None:
    schedule = build_constraint_schedule(
        _placements(),
        TwoAxisScaleRigProfile(),
        spine_target=SpineJsonTarget.SPINE_4_2,
    )

    assert len(schedule.all_orders) == 20
    assert len(schedule.unique_orders) == 15
    assert schedule.unique_orders == tuple(range(15))
    assert (
        schedule.global_rotation_x,
        schedule.global_scale_ik,
        schedule.global_scale,
        schedule.global_scale_depth,
        schedule.global_rotation_y,
    ) == (0, 1, 2, 3, 4)
    assert schedule.order_for("object_rotation_x", "cone-a") == 5
    assert schedule.order_for("object_rotation_x", "cone-b") == 5
    assert schedule.order_for("object_rotation_x", "cone-c") == 6


def test_spine_four_one_assigns_one_global_order_per_constraint() -> None:
    schedule = build_constraint_schedule(
        _placements(),
        TwoAxisScaleRigProfile(),
        spine_target=SpineJsonTarget.SPINE_4_1,
    )

    assert len(schedule.all_orders) == 20
    assert len(schedule.unique_orders) == 20
    assert tuple(sorted(schedule.all_orders)) == tuple(range(20))
    assert schedule.order_for("object_rotation_x", "cone-a") == 1
    assert schedule.order_for("object_rotation_x", "cone-b") == 2
    assert schedule.order_for("object_rotation_x", "cone-c") == 3
    assert validate_constraint_schedule_for_target(
        schedule,
        SpineJsonTarget.SPINE_4_1,
    ) is schedule


def test_spine_four_one_three_axis_fails_before_guessing_compensator_order() -> None:
    with pytest.raises(ValueError, match="scale compensator"):
        build_constraint_schedule(
            _placements(),
            LegacyRigProfile(),
            spine_target=SpineJsonTarget.SPINE_4_1,
        )


def test_unimplemented_connected_target_fails_closed() -> None:
    with pytest.raises(ValueError, match="not implemented"):
        build_constraint_schedule(
            _placements(),
            TwoAxisScaleRigProfile(),
            spine_target=SpineJsonTarget.SPINE_4_0,
        )
