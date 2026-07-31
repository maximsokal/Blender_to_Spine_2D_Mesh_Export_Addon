from __future__ import annotations

from Blender_to_Spine2D_Mesh_Exporter.domain.spine.connected_group_contracts import (
    ConnectedObjectPlacement,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.connected_group_schedule import (
    build_constraint_schedule,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.two_axis_scale_profile import (
    TwoAxisScaleRigProfile,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.version_target import SpineJsonTarget


def _placement(component_id: str, prefix: str, layer_index: int) -> ConnectedObjectPlacement:
    return ConnectedObjectPlacement(
        component_id=component_id,
        prefix=prefix,
        relative_x=0.0,
        relative_y=0.0,
        relative_z=float(layer_index),
        layer_index=layer_index,
        main_bone_name=f"{prefix}_main",
        parent_layer_bone_name=f"all_objects_layer_{layer_index}",
    )


def test_spine42_two_axis_schedule_completes_global_wrapper_before_objects() -> None:
    placements = (
        _placement("component_a", "ObjectA", 1),
        _placement("component_b", "ObjectB", 0),
        _placement("component_c", "ObjectC", 1),
    )

    schedule = build_constraint_schedule(
        placements,
        TwoAxisScaleRigProfile(),
        spine_target=SpineJsonTarget.SPINE_4_2,
    )

    assert (
        schedule.global_rotation_x,
        schedule.global_scale_ik,
        schedule.global_scale,
        schedule.global_scale_depth,
        schedule.global_rotation_y,
    ) == (0, 1, 2, 3, 4)

    first_object_order = min(
        order
        for assignments in (
            schedule.object_rotation_x,
            schedule.object_scale_ik,
            schedule.object_scale,
            schedule.object_scale_depth,
            schedule.object_rotation_y,
        )
        for _component_id, order in assignments
    )
    assert first_object_order == 5
    assert schedule.global_rotation_y < first_object_order

    for component_id in ("component_a", "component_b", "component_c"):
        assert (
            schedule.order_for("object_rotation_x", component_id)
            < schedule.order_for("object_scale_ik", component_id)
            < schedule.order_for("object_scale", component_id)
            < schedule.order_for("object_scale_depth", component_id)
            < schedule.order_for("object_rotation_y", component_id)
        )

    # Independent objects in the same connected Z layer retain their historical tie.
    assert schedule.order_for("object_rotation_x", "component_a") == 6
    assert schedule.order_for("object_rotation_x", "component_c") == 6
    assert schedule.order_for("object_rotation_x", "component_b") == 5
    assert schedule.unique_orders == tuple(range(15))


def test_spine41_two_axis_schedule_keeps_unique_phase_interleaving() -> None:
    placements = (
        _placement("component_a", "ObjectA", 0),
        _placement("component_b", "ObjectB", 0),
    )

    schedule = build_constraint_schedule(
        placements,
        TwoAxisScaleRigProfile(),
        spine_target=SpineJsonTarget.SPINE_4_1,
    )

    assert schedule.global_rotation_x == 0
    assert dict(schedule.object_rotation_x) == {
        "component_a": 1,
        "component_b": 2,
    }
    assert schedule.global_scale_ik == 3
    assert len(schedule.all_orders) == len(set(schedule.all_orders))
    assert tuple(sorted(schedule.all_orders)) == tuple(range(len(schedule.all_orders)))
