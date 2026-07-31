"""Exact two-axis connected global targets and layer-phase order contract."""

from Blender_to_Spine2D_Mesh_Exporter.domain.spine import (
    ConnectedGroupSettings,
    TwoAxisScaleRigProfile,
    build_connected_group_document,
)

from test_two_axis_connected_policy import _objects


def _constraint(document, name):
    return next(
        item
        for item in (*document.ik, *document.transform)
        if item.name == name
    )


def test_two_axis_connected_global_constraints_use_explicit_profile_targets():
    profile = TwoAxisScaleRigProfile()
    result = build_connected_group_document(
        _objects(),
        ConnectedGroupSettings(100, 100, anchor_component_id="first"),
        profile=profile,
    )
    prefix = "all_objects"

    rotation_x = _constraint(result.document, profile.rotation_x_constraint(prefix))
    assert rotation_x.bones == (
        "all_objects_rotate_X_constraint_rotate_IK",
        "all_objects_rotate_X",
    )
    assert rotation_x.target == "all_objects_rotation_X"
    assert rotation_x.extras == {
        "local": True,
        "relative": True,
        "mixX": 0,
        "mixScaleX": 0,
        "mixShearY": 0,
        "mixY": 0,
        "mixScaleY": 0,
    }

    scale = _constraint(result.document, profile.scale_constraint(prefix))
    assert scale.bones == (
        "all_objects_rotate_X",
        "all_objects_layer_0",
        "all_objects_layer_1",
    )
    assert scale.target == "all_objects_scale"
    assert scale.extras["relative"] is True
    assert scale.extras["mixRotate"] == 0
    assert scale.extras["mixX"] == 0
    assert scale.extras["mixY"] == 0
    assert scale.extras["mixShearY"] == 0

    depth = _constraint(result.document, profile.scale_depth_constraint(prefix))
    assert depth.bones == (
        "all_objects_0_scale",
        "all_objects_1_scale",
    )
    assert depth.target == "all_objects_rotate_X_constraint"

    rotation_y = _constraint(result.document, profile.rotation_y_constraint(prefix))
    assert rotation_y.bones == (
        "all_objects_layer_0",
        "all_objects_layer_1",
    )
    assert rotation_y.target == "all_objects_rotation_Y"
    assert rotation_y.extras == {
        "local": True,
        "relative": True,
        "mixX": 0,
        "mixScaleX": 0,
        "mixShearY": 0,
        "mixY": 0,
        "mixScaleY": 0,
    }


def test_two_axis_connected_orders_complete_global_rig_before_layer_phases():
    profile = TwoAxisScaleRigProfile()
    result = build_connected_group_document(
        _objects(),
        ConnectedGroupSettings(100, 100, anchor_component_id="first"),
        profile=profile,
    )
    schedule = result.constraint_schedule

    assert schedule.global_rotation_x == 0
    assert schedule.global_scale_ik == 1
    assert schedule.global_scale == 2
    assert schedule.global_scale_depth == 3
    assert schedule.global_rotation_y == 4

    # Assignment tuples preserve source-object order while each value follows Z layer.
    assert schedule.object_rotation_x == (("first", 6), ("second", 5))
    assert schedule.object_scale_ik == (("first", 8), ("second", 7))
    assert schedule.object_scale == (("first", 10), ("second", 9))
    assert schedule.object_scale_depth == (("first", 12), ("second", 11))
    assert schedule.object_rotation_y == (("first", 14), ("second", 13))
