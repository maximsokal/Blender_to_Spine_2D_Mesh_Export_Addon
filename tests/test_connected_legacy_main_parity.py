"""Exact connected 3-axis parity with the historical ``main`` implementation."""

from Blender_to_Spine2D_Mesh_Exporter.domain.spine import (
    LegacyRigProfile,
    build_connected_group_document,
)

from test_connected_group_document import connected_objects, settings


def _bone(document, name):
    return next(item for item in document.bones if item.name == name)


def _constraint(document, name):
    return next(
        item
        for item in (*document.ik, *document.transform)
        if item.name == name
    )


def test_connected_three_axis_global_wrapper_matches_main_hierarchy():
    profile = LegacyRigProfile()
    result = build_connected_group_document(
        connected_objects(),
        settings(),
        profile=profile,
    )
    prefix = "all_objects"

    assert _bone(result.document, profile.main_bone(prefix)).parent == "root"
    assert _bone(result.document, profile.scale_rotate_x_bone(prefix)).parent == prefix
    assert _bone(result.document, profile.rotate_x_bone(prefix)).parent == (
        profile.scale_rotate_x_bone(prefix)
    )

    for control in profile.control_bones(prefix):
        assert _bone(result.document, control).parent == "root"

    for layer in result.layers:
        scale = _bone(result.document, layer.scale_bone_name)
        wrapper = _bone(result.document, layer.layer_bone_name)
        assert scale.parent == profile.rotate_x_bone(prefix)
        assert scale.y in (None, 0.0)
        assert scale.rotation in (None, 0.0)
        assert scale.extras == {}
        assert wrapper.parent == layer.scale_bone_name
        assert wrapper.y in (None, 0.0)
        assert wrapper.rotation in (None, 0.0)
        assert wrapper.extras == {}


def test_connected_three_axis_global_constraints_match_main_payload():
    profile = LegacyRigProfile()
    objects = connected_objects()
    result = build_connected_group_document(objects, settings(), profile=profile)
    prefix = "all_objects"

    rotation_x = _constraint(result.document, profile.rotation_x_constraint(prefix))
    assert rotation_x.order == 0
    assert rotation_x.bones == (
        "all_objects_0_scale",
        "all_objects_1_scale",
        "all_objects",
    )
    assert rotation_x.target == "all_objects_rotation_X"
    assert rotation_x.extras == {
        "rotation": 90,
        "local": True,
        "relative": True,
        "x": -200.0,
        "y": -50.0,
        "scaleX": -1,
        "scaleY": -1,
        "mixX": 0,
        "mixScaleX": 0,
        "mixShearY": 0,
    }

    rotation_y = _constraint(result.document, profile.rotation_y_constraint(prefix))
    assert rotation_y.order == 1
    assert rotation_y.bones == (
        "all_objects_rotate_X",
        "all_objects_rotate_X_constraint_rotate_IK",
    )
    assert rotation_y.target == "all_objects_rotation_Y"
    assert rotation_y.extras == {
        "local": True,
        "relative": True,
        "x": 100.0,
        "scaleX": -1,
        "mixX": 0,
        "mixScaleX": 0,
        "mixShearY": 0,
    }

    rotation_z = _constraint(result.document, profile.rotation_z_constraint(prefix))
    assert rotation_z.order == 2
    assert rotation_z.bones == tuple(item.prefix for item in objects)
    assert rotation_z.target == "all_objects_rotation_Z"
    assert rotation_z.extras == {
        "local": True,
        "mixX": 0,
        "mixScaleX": 0,
        "mixShearY": 0,
    }

    scale = _constraint(result.document, profile.scale_constraint(prefix))
    assert scale.order == 10
    assert scale.bones == ("all_objects_0_scale", "all_objects_1_scale")
    assert scale.target == "all_objects_rotate_X_constraint"
    assert scale.extras == {
        "scaleX": -1,
        "mixX": 0,
        "mixScaleX": 0,
        "mixShearY": 0,
    }


def test_connected_three_axis_orders_are_grouped_by_z_layer_like_main():
    objects = connected_objects()
    result = build_connected_group_document(objects, settings())
    schedule = result.constraint_schedule

    assert schedule.object_rotation_x == (
        ("first", 4),
        ("second", 3),
        ("third", 4),
    )
    assert schedule.object_rotation_y == (
        ("first", 6),
        ("second", 5),
        ("third", 6),
    )
    assert schedule.global_scale_ik == 7
    assert schedule.object_scale_ik == (
        ("first", 9),
        ("second", 8),
        ("third", 9),
    )
    assert schedule.global_scale == 10
    assert schedule.object_scale == (
        ("first", 12),
        ("second", 11),
        ("third", 12),
    )
    assert schedule.object_rotation_z == (
        ("first", 14),
        ("second", 13),
        ("third", 14),
    )
    assert schedule.unique_orders == tuple(range(15))
    assert len(schedule.all_orders) > len(schedule.unique_orders)

    for item in objects:
        compensator = _constraint(
            result.document,
            f"{item.prefix}_scale_compensator",
        )
        assert compensator.order == 6
