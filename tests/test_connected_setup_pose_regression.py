"""Connected setup regressions derived from real Spine 4.2 exports."""

from Blender_to_Spine2D_Mesh_Exporter.domain.spine import (
    ConnectedGroupSettings,
    LegacyRigProfile,
    TwoAxisScaleRigProfile,
    build_connected_group_document,
)

from test_connected_group_document import connected_objects, settings
from test_two_axis_connected_policy import _objects as two_axis_objects


def _bones(document):
    return {bone.name: bone for bone in document.bones}


def _constraint(document, name):
    return next(
        constraint
        for constraint in (*document.ik, *document.transform)
        if constraint.name == name
    )


def test_connected_two_axis_global_wrapper_is_neutral():
    result = build_connected_group_document(
        two_axis_objects(),
        ConnectedGroupSettings(100, 100, anchor_component_id="first"),
        profile=TwoAxisScaleRigProfile(),
    )
    bones = _bones(result.document)

    assert bones["all_objects_rotation_X"].rotation == 0.0
    assert bones["all_objects_rotation_Y"].rotation == 0.0


def test_connected_two_axis_scale_controls_follow_each_object_main():
    objects = two_axis_objects()
    result = build_connected_group_document(
        objects,
        ConnectedGroupSettings(100, 100, anchor_component_id="first"),
        profile=TwoAxisScaleRigProfile(),
    )
    composed = _bones(result.document)

    for item in objects:
        local = _bones(item.document)
        main_name = f"{item.prefix}_main"
        scale_name = f"{item.prefix}_scale"
        expected_x = round(local[scale_name].x - local[main_name].x, 2)
        expected_y = round(local[scale_name].y - local[main_name].y, 2)

        assert composed[scale_name].parent == main_name
        assert composed[scale_name].x == expected_x
        assert composed[scale_name].y == expected_y


def test_connected_three_axis_global_x_keeps_exact_main_payload():
    profile = LegacyRigProfile()
    result = build_connected_group_document(
        connected_objects(),
        settings(),
        profile=profile,
    )
    rotation_x = _constraint(
        result.document,
        profile.rotation_x_constraint("all_objects"),
    )

    assert rotation_x.bones == (
        "all_objects_0_scale",
        "all_objects_1_scale",
        "all_objects",
    )
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


def test_connected_three_axis_global_z_targets_object_base_bones():
    profile = LegacyRigProfile()
    result = build_connected_group_document(
        connected_objects(),
        settings(),
        profile=profile,
    )
    rotation_z = _constraint(
        result.document,
        profile.rotation_z_constraint("all_objects"),
    )

    assert rotation_z.bones == ("First", "Second", "Third")
    assert rotation_z.target == "all_objects_rotation_Z"
    assert rotation_z.extras == {
        "local": True,
        "mixX": 0,
        "mixScaleX": 0,
        "mixShearY": 0,
    }
