"""Setup-pose regressions derived from real Spine 4.2 connected exports."""

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


def test_connected_three_axis_global_x_never_collapses_y_scale():
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

    assert "scaleY" not in rotation_x.extras
    assert rotation_x.extras["mixScaleY"] == 0


def test_connected_three_axis_global_z_uses_wrapper_layers():
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

    assert rotation_z.bones == (
        "all_objects_layer_0",
        "all_objects_layer_1",
    )
