"""Protect connected two-axis composition when all objects share one Blender Z layer."""

from __future__ import annotations

from Blender_to_Spine2D_Mesh_Exporter.domain.spine import (
    ConnectedGroupSettings,
    ConnectedObjectDocument,
    SpineValidator,
    TwoAxisScaleRigProfile,
    build_connected_group_document,
)

from test_two_axis_connected_policy import _two_axis_document


def test_three_connected_two_axis_objects_share_one_layer_and_keep_controls():
    objects = (
        ConnectedObjectDocument(
            component_id="first",
            prefix="First",
            document=_two_axis_document("First", (0.0, 0.0)),
            world_position=(0.0, 0.0, 4.0),
        ),
        ConnectedObjectDocument(
            component_id="second",
            prefix="Second",
            document=_two_axis_document("Second", (0.0, 0.0)),
            world_position=(0.0, 2.0, 4.0),
        ),
        ConnectedObjectDocument(
            component_id="third",
            prefix="Third",
            document=_two_axis_document("Third", (0.0, 0.0)),
            world_position=(0.0, -3.0, 4.0),
        ),
    )
    result = build_connected_group_document(
        objects,
        ConnectedGroupSettings(
            texture_width=100,
            texture_height=100,
            anchor_component_id="first",
        ),
        profile=TwoAxisScaleRigProfile(),
    )

    SpineValidator().validate_or_raise(result.document)
    assert len(result.layers) == 1
    assert result.layers[0].component_ids == ("first", "second", "third")

    bones = {bone.name: bone for bone in result.document.bones}
    for prefix in ("First", "Second", "Third"):
        assert bones[f"{prefix}_main"].parent == "all_objects_layer_0"
        assert f"{prefix}_rotation_X" in bones
        assert f"{prefix}_rotation_Y" in bones
        assert f"{prefix}_scale" in bones
        assert f"{prefix}_rotation_Z" not in bones

    assert (bones["First_main"].x, bones["First_main"].y) == (0.0, 0.0)
    assert (bones["Second_main"].x, bones["Second_main"].y) == (0.0, 200.0)
    assert (bones["Third_main"].x, bones["Third_main"].y) == (0.0, -300.0)

    constraints = (*result.document.ik, *result.document.transform)
    orders = tuple(constraint.order for constraint in constraints)
    assert len(orders) == 20
    assert len(set(orders)) == 20
    assert set(orders) == set(range(20))
