"""Runtime-level invariants for connected Spine 4.2 constraints."""

from Blender_to_Spine2D_Mesh_Exporter.domain.spine import (
    ConnectedGroupSettings,
    LegacyRigProfile,
    TwoAxisScaleRigProfile,
    build_connected_group_document,
)

from test_connected_group_document import connected_objects, settings
from test_two_axis_connected_policy import _objects as two_axis_objects


def _bone(document, name):
    return next(item for item in document.bones if item.name == name)


def _constraint(document, name):
    return next(item for item in document.transform if item.name == name)


def _number(value):
    return 0.0 if value is None else float(value)


def _relative_local_setup_delta(document, constraint_name):
    constraint = _constraint(document, constraint_name)
    assert constraint.extras.get("local", False) is True
    assert constraint.extras.get("relative", False) is True
    target = _bone(document, constraint.target)
    extras = constraint.extras

    mix_x = float(extras.get("mixX", 1.0))
    mix_y = float(extras.get("mixY", mix_x))
    mix_scale_x = float(extras.get("mixScaleX", 1.0))
    mix_scale_y = float(extras.get("mixScaleY", mix_scale_x))
    mix_shear_y = float(extras.get("mixShearY", 1.0))
    return {
        "rotation": (
            _number(target.rotation) + float(extras.get("rotation", 0.0))
        )
        * float(extras.get("mixRotate", 1.0)),
        "x": (_number(target.x) + float(extras.get("x", 0.0))) * mix_x,
        "y": (_number(target.y) + float(extras.get("y", 0.0))) * mix_y,
        "scale_x": (
            _number(target.scale_x or 1.0)
            - 1.0
            + float(extras.get("scaleX", 0.0))
        )
        * mix_scale_x,
        "scale_y": (
            _number(target.scale_y or 1.0)
            - 1.0
            + float(extras.get("scaleY", 0.0))
        )
        * mix_scale_y,
        "shear_y": float(extras.get("shearY", 0.0)) * mix_shear_y,
    }


def test_connected_two_axis_global_rotation_constraints_have_zero_setup_delta():
    profile = TwoAxisScaleRigProfile()
    result = build_connected_group_document(
        two_axis_objects(),
        ConnectedGroupSettings(100, 100, anchor_component_id="first"),
        profile=profile,
    )

    for name in (
        profile.rotation_x_constraint("all_objects"),
        profile.rotation_y_constraint("all_objects"),
    ):
        assert _relative_local_setup_delta(result.document, name) == {
            "rotation": 0.0,
            "x": 0.0,
            "y": 0.0,
            "scale_x": 0.0,
            "scale_y": 0.0,
            "shear_y": 0.0,
        }


def test_connected_three_axis_global_constraints_match_main_runtime_channels():
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
    assert rotation_x.extras["rotation"] == 90
    assert rotation_x.extras["scaleX"] == -1
    assert rotation_x.extras["scaleY"] == -1
    assert "mixY" not in rotation_x.extras
    assert "mixScaleY" not in rotation_x.extras

    rotation_y = _constraint(
        result.document,
        profile.rotation_y_constraint("all_objects"),
    )
    assert rotation_y.extras["x"] == result.uniform_scale
    assert rotation_y.extras["scaleX"] == -1

    rotation_z = _constraint(
        result.document,
        profile.rotation_z_constraint("all_objects"),
    )
    assert rotation_z.extras.get("relative") is None
    assert rotation_z.bones == ("First", "Second", "Third")


def test_connected_three_axis_global_scale_keeps_main_channels():
    profile = LegacyRigProfile()
    result = build_connected_group_document(
        connected_objects(),
        settings(),
        profile=profile,
    )
    scale = _constraint(result.document, profile.scale_constraint("all_objects"))

    assert scale.extras == {
        "scaleX": -1,
        "mixX": 0,
        "mixScaleX": 0,
        "mixShearY": 0,
    }


def test_connected_legacy_layers_are_neutral_and_main_keeps_full_xy():
    objects = connected_objects()
    result = build_connected_group_document(objects, settings())
    object_by_component = {item.component_id: item for item in objects}

    for layer in result.layers:
        scale = _bone(result.document, layer.scale_bone_name)
        wrapper = _bone(result.document, layer.layer_bone_name)
        assert _number(scale.x) == 0.0
        assert _number(scale.y) == 0.0
        assert _number(scale.rotation) == 0.0
        assert _number(wrapper.x) == 0.0
        assert _number(wrapper.y) == 0.0
        assert _number(wrapper.rotation) == 0.0

    for placement in result.placements:
        source = object_by_component[placement.component_id]
        source_main = _bone(source.document, placement.main_bone_name)
        composed_main = _bone(result.document, placement.main_bone_name)
        expected_x = round(
            _number(source_main.x) + placement.relative_x * result.uniform_scale,
            2,
        )
        expected_y = round(
            _number(source_main.y) + placement.relative_y * result.uniform_scale,
            2,
        )
        assert composed_main.x == expected_x
        assert composed_main.y == expected_y
