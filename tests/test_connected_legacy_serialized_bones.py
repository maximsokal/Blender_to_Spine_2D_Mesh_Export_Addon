"""Exact serialized helper-bone fields from historical main._mk_bone."""

from Blender_to_Spine2D_Mesh_Exporter.domain.spine import (
    SpineSerializer,
    build_connected_group_document,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.connected_group_serialization_validator import (
    ConnectedGroupSerializationValidator,
)

from test_connected_group_document import connected_objects, settings


def test_connected_legacy_generated_bones_include_main_zero_defaults():
    result = build_connected_group_document(connected_objects(), settings())
    serialized = SpineSerializer(
        validator=ConnectedGroupSerializationValidator()
    ).to_dict(result.document)
    bones = {item["name"]: item for item in serialized["bones"]}

    assert bones["all_objects_main"] == {
        "name": "all_objects_main",
        "parent": "root",
        "length": 50.0,
        "x": 0.0,
        "y": 0.0,
    }
    assert bones["all_objects"] == {
        "name": "all_objects",
        "parent": "all_objects_main",
        "length": 0.0,
        "x": 0.0,
        "y": 0.0,
    }
    assert bones["all_objects_scale_rotate_X"] == {
        "name": "all_objects_scale_rotate_X",
        "parent": "all_objects",
        "length": 50.0,
        "x": 0.0,
        "y": 0.0,
    }
    assert bones["all_objects_rotate_X_constraint_scale_IK"] == {
        "name": "all_objects_rotate_X_constraint_scale_IK",
        "parent": "all_objects",
        "length": 0.0,
        "x": 0.0,
        "y": 0.0,
        "rotation": -90.0,
    }
    for name, parent in (
        ("all_objects_0_scale", "all_objects_rotate_X"),
        ("all_objects_1_scale", "all_objects_rotate_X"),
        ("all_objects_layer_0", "all_objects_0_scale"),
        ("all_objects_layer_1", "all_objects_1_scale"),
    ):
        assert bones[name] == {
            "name": name,
            "parent": parent,
            "length": 5.0,
            "x": 0.0,
            "y": 0.0,
        }
