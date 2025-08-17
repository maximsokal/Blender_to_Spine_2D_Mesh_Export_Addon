from Blender_to_Spine2D_Mesh_Exporter import json_export


def _scale_bones(bones):
    """Return only the scale bones from process_z_groups output."""
    return [b for b in bones if b["name"].endswith("_scale")]


def test_process_z_groups_height_real_pixels():
    z_groups = [[0, 0, 0.1], [0, 0, 0.2]]
    z_info = {0.1: {"height_real_pixels": 5.0}, 0.2: {"height_real_pixels": 10.0}}
    result = json_export.process_z_groups(
        "att",
        z_groups,
        [],
        "parent",
        half_scale=1.0,
        z_groups_info=z_info,
        texture_width=64,
        texture_height=64,
    )
    scales = _scale_bones(result["bones"])
    assert [b["y"] for b in scales] == [5.0, 10.0]
    assert result["bones_info"]["z_value_to_bone_name"][0.1] == "att_1"
    assert result["bones_info"]["z_value_to_bone_name"][0.2] == "att_2"


def test_process_z_groups_direct_scaling():
    original = [0.0, 1.0, 2.0]
    result = json_export.process_z_groups(
        "att",
        [],
        original,
        "parent",
        half_scale=1.0,
        texture_width=100,
        texture_height=200,
    )
    scales = _scale_bones(result["bones"])
    assert [b["y"] for b in scales] == [0.0, 150.0, 300.0]
    assert result["bones_info"]["z_value_to_bone_name"][1.0] == "att_2"


def test_process_z_groups_legacy_fallback():
    z_groups = [[0, 0, 0.5], [0, 0, 0.7]]
    result = json_export.process_z_groups(
        "att",
        z_groups,
        [],
        "parent",
        half_scale=1.0,
    )
    scales = _scale_bones(result["bones"])
    assert [b["y"] for b in scales] == [0.0, -1.0]
    assert result["bones_info"]["z_group_z_values"] == [0.5, 0.7]
