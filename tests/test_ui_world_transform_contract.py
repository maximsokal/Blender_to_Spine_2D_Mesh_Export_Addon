from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
UI_SOURCE = ROOT / "Blender_to_Spine2D_Mesh_Exporter" / "ui.py"


def test_ui_allows_non_singular_rotation_and_scale_normalization():
    source = UI_SOURCE.read_text(encoding="utf-8")

    assert "_world_linear_transform_status" in source
    assert "Rotation/scale will be normalized during export" in source
    assert "export_allowed = not is_singular" in source
    assert "Scale is not applied (Apply > All Transforms)" not in source


def test_ui_blocks_only_relative_singular_linear_object_transforms():
    source = UI_SOURCE.read_text(encoding="utf-8")

    assert "Object transform is singular" in source
    assert "scale_product = first_length * second_length * third_length" in source
    assert "relative_determinant <= tolerance" in source
    assert "Mirrored transform will preserve mirrored winding" in source
    assert "orientation_sign = -1.0 if determinant < 0.0 else 1.0" in source
