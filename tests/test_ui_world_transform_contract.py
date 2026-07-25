from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "Blender_to_Spine2D_Mesh_Exporter"
UI_SOURCE = PACKAGE / "ui.py"
PREPARATION_SOURCE = PACKAGE / "blender_adapter" / "a1_source_geometry_preparation.py"
WORLD_TRANSFORM_SOURCE = PACKAGE / "domain" / "geometry" / "world_transform.py"


def test_ui_allows_non_singular_rotation_and_scale_normalization():
    ui = UI_SOURCE.read_text(encoding="utf-8")
    preparation = PREPARATION_SOURCE.read_text(encoding="utf-8")

    assert "_world_linear_transform_status" in ui
    assert "requires_normalization" in ui
    assert "normalize_mesh_snapshot_world_transform" in preparation
    assert '"object_linear_transform_baked"' in preparation
    assert "Scale is not applied (Apply > All Transforms)" not in ui


def test_ui_and_domain_use_relative_singular_transform_detection():
    ui = UI_SOURCE.read_text(encoding="utf-8")
    domain = WORLD_TRANSFORM_SOURCE.read_text(encoding="utf-8")
    preparation = PREPARATION_SOURCE.read_text(encoding="utf-8")

    assert "scale_product = first_length * second_length * third_length" in ui
    assert "relative_determinant <= tolerance" in ui
    assert "relative_determinant = _relative_determinant" in domain
    assert "Object world transform is singular or numerically unstable" in domain
    assert "orientation_sign = -1.0 if determinant < 0.0 else 1.0" in ui
    assert 'code="MIRRORED_OBJECT_TRANSFORM"' in preparation
