from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "Blender_to_Spine2D_Mesh_Exporter"


def test_single_operator_routes_directly_to_rewrite_backend():
    source = (PACKAGE / "single_object_operator.py").read_text(encoding="utf-8")

    assert "export_active_object_a1" in source
    assert "legacy" not in source.casefold()
    assert "DEFAULT_SINGLE_BACKEND" not in source
    assert "SINGLE_BACKEND_PROPERTY" not in source


def test_multi_ui_routes_directly_to_rewrite_backends():
    source = (PACKAGE / "ui.py").read_text(encoding="utf-8")

    assert "export_active_object_a1" in source
    assert "export_selected_objects_a1" in source
    assert "DEFAULT_MULTI_BACKEND" not in source
    assert "MULTI_BACKEND_PROPERTY" not in source
