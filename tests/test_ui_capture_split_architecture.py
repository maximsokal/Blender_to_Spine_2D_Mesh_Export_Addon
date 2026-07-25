import ast
from pathlib import Path


ADAPTER = (
    Path(__file__).resolve().parents[1]
    / "Blender_to_Spine2D_Mesh_Exporter"
    / "blender_adapter"
)


def _source(name: str) -> str:
    return (ADAPTER / name).read_text(encoding="utf-8")


def _tree(name: str) -> ast.Module:
    return ast.parse(_source(name), filename=name)


def _relative_imports(name: str) -> set[str]:
    return {
        node.module.rsplit(".", 1)[-1]
        for node in _tree(name).body
        if isinstance(node, ast.ImportFrom) and node.module
    }


def test_retired_rna_compatibility_module_is_absent():
    assert not (ADAPTER / "a1_ui_rna.py").exists()
    for name in ADAPTER.glob("a1_ui_*.py"):
        assert "a1_ui_rna" not in _source(name.name)


def test_selection_and_scene_capture_are_physically_independent():
    selection = _source("a1_ui_selection.py")
    scene = _source("a1_ui_scene_capture.py")
    assert "a1_ui_scene_capture" not in selection
    assert "a1_ui_selection" not in scene
    for source in (selection, scene):
        assert "export_a1_single_object" not in source
        assert "export_a1_multi_object" not in source
        assert "export_a1_mixed_object" not in source


def test_settings_and_export_plan_depend_on_physical_capture_owners():
    settings_imports = _relative_imports("a1_ui_settings.py")
    assert {"a1_ui_selection", "a1_ui_scene_capture"}.issubset(settings_imports)

    plan_imports = _relative_imports("a1_ui_export_plan.py")
    assert {
        "a1_ui_selection",
        "a1_ui_scene_capture",
        "a1_ui_settings",
    }.issubset(plan_imports)


def test_selection_module_owns_object_profile_and_rna_identity():
    source = _source("a1_ui_selection.py")
    assert "class _ObjectExportProfile" in source
    assert "def _rna_identity" in source
    assert "as_pointer" in source
    assert "def _ordered_selected_meshes" in source


def test_scene_module_owns_scene_profile_and_scene_property_capture():
    source = _source("a1_ui_scene_capture.py")
    assert "class _SceneExportProfile" in source
    assert "def _capture_scene_profile" in source
    assert "spine2d_projection_alpha_threshold" in source
    assert "BakeExecutionSettings(" in source
