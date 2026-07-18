import ast
from pathlib import Path


ADAPTER = (
    Path(__file__).resolve().parents[1]
    / "Blender_to_Spine2D_Mesh_Exporter"
    / "blender_adapter"
)


def _tree(name: str) -> ast.Module:
    return ast.parse((ADAPTER / name).read_text(encoding="utf-8"), filename=name)


def _relative_imports(name: str) -> set[str]:
    return {
        node.module.rsplit(".", 1)[-1]
        for node in _tree(name).body
        if isinstance(node, ast.ImportFrom) and node.module
    }


def test_rna_module_is_only_a_compatibility_facade():
    tree = _tree("a1_ui_rna.py")
    assert not any(
        isinstance(node, (ast.FunctionDef, ast.ClassDef)) for node in tree.body
    )
    imports = _relative_imports("a1_ui_rna.py")
    assert {"a1_ui_selection", "a1_ui_scene_capture"}.issubset(imports)


def test_selection_and_scene_capture_are_physically_independent():
    selection = (ADAPTER / "a1_ui_selection.py").read_text(encoding="utf-8")
    scene = (ADAPTER / "a1_ui_scene_capture.py").read_text(encoding="utf-8")
    assert "a1_ui_scene_capture" not in selection
    assert "a1_ui_selection" not in scene
    for source in (selection, scene):
        assert "export_a1_single_object" not in source
        assert "export_a1_multi_object" not in source
        assert "export_a1_mixed_object" not in source


def test_runtime_modules_do_not_depend_on_compatibility_rna_facade():
    for name in ("a1_ui_settings.py", "a1_ui_router.py"):
        imports = _relative_imports(name)
        assert "a1_ui_rna" not in imports
        assert "a1_ui_selection" in imports
        assert "a1_ui_scene_capture" in imports


def test_selection_module_owns_object_profile_and_rna_identity():
    source = (ADAPTER / "a1_ui_selection.py").read_text(encoding="utf-8")
    assert "class _ObjectExportProfile" in source
    assert "def _rna_identity" in source
    assert "as_pointer" in source
    assert "def _ordered_selected_meshes" in source


def test_scene_module_owns_scene_profile_and_scene_property_capture():
    source = (ADAPTER / "a1_ui_scene_capture.py").read_text(encoding="utf-8")
    assert "class _SceneExportProfile" in source
    assert "def _capture_scene_profile" in source
    assert "spine2d_projection_alpha_threshold" in source
    assert "BakeExecutionSettings(" in source
