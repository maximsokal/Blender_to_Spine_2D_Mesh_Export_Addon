import ast
from pathlib import Path


ADAPTER = (
    Path(__file__).resolve().parents[1]
    / "Blender_to_Spine2D_Mesh_Exporter"
    / "blender_adapter"
)


def _tree(filename: str) -> ast.Module:
    path = ADAPTER / filename
    return ast.parse(path.read_text(encoding="utf-8"), filename=filename)


def _function_lengths(filename: str) -> dict[str, int]:
    tree = _tree(filename)
    return {
        node.name: node.end_lineno - node.lineno + 1
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }


def test_ui_bridge_is_a_small_compatibility_facade():
    path = ADAPTER / "a1_ui_bridge.py"
    tree = _tree(path.name)

    assert len(path.read_text(encoding="utf-8").splitlines()) < 60
    assert not any(isinstance(node, ast.FunctionDef) for node in tree.body)
    imported_modules = {
        node.module
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert {
        "a1_ui_rna",
        "a1_ui_router",
        "a1_ui_settings",
    }.issubset({module.rsplit(".", 1)[-1] for module in imported_modules})


def test_ui_responsibilities_are_split_without_new_monolithic_functions():
    limits = {
        "a1_ui_selection.py": 60,
        "a1_ui_scene_capture.py": 60,
        "a1_ui_settings.py": 60,
        "a1_ui_router.py": 100,
    }
    for filename, maximum in limits.items():
        lengths = _function_lengths(filename)
        assert lengths
        assert max(lengths.values()) < maximum, (filename, lengths)

    rna_tree = _tree("a1_ui_rna.py")
    assert not any(
        isinstance(node, (ast.FunctionDef, ast.ClassDef))
        for node in rna_tree.body
    )


def test_capture_and_settings_modules_do_not_call_output_services():
    forbidden = {
        "export_a1_single_object",
        "export_a1_multi_object",
        "export_a1_mixed_object",
    }
    for filename in (
        "a1_ui_selection.py",
        "a1_ui_scene_capture.py",
        "a1_ui_settings.py",
    ):
        tree = _tree(filename)
        called = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert not called.intersection(forbidden), filename


def test_runtime_ui_modules_do_not_import_the_rna_compatibility_facade():
    for filename in ("a1_ui_settings.py", "a1_ui_router.py"):
        imported_modules = {
            node.module.rsplit(".", 1)[-1]
            for node in _tree(filename).body
            if isinstance(node, ast.ImportFrom) and node.module
        }
        assert "a1_ui_rna" not in imported_modules
        assert "a1_ui_selection" in imported_modules
        assert "a1_ui_scene_capture" in imported_modules


def test_legacy_private_bridge_helpers_remain_reexported():
    source = (ADAPTER / "a1_ui_bridge.py").read_text(encoding="utf-8")
    for name in (
        "_build_sources",
        "_common_object_settings",
        "_resolve_geometry_settings",
        "_ordered_selected_meshes",
    ):
        assert name in source
