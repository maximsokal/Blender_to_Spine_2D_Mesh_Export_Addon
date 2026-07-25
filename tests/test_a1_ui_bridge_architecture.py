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


def _relative_imports(filename: str) -> set[str]:
    return {
        node.module.rsplit(".", 1)[-1]
        for node in _tree(filename).body
        if isinstance(node, ast.ImportFrom) and node.module
    }


def _function_lengths(filename: str) -> dict[str, int]:
    return {
        node.name: node.end_lineno - node.lineno + 1
        for node in _tree(filename).body
        if isinstance(node, ast.FunctionDef)
    }


def test_ui_bridge_is_a_small_definition_free_public_boundary():
    path = ADAPTER / "a1_ui_bridge.py"
    tree = _tree(path.name)

    assert len(path.read_text(encoding="utf-8").splitlines()) < 60
    assert not any(isinstance(node, (ast.FunctionDef, ast.ClassDef)) for node in tree.body)
    assert {
        "a1_ui_router",
        "a1_ui_scene_capture",
        "a1_ui_selection",
        "a1_ui_settings",
    }.issubset(_relative_imports(path.name))
    assert not (ADAPTER / "a1_ui_rna.py").exists()


def test_ui_responsibilities_remain_split_without_monolithic_functions():
    limits = {
        "a1_ui_selection.py": 60,
        "a1_ui_scene_capture.py": 100,
        "a1_ui_settings.py": 60,
        "a1_ui_export_plan.py": 60,
        "a1_ui_router.py": 60,
    }
    for filename, maximum in limits.items():
        lengths = _function_lengths(filename)
        assert lengths
        assert max(lengths.values()) < maximum, (filename, lengths)


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
        "a1_ui_export_plan.py",
    ):
        called = {
            node.func.id
            for node in ast.walk(_tree(filename))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert not called.intersection(forbidden), filename


def test_runtime_ui_dependency_direction_is_one_way():
    assert {"a1_ui_selection", "a1_ui_scene_capture"}.issubset(
        _relative_imports("a1_ui_settings.py")
    )
    assert {
        "a1_ui_selection",
        "a1_ui_scene_capture",
        "a1_ui_settings",
    }.issubset(_relative_imports("a1_ui_export_plan.py"))
    router_imports = _relative_imports("a1_ui_router.py")
    assert "a1_ui_export_plan" in router_imports
    assert "a1_ui_selection" not in router_imports
    assert "a1_ui_scene_capture" not in router_imports


def test_private_compatibility_helpers_are_reexported_only_by_bridge():
    source = (ADAPTER / "a1_ui_bridge.py").read_text(encoding="utf-8")
    for name in (
        "_build_sources",
        "_common_object_settings",
        "_resolve_geometry_settings",
        "_ordered_selected_meshes",
    ):
        assert name in source
