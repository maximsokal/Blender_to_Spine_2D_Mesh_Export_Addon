import ast
from pathlib import Path


ADAPTER = (
    Path(__file__).resolve().parents[1]
    / "Blender_to_Spine2D_Mesh_Exporter"
    / "blender_adapter"
)


def _tree(name: str) -> ast.Module:
    return ast.parse((ADAPTER / name).read_text(encoding="utf-8"), filename=name)


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    return next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def test_public_orchestrator_is_short_and_calls_typed_stages_in_order():
    tree = _tree("a1_object_preparation.py")
    function = _function(tree, "prepare_a1_object")
    assert function.end_lineno - function.lineno + 1 < 85
    calls = [
        node.func.id
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id.startswith("prepare_a1_")
    ]
    assert calls[:4] == [
        "prepare_a1_source_geometry",
        "prepare_a1_uv",
        "prepare_a1_texture_plan",
        "prepare_a1_document",
    ]


def test_orchestrator_has_no_low_level_blender_preparation_dependencies():
    tree = _tree("a1_object_preparation.py")
    imported = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    forbidden_suffixes = {
        "evaluated_mesh_reader",
        "mesh_reader",
        "uv_unwrap",
        "material_analyzer",
        "scene_bake_analyzer",
        "production_shader_capabilities",
    }
    assert not any(
        module.rsplit(".", 1)[-1] in forbidden_suffixes
        for module in imported
    )


def test_each_stage_function_stays_below_monolith_threshold():
    stages = {
        "a1_source_geometry_preparation.py": "prepare_a1_source_geometry",
        "a1_uv_preparation.py": "prepare_a1_uv",
        "a1_texture_planning.py": "prepare_a1_texture_plan",
        "a1_document_preparation.py": "prepare_a1_document",
    }
    for filename, function_name in stages.items():
        function = _function(_tree(filename), function_name)
        assert function.end_lineno - function.lineno + 1 < 150, filename


def test_stage_modules_do_not_write_output_files():
    forbidden_calls = {"open", "write_text", "write_bytes", "unlink", "replace"}
    stage_files = (
        "a1_source_geometry_preparation.py",
        "a1_uv_preparation.py",
        "a1_texture_planning.py",
        "a1_document_preparation.py",
    )
    for filename in stage_files:
        tree = _tree(filename)
        calls = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name):
                calls.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                calls.add(node.func.attr)
        assert not calls.intersection(forbidden_calls), filename
