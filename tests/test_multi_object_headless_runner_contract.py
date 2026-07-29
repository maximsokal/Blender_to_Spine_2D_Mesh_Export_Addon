import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tests" / "blender_headless" / "run_multi_object_export_integration.py"


def _runner_tree() -> ast.Module:
    return ast.parse(RUNNER.read_text(encoding="utf-8"), filename=RUNNER.name)


def _function(tree: ast.AST, name: str) -> ast.FunctionDef:
    return next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def test_multi_object_headless_runner_uses_current_semantic_bake_owner():
    tree = _runner_tree()
    imported_modules = {
        (alias.name, alias.asname)
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    }

    assert (
        "Blender_to_Spine2D_Mesh_Exporter.blender_adapter.semantic_bake_execution",
        "bake_module",
    ) in imported_modules
    assert not any("bake_executor" in module for module, _alias in imported_modules)


def test_multi_object_rollback_hook_preserves_current_bake_signature():
    rollback = _function(
        _function(_runner_tree(), "test_second_bake_failure_rolls_back_json_and_both_textures"),
        "fail_second_bake",
    )

    assert tuple(argument.arg for argument in rollback.args.args) == (
        "bpy_module",
        "bake_type",
    )
    assert tuple(argument.arg for argument in rollback.args.kwonlyargs) == (
        "uv_layer_name",
    )

    original_calls = tuple(
        node
        for node in ast.walk(rollback)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "original_call"
    )
    assert len(original_calls) == 1
    forwarded_keywords = {
        keyword.arg: keyword.value
        for keyword in original_calls[0].keywords
        if keyword.arg is not None
    }
    forwarded_uv = forwarded_keywords["uv_layer_name"]
    assert isinstance(forwarded_uv, ast.Name)
    assert forwarded_uv.id == "uv_layer_name"
