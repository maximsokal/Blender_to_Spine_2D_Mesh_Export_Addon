import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HEADLESS_ROOT = ROOT / "tests" / "blender_headless"
RUNNER = HEADLESS_ROOT / "run_multi_object_export_integration.py"


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=path.name)


def _runner_tree() -> ast.Module:
    return _tree(RUNNER)


def _function(tree: ast.AST, name: str) -> ast.FunctionDef:
    return next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _imported_module_names(tree: ast.Module) -> tuple[str, ...]:
    names: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            names.append(module)
            names.extend(
                f"{module}.{alias.name}" if module else alias.name
                for alias in node.names
            )
    return tuple(names)


def test_every_headless_runner_rejects_retired_bake_executor_imports():
    stale_imports: list[str] = []

    for path in sorted(HEADLESS_ROOT.glob("*.py")):
        for module_name in _imported_module_names(_tree(path)):
            if "bake_executor" in module_name:
                stale_imports.append(f"{path.name}: {module_name}")

    assert stale_imports == []


def test_shared_bake_and_multi_runners_use_current_semantic_bake_owner():
    expected_module = (
        "Blender_to_Spine2D_Mesh_Exporter.blender_adapter."
        "semantic_bake_execution"
    )

    for filename in (
        "run_bake_integration.py",
        "run_multi_object_export_integration.py",
    ):
        imported_modules = {
            (alias.name, alias.asname)
            for node in _tree(HEADLESS_ROOT / filename).body
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        assert (expected_module, "bake_module") in imported_modules, filename


def test_multi_object_rollback_hook_preserves_current_bake_signature():
    rollback = _function(
        _function(
            _runner_tree(),
            "test_second_bake_failure_rolls_back_json_and_both_textures",
        ),
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
