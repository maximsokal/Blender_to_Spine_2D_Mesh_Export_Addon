import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HEADLESS_ROOT = ROOT / "tests" / "blender_headless"

SEMANTIC_BAKE_OWNER = (
    "Blender_to_Spine2D_Mesh_Exporter.blender_adapter."
    "semantic_bake_execution"
)
CAMERA_RENDER_OWNER = (
    "Blender_to_Spine2D_Mesh_Exporter.blender_adapter."
    "camera_projection_execution"
)

SEMANTIC_BAKE_RUNNERS = (
    "run_bake_integration.py",
    "run_multi_object_export_integration.py",
    "run_alpha_sequence_rollback_integration.py",
    "run_legacy_bake_matrix_integration.py",
    "run_scene_bake_extended_integration.py",
)
CAMERA_RENDER_RUNNERS = (
    "run_camera_projection_integration.py",
    "run_camera_projection_postprocess_isolation_integration.py",
    "run_custom_compositor_isolation_matrix.py",
    "run_view_layer_contract_integration.py",
)


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=path.name)


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


def _direct_import_aliases(tree: ast.Module) -> set[tuple[str, str | None]]:
    return {
        (alias.name, alias.asname)
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    }


def _assert_current_owner(filename: str, module_name: str, alias: str) -> None:
    aliases = _direct_import_aliases(_tree(HEADLESS_ROOT / filename))
    assert (module_name, alias) in aliases, filename


def _assert_forwarding_hook(
    *,
    filename: str,
    outer_function: str,
    hook_function: str,
) -> None:
    outer = _function(_tree(HEADLESS_ROOT / filename), outer_function)
    hook = _function(outer, hook_function)

    assert tuple(argument.arg for argument in hook.args.args) == (
        "bpy_module",
        "bake_type",
    ), filename
    assert tuple(argument.arg for argument in hook.args.kwonlyargs) == (
        "uv_layer_name",
    ), filename

    original_calls = tuple(
        node
        for node in ast.walk(hook)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "original_call"
    )
    assert len(original_calls) == 1, filename

    forwarded_keywords = {
        keyword.arg: keyword.value
        for keyword in original_calls[0].keywords
        if keyword.arg is not None
    }
    forwarded_uv = forwarded_keywords["uv_layer_name"]
    assert isinstance(forwarded_uv, ast.Name), filename
    assert forwarded_uv.id == "uv_layer_name", filename


def test_every_headless_runner_rejects_retired_bake_executor_imports():
    stale_imports: list[str] = []

    for path in sorted(HEADLESS_ROOT.glob("*.py")):
        for module_name in _imported_module_names(_tree(path)):
            if "bake_executor" in module_name:
                stale_imports.append(f"{path.name}: {module_name}")

    assert stale_imports == []


def test_semantic_bake_runners_import_the_current_execution_owner():
    for filename in SEMANTIC_BAKE_RUNNERS:
        _assert_current_owner(filename, SEMANTIC_BAKE_OWNER, "bake_module")


def test_camera_projection_runners_import_the_current_render_owner():
    for filename in CAMERA_RENDER_RUNNERS:
        _assert_current_owner(filename, CAMERA_RENDER_OWNER, "render_module")


def test_bake_rollback_hooks_preserve_current_operator_signature():
    cases = (
        (
            "run_multi_object_export_integration.py",
            "test_second_bake_failure_rolls_back_json_and_both_textures",
            "fail_second_bake",
        ),
        (
            "run_alpha_sequence_rollback_integration.py",
            "test_failure_on_alpha_pass_rolls_back_existing_png_and_restores_state",
            "fail_on_alpha_pass",
        ),
        (
            "run_legacy_bake_matrix_integration.py",
            "test_sequence_failure_rolls_back_json_static_and_all_frames",
            "fail_frame_two",
        ),
    )

    for filename, outer_function, hook_function in cases:
        _assert_forwarding_hook(
            filename=filename,
            outer_function=outer_function,
            hook_function=hook_function,
        )
