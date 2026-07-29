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


def _direct_function_call_names(function: ast.FunctionDef) -> tuple[str, ...]:
    result: list[str] = []
    for statement in function.body:
        if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
            continue
        callee = statement.value.func
        if isinstance(callee, ast.Name):
            result.append(callee.id)
    return tuple(result)


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


def test_shared_bake_fixture_uses_explicit_cycles_material_target():
    fixture = _function(_tree(HEADLESS_ROOT / "run_bake_integration.py"), "_build_fixture")
    calls = tuple(
        node
        for node in ast.walk(fixture)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "analyse_object_materials"
    )
    assert len(calls) == 1
    keywords = {
        keyword.arg: keyword.value
        for keyword in calls[0].keywords
        if keyword.arg is not None
    }
    render_target = keywords["render_target"]
    assert isinstance(render_target, ast.Constant)
    assert render_target.value == "CYCLES"


def test_shared_bake_tests_configure_cycles_after_factory_scene_reset():
    tree = _tree(HEADLESS_ROOT / "run_bake_integration.py")
    test_names = (
        "test_real_cycles_emit_bake_commits_png_and_restores_state",
        "test_forced_bake_failure_rolls_back_file_and_restores_state",
        "test_complete_a1_service_commits_valid_png_and_spine_json",
        "test_complete_a1_service_rolls_back_png_and_json_together",
    )

    for test_name in test_names:
        calls = _direct_function_call_names(_function(tree, test_name))
        assert calls[:2] == ("_clear_scene", "_configure_cycles_scene"), test_name


def test_multi_fixture_configures_cycles_before_capture():
    prepare_state = _function(
        _tree(HEADLESS_ROOT / "run_multi_object_export_integration.py"),
        "_prepare_state",
    )
    calls = _direct_function_call_names(prepare_state)
    assert calls[0] == "_configure_cycles_scene"
