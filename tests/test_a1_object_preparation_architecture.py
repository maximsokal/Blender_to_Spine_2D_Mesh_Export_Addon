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


def _ordered_direct_call_names(function: ast.FunctionDef) -> tuple[str, ...]:
    calls = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    calls.sort(key=lambda node: (node.lineno, node.col_offset))
    return tuple(node.func.id for node in calls)


def test_public_orchestrator_is_short_and_calls_typed_stages_in_order():
    tree = _tree("a1_object_preparation.py")
    function = _function(tree, "prepare_a1_object")
    assert function.end_lineno - function.lineno + 1 < 100
    calls = [
        node.func.id
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id
        in {
            "_prepare_source_geometry",
            "prepare_a1_uv",
            "_prepare_texture",
            "_prepare_document",
        }
    ]
    calls.sort(
        key=lambda name: next(
            node.lineno
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == name
        )
    )
    assert calls == [
        "_prepare_source_geometry",
        "prepare_a1_uv",
        "_prepare_texture",
        "_prepare_document",
    ]


def test_depth_dispatch_helpers_preserve_explicit_stage_owners():
    tree = _tree("a1_object_preparation.py")

    source_dispatch = _ordered_direct_call_names(
        _function(tree, "_prepare_source_geometry")
    )
    texture_dispatch = _ordered_direct_call_names(_function(tree, "_prepare_texture"))
    document_dispatch = _ordered_direct_call_names(_function(tree, "_prepare_document"))

    assert "prepare_a1_source_geometry" in source_dispatch
    assert "prepare_a1_depth_source_geometry" in source_dispatch
    assert "prepare_a1_texture_plan" in texture_dispatch
    assert "prepare_a1_document" in document_dispatch
    assert "prepare_a1_depth_document" in document_dispatch


def test_public_orchestrator_owns_source_uv_integrity_guard():
    function = _function(_tree("a1_object_preparation.py"), "prepare_a1_object")
    with_calls = [
        item.context_expr.func.id
        for node in ast.walk(function)
        if isinstance(node, ast.With)
        for item in node.items
        if isinstance(item.context_expr, ast.Call)
        and isinstance(item.context_expr.func, ast.Name)
    ]

    assert with_calls == ["_source_uv_integrity_guard"]


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
        "a1_depth_source_geometry_preparation.py": "prepare_a1_depth_source_geometry",
        "a1_uv_preparation.py": "prepare_a1_uv",
        "a1_texture_planning.py": "prepare_a1_texture_plan",
        "a1_document_preparation.py": "prepare_a1_document",
        "a1_depth_document_preparation.py": "prepare_a1_depth_document",
    }
    for filename, function_name in stages.items():
        function = _function(_tree(filename), function_name)
        assert function.end_lineno - function.lineno + 1 < 180, filename


def test_source_geometry_decomposition_has_small_explicit_owners():
    tree = _tree("a1_source_geometry_preparation.py")
    helper_names = (
        "_resolve_source_request",
        "_normalize_source_geometry",
        "_prepare_projection_route",
        "_complete_projected_geometry",
        "_build_prepared_statistics",
        "_log_prepared_source",
    )
    for function_name in helper_names:
        function = _function(tree, function_name)
        assert function.end_lineno - function.lineno + 1 < 180, function_name

    public_calls = _ordered_direct_call_names(
        _function(tree, "prepare_a1_source_geometry")
    )
    required_order = (
        "_resolve_source_request",
        "_read_source_snapshot",
        "_normalize_source_geometry",
        "_prepare_projection_route",
        "build_a1_z_group_assignment",
        "_complete_projected_geometry",
        "_build_prepared_statistics",
        "_log_prepared_source",
    )
    positions = tuple(public_calls.index(name) for name in required_order)
    assert positions == tuple(sorted(positions))


def test_texture_planning_decomposition_has_small_explicit_owners():
    tree = _tree("a1_texture_planning.py")
    helper_names = (
        "_analyse_texture_material_inputs",
        "_build_generated_texture_result",
        "_preflight_source_material_images",
        "_build_source_texture_result",
    )
    for function_name in helper_names:
        function = _function(tree, function_name)
        assert function.end_lineno - function.lineno + 1 < 180, function_name

    public = _function(tree, "prepare_a1_texture_plan")
    direct_calls = {
        node.func.id
        for node in ast.walk(public)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
    }
    assert set(helper_names).issubset(direct_calls)


def test_stage_modules_do_not_write_output_files():
    forbidden_calls = {"open", "write_text", "write_bytes", "unlink"}
    stage_files = (
        "a1_source_geometry_preparation.py",
        "a1_depth_source_geometry_preparation.py",
        "a1_uv_preparation.py",
        "a1_texture_planning.py",
        "a1_document_preparation.py",
        "a1_depth_document_preparation.py",
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
