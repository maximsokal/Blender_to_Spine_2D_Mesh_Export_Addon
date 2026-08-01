from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / "Blender_to_Spine2D_Mesh_Exporter"
    / "blender_adapter"
    / "a1_source_geometry_preparation.py"
)


def _source_text() -> str:
    return SOURCE.read_text(encoding="utf-8")


def _tree() -> ast.Module:
    return ast.parse(_source_text(), filename=SOURCE.name)


def _function(name: str) -> ast.FunctionDef:
    return next(
        node
        for node in _tree().body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _ordered_call_names(node: ast.AST) -> tuple[str, ...]:
    calls = [
        item
        for item in ast.walk(node)
        if isinstance(item, ast.Call) and isinstance(item.func, ast.Name)
    ]
    calls.sort(key=lambda item: (item.lineno, item.col_offset))
    return tuple(item.func.id for item in calls)


def test_a1_source_geometry_normalizes_before_uv_projection_z_and_completion():
    normalize = _function("_normalize_source_geometry")
    normalize_calls = _ordered_call_names(normalize)
    assert normalize_calls.index("normalize_mesh_snapshot_world_transform") < (
        normalize_calls.index("_resolve_source_uv_boundary_layer")
    )

    public = _function("prepare_a1_source_geometry")
    public_calls = _ordered_call_names(public)
    required = (
        "_read_source_snapshot",
        "_normalize_source_geometry",
        "_prepare_projection_route",
        "build_a1_z_group_assignment",
        "_complete_projected_geometry",
    )
    positions = tuple(public_calls.index(name) for name in required)
    assert positions == tuple(sorted(positions))

    source = _source_text()
    assert "source_snapshot = world_transform.snapshot" not in source
    assert "normalized_snapshot = world_transform.snapshot" in source


def test_projection_routes_preserve_axis_and_active_camera_stage_order():
    route = _function("_prepare_projection_route")
    branch = next(
        node
        for node in route.body
        if isinstance(node, ast.If) and "axis_aligned" in ast.unparse(node.test)
    )

    axis_calls = _ordered_call_names(ast.Module(body=branch.body, type_ignores=[]))
    camera_calls = _ordered_call_names(ast.Module(body=branch.orelse, type_ignores=[]))

    assert "project_a1_mesh_snapshot_axis" in axis_calls
    assert "prepare_a1_geometry_regions" not in axis_calls

    camera_required = (
        "prepare_a1_geometry_regions",
        "resolve_a1_active_camera_projection_frame",
        "calculate_uniform_scale",
        "project_a1_mesh_snapshot_camera",
        "project_a1_prepared_geometry_camera",
    )
    camera_positions = tuple(camera_calls.index(name) for name in camera_required)
    assert camera_positions == tuple(sorted(camera_positions))


def test_a1_source_geometry_records_transform_diagnostics():
    source = _source_text()

    assert '"object_linear_transform_baked"' in source
    assert '"object_world_determinant"' in source
    assert '"object_world_mirrored"' in source
    assert 'code="MIRRORED_OBJECT_TRANSFORM"' in source


def test_evaluated_geometry_binds_scene_dependency_graph_mesh_and_matrix():
    source = _source_text()

    owners_index = source.index("_resolved_evaluation_owners(scene)")
    read_index = source.index("read_evaluated_mesh_snapshot(", owners_index)
    matrix_index = source.index("_evaluated_source_world_matrix(", read_index)
    normalize_index = source.index("normalize_mesh_snapshot_world_transform(", matrix_index)

    assert owners_index < read_index < matrix_index < normalize_index
    assert "require_depsgraph_scene_consistency" in source
    assert "depsgraph=resolved_depsgraph" in source
    assert "evaluated_get(depsgraph)" in source
    assert "evaluated_snapshot = replace(" in source
    assert "world_matrix=_evaluated_source_world_matrix(" in source
