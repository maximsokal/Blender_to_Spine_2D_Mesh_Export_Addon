"""Architectural contract for explicit Depth source-preparation stages."""

from __future__ import annotations

import ast
from pathlib import Path


SOURCE = (
    Path(__file__).resolve().parents[1]
    / "Blender_to_Spine2D_Mesh_Exporter"
    / "blender_adapter"
    / "a1_depth_source_geometry_preparation.py"
)


def _tree() -> ast.Module:
    return ast.parse(SOURCE.read_text(encoding="utf-8"), filename=SOURCE.name)


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _ordered_direct_calls(function: ast.FunctionDef) -> tuple[str, ...]:
    calls = tuple(
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    )
    return tuple(
        node.func.id
        for node in sorted(calls, key=lambda item: (item.lineno, item.col_offset))
    )


def test_depth_stage_owners_remain_small_and_explicit() -> None:
    tree = _tree()
    helper_names = (
        "_prepare_depth_projection_stage",
        "_finalize_depth_geometry_stage",
        "_log_depth_projection_summary",
    )

    for helper_name in helper_names:
        function = _function(tree, helper_name)
        assert function.end_lineno is not None
        assert function.end_lineno - function.lineno + 1 < 180, helper_name


def test_depth_public_orchestrator_calls_stages_in_pipeline_order() -> None:
    tree = _tree()
    public = _function(tree, "prepare_a1_depth_source_geometry")
    assert public.end_lineno is not None
    assert public.end_lineno - public.lineno + 1 < 180

    calls = _ordered_direct_calls(public)
    required = (
        "_normal_camera_request_settings",
        "_resolve_source_request",
        "_read_source_snapshot",
        "_canonicalize_depth_evaluated_identity",
        "_normalize_source_geometry",
        "_prepare_depth_projection_stage",
        "build_a1_z_group_assignment",
        "_finalize_depth_geometry_stage",
        "_log_depth_projection_summary",
    )
    positions = tuple(calls.index(name) for name in required)
    assert positions == tuple(sorted(positions))


def test_depth_helpers_remain_blender_operator_free() -> None:
    source = SOURCE.read_text(encoding="utf-8")

    assert "bpy.ops" not in source
    assert "import bmesh" not in source
    assert "bmesh.new" not in source
