"""Static contract for the Active Camera Blender acceptance numeric oracle."""

from __future__ import annotations

import ast
from pathlib import Path

from Blender_to_Spine2D_Mesh_Exporter.application.a1_z_groups import (
    LEGACY_Z_GROUP_DECIMALS,
)


WORKER = (
    Path(__file__).resolve().parents[1]
    / "tests"
    / "blender_headless"
    / "run_active_camera_normal_uv_acceptance.py"
)


def _worker_tree() -> ast.Module:
    return ast.parse(WORKER.read_text(encoding="utf-8"), filename=str(WORKER))


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    matches = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    assert len(matches) == 1, f"Expected exactly one worker function {name!r}"
    return matches[0]


def _called_names(node: ast.AST) -> set[str]:
    return {
        call.func.id
        for call in ast.walk(node)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
    }


def _module_constant(tree: ast.Module, name: str) -> object:
    matches = [
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        )
    ]
    assert len(matches) == 1, f"Expected exactly one module constant {name!r}"
    value = matches[0]
    assert isinstance(value, ast.Constant), f"{name} must remain a literal constant"
    return value.value


def test_depth_expectation_uses_captured_tuple_affine_arithmetic() -> None:
    source = WORKER.read_text(encoding="utf-8")
    tree = _worker_tree()
    expected_vertices = _function(tree, "_expected_vertices")
    camera_z = _function(tree, "_camera_z")

    model = _module_constant(tree, "_DEPTH_EXPECTATION_MODEL")
    assert isinstance(model, str)
    assert "CAMERA_SPACE_OBJECT_ORIGIN" in model

    assert "_affine_transform_point" in _called_names(expected_vertices)
    assert "_camera_z" in _called_names(expected_vertices)
    assert "_affine_transform_point" in _called_names(camera_z)
    assert not any(
        isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult)
        for function in (expected_vertices, camera_z)
        for node in ast.walk(function)
    )
    assert "source_object.matrix_world @ vertex.co" not in source


def test_screen_oracle_remains_blender_world_to_camera_view() -> None:
    tree = _worker_tree()
    expected_screen_point = _function(tree, "_expected_screen_point")

    assert "world_to_camera_view" in _called_names(expected_screen_point)


def test_depth_tolerance_matches_legacy_z_group_quantization() -> None:
    tree = _worker_tree()
    tolerance = _module_constant(tree, "_DEPTH_TOLERANCE")

    assert isinstance(tolerance, float)
    assert tolerance > 0.0

    # Camera-space Object Origin depth is calculated from captured affine tuples, while
    # the generated LegacyZGroup identity is canonicalized to the public four-decimal
    # contract. Keep the acceptance bound within one canonicalization step plus a small
    # cross-platform floating-point margin; arbitrary weakening must still fail closed.
    quantization_step = 10.0 ** (-LEGACY_Z_GROUP_DECIMALS)
    assert tolerance <= quantization_step * 1.1
