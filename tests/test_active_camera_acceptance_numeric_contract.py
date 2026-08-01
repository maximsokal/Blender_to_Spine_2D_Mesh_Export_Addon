"""Static contract for the Active Camera Blender acceptance numeric oracle."""

from __future__ import annotations

import ast
from pathlib import Path


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


def test_depth_expectation_uses_captured_tuple_affine_arithmetic() -> None:
    source = WORKER.read_text(encoding="utf-8")
    tree = _worker_tree()
    expected_vertices = _function(tree, "_expected_vertices")
    camera_z = _function(tree, "_camera_z")

    assert "CAPTURED_TUPLE_AFFINE" in source
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


def test_depth_tolerance_was_not_weakened() -> None:
    tree = _worker_tree()
    assignments = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "_DEPTH_TOLERANCE"
            for target in node.targets
        )
    ]

    assert len(assignments) == 1
    value = assignments[0].value
    assert isinstance(value, ast.Constant)
    assert float(value.value) == 1.0e-8
