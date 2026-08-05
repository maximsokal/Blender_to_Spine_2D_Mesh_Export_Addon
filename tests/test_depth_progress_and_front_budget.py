"""Contracts for responsive Depth preparation and shared front/reserve budgeting."""

from __future__ import annotations

import ast
from math import radians
from pathlib import Path

import pytest

from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.a1_depth_source_geometry_preparation import (
    _depth_front_projection_settings,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import (
    DepthCameraProjectionSettings,
)


ROOT = Path(__file__).resolve().parents[1]
OBJECT_PREPARATION = (
    ROOT
    / "Blender_to_Spine2D_Mesh_Exporter"
    / "blender_adapter"
    / "a1_object_preparation.py"
)
DEPTH_PREPARATION = (
    ROOT
    / "Blender_to_Spine2D_Mesh_Exporter"
    / "blender_adapter"
    / "a1_depth_source_geometry_preparation.py"
)


def _settings(max_points: int) -> DepthCameraProjectionSettings:
    return DepthCameraProjectionSettings(
        smoothing=0.0,
        edge_threshold_fraction=1.0,
        mesh_error_pixels=4.0,
        max_points=max_points,
    )


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _direct_call_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    return None


def _attribute_path(node: ast.AST) -> tuple[str, ...] | None:
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        return None
    parts.append(current.id)
    return tuple(reversed(parts))


def _keyword(call: ast.Call, name: str) -> ast.keyword:
    return next(
        keyword
        for keyword in call.keywords
        if keyword.arg == name
    )


def test_zero_horizon_preserves_complete_front_budget() -> None:
    source = _settings(128)

    resolved = _depth_front_projection_settings(
        source,
        horizon_angle_radians=0.0,
    )

    assert resolved is source
    assert resolved.max_points == 128


def test_positive_horizon_reserves_one_quarter_up_to_32_points() -> None:
    large = _depth_front_projection_settings(
        _settings(128),
        horizon_angle_radians=radians(50.0),
    )
    smoke = _depth_front_projection_settings(
        _settings(24),
        horizon_angle_radians=radians(50.0),
    )

    assert large.max_points == 96
    assert smoke.max_points == 18


def test_positive_horizon_rejects_impossible_shared_budget_before_geometry_work() -> None:
    with pytest.raises(
        ValueError,
        match="requires at least seven Max Depth Points",
    ):
        _depth_front_projection_settings(
            _settings(6),
            horizon_angle_radians=radians(50.0),
        )


def test_object_orchestrator_threads_progress_without_breaking_legacy_callers() -> None:
    source = OBJECT_PREPARATION.read_text(encoding="utf-8")

    assert "progress_callback: A1ExportProgressCallback | None = None," in source
    assert "if progress_callback is None:" in source
    assert source.count("return prepare_a1_depth_source_geometry(") == 2
    assert "progress_callback=progress_callback," in source

    no_callback_branch = source.split("if progress_callback is None:", 1)[1].split(
        "return prepare_a1_depth_source_geometry(",
        1,
    )[1].split(")", 1)[0]
    callback_branch = source.rsplit(
        "return prepare_a1_depth_source_geometry(",
        1,
    )[1].split(")", 1)[0]
    assert "progress_callback=" not in no_callback_branch
    assert "progress_callback=progress_callback" in callback_branch


def test_depth_preparation_emits_intermediate_progress_and_reserves_front_budget() -> None:
    source = DEPTH_PREPARATION.read_text(encoding="utf-8")

    for percent in (13, 18, 26, 30, 38, 43):
        assert f"percent={percent}" in source
    for message in (
        "Reading evaluated Depth geometry",
        "Projecting active-camera front surface",
        "Resolving virtual parallax camera views",
        "Expanding and budgeting parallax reserve",
        "Preparing Depth regions and UV lineage",
        "Depth source geometry prepared",
    ):
        assert message in source

    assert "def _depth_front_projection_settings(" in source
    assert "shared_budget // 4" in source
    assert "_MAXIMUM_RESERVE_POINT_ALLOCATION = 32" in source

    tree = ast.parse(source, filename=DEPTH_PREPARATION.name)
    projection_stage = _function(tree, "_prepare_depth_projection_stage")

    budget_assignments = tuple(
        node
        for node in ast.walk(projection_stage)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and _direct_call_name(node.value) == "_depth_front_projection_settings"
    )
    assert len(budget_assignments) == 1

    assignment = budget_assignments[0]
    assert len(assignment.targets) == 1
    assert isinstance(assignment.targets[0], ast.Name)
    front_budget_name = assignment.targets[0].id
    assert _attribute_path(assignment.value.args[0]) == (
        "settings",
        "bake_execution",
        "depth_projection",
    )

    surface_calls = tuple(
        node
        for node in ast.walk(projection_stage)
        if isinstance(node, ast.Call)
        and _direct_call_name(node) == "build_depth_camera_projection_surface"
    )
    assert len(surface_calls) == 1
    surface_settings = _keyword(surface_calls[0], "settings").value
    assert isinstance(surface_settings, ast.Name)
    assert surface_settings.id == front_budget_name

    parallax_calls = tuple(
        node
        for node in ast.walk(projection_stage)
        if isinstance(node, ast.Call)
        and _direct_call_name(node) == "build_depth_parallax_geometry_package"
    )
    assert len(parallax_calls) == 1
    assert _attribute_path(_keyword(parallax_calls[0], "max_points").value) == (
        "settings",
        "bake_execution",
        "depth_projection",
        "max_points",
    )
