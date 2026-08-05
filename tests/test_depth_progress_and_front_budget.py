"""Contracts for responsive Depth preparation and shared front/reserve budgeting."""

from __future__ import annotations

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


def test_object_orchestrator_threads_progress_callback_into_depth_preparation() -> None:
    source = OBJECT_PREPARATION.read_text(encoding="utf-8")

    assert "progress_callback: A1ExportProgressCallback | None," in source
    assert "progress_callback=progress_callback," in source
    assert "prepare_a1_depth_source_geometry(" in source
    depth_call = source.split("return prepare_a1_depth_source_geometry(", 1)[1].split(
        ")",
        1,
    )[0]
    assert "progress_callback=progress_callback" in depth_call


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
    assert "settings=front_projection_settings" in source
    assert "max_points=settings.bake_execution.depth_projection.max_points" in source
