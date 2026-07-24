from dataclasses import replace

import pytest

from Blender_to_Spine2D_Mesh_Exporter.blender_adapter import scene_bake_runtime
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.scene_bake_error import (
    SceneBakeAnalysisError,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import ObjectBakeContext


IDENTITY = (
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
)


def _context(matrix=IDENTITY):
    return ObjectBakeContext(
        source_object_id="Object",
        object_type="MESH",
        world_matrix=tuple(float(value) for value in matrix),
        animated=True,
    )


def _install_current(monkeypatch, current):
    monkeypatch.setattr(
        scene_bake_runtime,
        "analyse_object_bake_context",
        lambda _source: current,
    )


def test_unchanged_source_transform_is_accepted_on_sequence_frame(monkeypatch):
    expected = _context()
    _install_current(monkeypatch, expected)

    scene_bake_runtime.validate_runtime_object_transform(
        object(),
        expected,
        timeline_frame=12,
    )


def test_float_noise_below_matrix_tolerance_is_accepted(monkeypatch):
    expected = _context()
    noisy_matrix = list(IDENTITY)
    noisy_matrix[3] = 5.0e-11
    current = replace(expected, world_matrix=tuple(noisy_matrix))
    _install_current(monkeypatch, current)

    scene_bake_runtime.validate_runtime_object_transform(
        object(),
        expected,
        timeline_frame=13,
    )


def test_translation_change_is_rejected_with_camera_projection_guidance(monkeypatch):
    expected = _context()
    moved_matrix = list(IDENTITY)
    moved_matrix[3] = 1.0
    current = replace(expected, world_matrix=tuple(moved_matrix))
    _install_current(monkeypatch, current)

    with pytest.raises(SceneBakeAnalysisError, match="Use camera projection") as captured:
        scene_bake_runtime.validate_runtime_object_transform(
            object(),
            expected,
            timeline_frame=14,
        )

    assert "frame=14" in str(captured.value)
    assert "matrix_index=3" in str(captured.value)


def test_rotation_or_scale_change_is_rejected(monkeypatch):
    expected = _context()
    changed_matrix = list(IDENTITY)
    changed_matrix[0] = 2.0
    current = replace(expected, world_matrix=tuple(changed_matrix))
    _install_current(monkeypatch, current)

    with pytest.raises(SceneBakeAnalysisError, match="fixed UV target"):
        scene_bake_runtime.validate_runtime_object_transform(
            object(),
            expected,
            timeline_frame=15,
        )


def test_missing_object_context_keeps_blender_independent_plans_supported():
    scene_bake_runtime.validate_runtime_object_transform(
        object(),
        None,
        timeline_frame=None,
    )
