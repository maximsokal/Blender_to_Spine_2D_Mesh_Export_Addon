"""Architecture tests for physical frame progress wiring."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "Blender_to_Spine2D_Mesh_Exporter" / "blender_adapter"


def _tree(filename: str) -> ast.Module:
    path = ADAPTER / filename
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _function(filename: str, name: str) -> ast.FunctionDef:
    return next(
        node
        for node in _tree(filename).body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _call_lines(function: ast.FunctionDef, name: str) -> list[int]:
    result: list[int] = []
    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        target = node.func
        if isinstance(target, ast.Name) and target.id == name:
            result.append(node.lineno)
        elif isinstance(target, ast.Attribute) and target.attr == name:
            result.append(node.lineno)
    return sorted(result)


def _keyword_names(function: ast.FunctionDef, called_name: str) -> set[str]:
    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        target = node.func
        name = target.id if isinstance(target, ast.Name) else getattr(target, "attr", None)
        if name == called_name:
            return {keyword.arg for keyword in node.keywords if keyword.arg is not None}
    raise AssertionError(f"call {called_name!r} not found")


def test_each_physical_frame_loop_emits_before_and_after_real_work():
    cases = (
        (
            "semantic_bake_execution.py",
            "run_semantic_bake",
            "_bake_frame_task",
        ),
        (
            "camera_projection_execution.py",
            "render_camera_projection_frames",
            "_call_render_operator",
        ),
        (
            "grouped_camera_projection_execution.py",
            "render_grouped_camera_projection_frames",
            "_call_render_operator",
        ),
    )
    for filename, function_name, work_call in cases:
        function = _function(filename, function_name)
        progress_lines = _call_lines(function, "emit_a1_frame_progress")
        work_lines = _call_lines(function, work_call)

        assert len(progress_lines) == 2, filename
        assert len(work_lines) == 1, filename
        assert progress_lines[0] < work_lines[0] < progress_lines[1], filename
        assert any(isinstance(node, ast.For) for node in ast.walk(function)), filename


def test_texture_dispatcher_forwards_progress_to_both_execution_routes():
    camera_stage = _function("texture_executor.py", "_stage_camera_plan")
    stage = _function("texture_executor.py", "stage_texture_plan_outputs")
    execute = _function("texture_executor.py", "execute_bake_plan")

    assert "progress_callback" in _keyword_names(
        camera_stage,
        "stage_camera_projection_outputs_detailed",
    )
    assert "progress_callback" in _keyword_names(stage, "_stage_camera_plan")
    assert "progress_callback" in _keyword_names(stage, "stage_object_bake_outputs")
    assert "progress_callback" in _keyword_names(
        execute,
        "execute_camera_projection_plan",
    )
    assert "progress_callback" in _keyword_names(
        execute,
        "execute_object_bake_plan",
    )


def test_output_services_scale_frame_ranges_before_texture_and_grouped_staging():
    single = _function("a1_single_object_export.py", "export_a1_single_object")
    multi = _function("a1_multi_object_output.py", "export_a1_multi_object")
    mixed = _function("a1_mixed_object_output.py", "export_a1_mixed_object")
    shared = _function("a1_output_staging.py", "stage_and_finalize_a1_objects")

    assert "progress_callback" in _keyword_names(single, "stage_texture_plan_outputs")
    assert "progress_callback" in _keyword_names(shared, "stage_texture_plan_outputs")
    assert "progress_callback" in _keyword_names(
        multi,
        "stage_grouped_camera_projection_outputs",
    )
    assert "progress_callback" in _keyword_names(
        mixed,
        "stage_grouped_camera_projection_outputs",
    )


def test_final_frame_completion_event_occurs_after_output_validation():
    camera = _function(
        "camera_projection_execution.py",
        "render_camera_projection_frames",
    )
    grouped = _function(
        "grouped_camera_projection_execution.py",
        "render_grouped_camera_projection_frames",
    )

    camera_progress = _call_lines(camera, "emit_a1_frame_progress")
    camera_validation = _call_lines(camera, "_require_nonempty_staged_output")
    grouped_progress = _call_lines(grouped, "emit_a1_frame_progress")
    grouped_validation = _call_lines(
        grouped,
        "require_nonempty_grouped_staged_output",
    )

    assert camera_validation[0] < camera_progress[-1]
    assert grouped_validation[0] < grouped_progress[-1]
