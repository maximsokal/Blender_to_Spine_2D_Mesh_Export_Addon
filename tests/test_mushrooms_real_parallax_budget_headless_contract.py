"""Static contract for the direct real-asset positive parallax regression."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = (
    ROOT
    / "tests"
    / "blender_headless"
    / "run_mushrooms_real_blend_parallax_budget_integration.py"
)


def test_runner_uses_real_asset_positive_horizon_budget_and_live_progress() -> None:
    source = RUNNER.read_text(encoding="utf-8")

    assert "_MAX_POINTS = 128" in source
    assert "_HORIZON_DEGREES = 50.0" in source
    assert "_MAX_PREPARATION_SECONDS = 120.0" in source
    assert '_PLANE_COMPONENT_ID = "object_1:Plane.008"' in source
    assert "_require_loaded_blend(expected_blend)" in source
    assert "_require_source_objects()" in source
    assert "DepthParallaxSettings(" in source
    assert "class _TimedProgressEvent:" in source
    assert "class _ProgressRecorder:" in source
    assert 'if update.object_id == "Plane.008":' in source
    assert "replace(update, object_id=_PLANE_COMPONENT_ID)" in source
    assert "[MUSHROOMS-PROGRESS]" in source
    assert "[MUSHROOMS-STAGE-TIMING]" in source
    assert "def _print_plane_stage_timings(" in source
    assert "read_geometry=" in source
    assert "front_projection=" in source
    assert "parallax=" in source
    assert "regions_lineage=" in source
    assert "prepare_a1_multi_object(" in source
    assert "progress_callback=progress" in source
    assert "_require_responsive_depth_progress(progress.updates)" in source
    assert "_print_plane_stage_timings(progress.events)" in source
    assert source.index("_print_plane_stage_timings(progress.events)") < source.index(
        "elapsed <= _MAX_PREPARATION_SECONDS"
    )
    assert "Projecting active-camera front surface" in source
    assert "Resolving virtual parallax camera views" in source
    assert "Expanding and budgeting parallax reserve" in source
    assert "any(update.percent > 12 for update in matches)" in source
    assert "len(package.union_snapshot.vertices) <= _MAX_POINTS" in source
    assert "len(surface.source_face_indices) > len(surface.snapshot.faces)" in source
    assert '"parallax-budget-proxy" in plane_package.union_snapshot.snapshot_id' in source
    assert "_temporary_datablock_names() == temporary_before" in source
    assert "progress_events={len(progress.events)}" in source
    assert "[MUSHROOMS-REAL-PARALLAX-BUDGET] PASS" in source
    assert "_create_mesh_object" not in source
    assert "_clear_scene" not in source
    assert "bpy.ops" not in source
    assert "import bmesh" not in source
