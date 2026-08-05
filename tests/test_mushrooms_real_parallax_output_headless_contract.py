"""Static contract for the real render/crop/remap mushrooms regression."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = (
    ROOT
    / "tests"
    / "blender_headless"
    / "run_mushrooms_real_blend_parallax_output_integration.py"
)


def test_runner_executes_complete_real_output_not_preparation_only() -> None:
    source = RUNNER.read_text(encoding="utf-8")

    assert "export_a1_multi_object(" in source
    assert "prepare_a1_multi_object(" not in source
    assert "_require_loaded_blend(expected_blend)" in source
    assert "_require_source_objects()" in source
    assert "_MAX_POINTS = 128" in source
    assert "_HORIZON_DEGREES = 50.0" in source
    assert "_MAX_EXPORT_SECONDS = 300.0" in source
    assert '"RIGHT"' in source
    assert '"DOWN_RIGHT"' in source
    assert "path.read_bytes().startswith(PNG_SIGNATURE)" in source
    assert "_assert_serialized_uvs(document)" in source
    assert '"spine2d-stage-v3" in path.name' in source
    assert "depth_parallax_cropped_view_count" in source
    assert "parallax_texture_output_count" in source
    assert "[MUSHROOMS-REAL-PARALLAX-OUTPUT] PASS" in source
    assert "crop=alpha-union-geometry" in source


def test_runner_never_synthesizes_or_mutates_the_real_asset() -> None:
    source = RUNNER.read_text(encoding="utf-8")

    assert "_object_fingerprint(source)" in source
    assert "_temporary_datablock_names() == temporary_before" in source
    assert "_capture_scene_bake_state() == bake_before" in source
    assert "_scene_render_fingerprint() == render_before" in source
    assert "_loaded_scene_render_engine() == engine_before" in source
    assert "_camera_fingerprint() == camera_before" in source
    assert "_create_mesh_object" not in source
    assert "_clear_scene" not in source
    assert "bpy.ops" not in source
    assert "import bmesh" not in source
