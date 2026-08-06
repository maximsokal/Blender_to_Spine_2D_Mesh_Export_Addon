"""Static contract for the real coin Normal UV full-export runner."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = (
    ROOT
    / "tests"
    / "blender_headless"
    / "run_coin_star_real_blend_normal_export_integration.py"
)


def test_runner_exports_real_coin_in_normal_uv_mode() -> None:
    source = RUNNER.read_text(encoding="utf-8")

    assert "_require_source_object()" in source
    assert 'prefix="Game Gold Coin"' in source
    assert "--expected-blend" in source
    assert "export_a1_single_object(" in source
    assert "A1TextureExportMode.NORMAL_UV_SEGMENTS" in source
    assert 'render_engine="CYCLES"' in source
    assert "BakeMode.COMBINED" in source
    assert 'statistics.get("texture_pipeline") == "OBJECT_BAKE"' in source
    assert 'statistics.get("bake_mode") == BakeMode.COMBINED.value' in source
    assert '"CAMERA_COMBINED" in strategy_ids' in source
    assert 'statistics.get("shader_capability") == "CAMERA_RENDER_REQUIRED"' in source
    assert '"projection_crop_width" not in statistics' in source
    assert "_read_visible_image_signal(" in source
    assert "_mesh_uv_stream_count(document)" in source
    assert "[COIN-REAL-NORMAL-EXPORT] PASS" in source
    assert "mode=NORMAL_UV_SEGMENTS" in source
    assert "pipeline=OBJECT_BAKE" in source
    assert "bake=COMBINED" in source
    assert "strategy=CAMERA_COMBINED" in source


def test_runner_uses_real_asset_without_synthesizing_or_mutating_it() -> None:
    source = RUNNER.read_text(encoding="utf-8")

    assert "_require_loaded_blend(expected_blend)" in source
    assert "_scene_fingerprint() == scene_before" in source
    assert "_capture_scene_bake_state() == bake_before" in source
    assert "_object_fingerprint(source) == object_before" in source
    assert "_datablock_fingerprint() == datablocks_before" in source
    assert "_temporary_datablock_names() == temporary_before" in source
    assert "bpy.ops" not in source
    assert "import bmesh" not in source
    assert "bmesh.new" not in source
    assert "from_pydata" not in source
    assert "bpy.data.objects.new" not in source
    assert "bpy.data.materials.new" not in source
