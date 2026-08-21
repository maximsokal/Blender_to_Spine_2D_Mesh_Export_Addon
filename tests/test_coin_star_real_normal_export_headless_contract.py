"""Static contract for the real coin artist-material Normal/UV rejection runner."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = (
    ROOT
    / "tests"
    / "blender_headless"
    / "run_coin_star_real_blend_normal_export_integration.py"
)


def test_runner_rejects_real_coin_displacement_in_normal_uv_mode() -> None:
    source = RUNNER.read_text(encoding="utf-8")

    assert "_require_source_object()" in source
    assert 'prefix="Game Gold Coin"' in source
    assert "--expected-blend" in source
    assert "export_a1_single_object(" in source
    assert "A1TextureExportMode.NORMAL_UV_SEGMENTS" in source
    assert "source_geometry_mode=A1SourceGeometryMode.ORIGINAL" in source
    assert 'render_engine="CYCLES"' in source
    assert "BakeMode.COMBINED" in source
    assert "not result.success" in source
    assert '_EXPECTED_STAGE = "PLAN_BAKE"' in source
    assert '_EXPECTED_CODE = "A1_PLAN_BAKE_FAILED"' in source
    assert '_EXPECTED_BLOCKER = "DISPLACEMENT_RENDER_REQUIRED"' in source
    assert '"Camera Projection" in combined' in source
    assert '"Depth Camera Projection" in combined' in source
    assert "not tuple(result.output_files)" in source
    assert "[COIN-REAL-NORMAL-FAIL-CLOSED] PASS" in source
    assert "outputs=0" in source


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
