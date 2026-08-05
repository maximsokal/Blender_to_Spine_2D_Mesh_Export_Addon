"""Static contract for the real coin shader-capability regression runner."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = (
    ROOT
    / "tests"
    / "blender_headless"
    / "run_coin_star_real_blend_shader_capability_integration.py"
)


def test_runner_uses_the_real_coin_asset_and_live_material_audit() -> None:
    source = RUNNER.read_text(encoding="utf-8")

    assert '"Game Gold Coin"' in source
    assert '"Gold coin"' in source
    assert '"CYCLES"' in source
    assert "--expected-blend" in source
    assert "analyse_object_materials(" in source
    assert "audit_object_material_capabilities(" in source
    assert "strongest_object_capability(audits)" in source
    assert "ShaderBakeCapability.CAMERA_RENDER_REQUIRED" in source
    assert 'finding.node_type == "FRESNEL"' in source
    assert 'finding.node_type == "BSDF_GLOSSY"' in source
    assert 'finding.output_socket == "Generated"' in source
    assert 'finding.code == "GRAPH_ANALYSIS_INCOMPLETE"' in source
    assert "normal_mode_camera_requirement_message(audits)" in source
    assert '"Camera Projection or Depth Camera Projection"' in source
    assert "[COIN-REAL-SHADER-CAPABILITY] PASS" in source
    assert "muted_fallback=advisory" in source
    assert "normal_mode=camera-required" in source


def test_runner_cannot_synthesize_or_mutate_the_coin_scene() -> None:
    source = RUNNER.read_text(encoding="utf-8")

    assert "_require_loaded_blend(expected_blend)" in source
    assert "_require_source_object()" in source
    assert "_scene_fingerprint() == scene_before" in source
    assert "_object_fingerprint(source) == object_before" in source
    assert "_datablock_fingerprint() == datablocks_before" in source
    assert "bpy.ops" not in source
    assert "import bmesh" not in source
    assert "bmesh.new" not in source
    assert "from_pydata" not in source
    assert "bpy.data.objects.new" not in source
    assert "bpy.data.materials.new" not in source
