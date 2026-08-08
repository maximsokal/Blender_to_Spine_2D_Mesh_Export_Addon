"""Static contract for the real coin shader-capability regression runner.

The real ``coin_star.blend`` fixture is artist-authored and its material display name or
exact node topology may legitimately evolve.  This contract therefore protects the
stable production boundaries instead of pinning one historical shader graph.
"""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = (
    ROOT
    / "tests"
    / "blender_headless"
    / "run_coin_star_real_blend_shader_capability_integration.py"
)


def _runner_source() -> str:
    source = RUNNER.read_text(encoding="utf-8")
    assert source.strip(), "real coin shader-capability runner is empty"
    return source


def test_runner_uses_real_coin_fixture_and_live_capability_boundaries() -> None:
    source = _runner_source()

    # Stable fixture identity is the real Blender object plus the caller-provided file,
    # not a historical material display name or one exact shader-node arrangement.
    assert '"Game Gold Coin"' in source
    assert '"CYCLES"' in source
    assert "--expected-blend" in source
    assert "_require_loaded_blend(expected_blend)" in source
    assert "_require_source_object()" in source

    # The runner must execute the same live analysis/audit/routing boundaries used by
    # production instead of reproducing capability decisions in the test itself.
    assert "analyse_object_materials(" in source
    assert "audit_object_material_capabilities(" in source
    assert "strongest_object_capability(audits)" in source
    assert "_normal_uv_blocking_camera_findings(audits)" in source
    assert "_PUBLICLY_ROUTABLE_CAPABILITIES" in source
    assert "_blocking_codes(blockers)" in source

    # Fail closed for unsupported or incompletely analysed shader graphs while allowing
    # every capability that has a real public execution route.
    assert "ShaderBakeCapability.UNSUPPORTED" in source
    assert 'finding.code == "GRAPH_ANALYSIS_INCOMPLETE"' in source
    assert "ShaderBakeCapability.LOCAL_UV_SAFE" in source
    assert "ShaderBakeCapability.SCENE_UV_SAFE" in source
    assert "ShaderBakeCapability.CAMERA_RENDER_REQUIRED" in source
    assert "not unsupported" in source
    assert "not incomplete" in source
    assert "capability in _PUBLICLY_ROUTABLE_CAPABILITIES" in source

    # Normal/UV may either be supported or honestly blocked by the live audit; Camera
    # modes remain the public fallback for CAMERA_RENDER_REQUIRED findings.
    assert 'normal_route = "blocked"' in source
    assert 'normal_route = "supported"' in source
    assert 'camera_route = "supported"' in source
    assert "blocker_codes" in source

    assert "[COIN-REAL-SHADER-CAPABILITY] PASS" in source
    assert "scene=unchanged" in source


def test_runner_preserves_real_coin_scene_and_cannot_synthesize_fixture() -> None:
    source = _runner_source()

    assert "scene_before = _scene_fingerprint()" in source
    assert "object_before = _object_fingerprint(source)" in source
    assert "datablocks_before = _datablock_fingerprint()" in source
    assert "_scene_fingerprint() == scene_before" in source
    assert "_object_fingerprint(source) == object_before" in source
    assert "_datablock_fingerprint() == datablocks_before" in source

    # This is an audit-only real-file runner.  It must not synthesize replacement
    # geometry/materials or mutate Blender through operators/bmesh.
    assert "bpy.ops" not in source
    assert "import bmesh" not in source
    assert "bmesh.new" not in source
    assert "from_pydata" not in source
    assert "bpy.data.objects.new" not in source
    assert "bpy.data.meshes.new" not in source
    assert "bpy.data.materials.new" not in source
