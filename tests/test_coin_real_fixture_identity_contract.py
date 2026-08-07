"""Keep the real coin Blender gate stable across artist material revisions."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = (
    ROOT
    / "tests"
    / "blender_headless"
    / "run_coin_star_real_blend_shader_capability_integration.py"
)


def test_real_coin_fixture_uses_stable_object_identity_only() -> None:
    source = RUNNER.read_text(encoding="utf-8")

    assert '_EXPECTED_OBJECT_NAME = "Game Gold Coin"' in source
    assert "_EXPECTED_MATERIAL_NAME" not in source
    assert "_MUTED_ADVISORY" not in source
    assert "unexpected real coin material" not in source
    assert 'len(source.material_slots) == 1' in source
    assert 'material is not None, "real coin material slot is empty"' in source


def test_real_coin_material_is_validated_by_public_route_capability() -> None:
    source = RUNNER.read_text(encoding="utf-8")

    for marker in (
        "ShaderBakeCapability.LOCAL_UV_SAFE",
        "ShaderBakeCapability.SCENE_UV_SAFE",
        "ShaderBakeCapability.CAMERA_RENDER_REQUIRED",
        "ShaderBakeCapability.UNSUPPORTED",
        'finding.code == "GRAPH_ANALYSIS_INCOMPLETE"',
        "strongest_object_capability(audits)",
        "_normal_uv_blocking_camera_findings(audits)",
        "capability in _PUBLICLY_ROUTABLE_CAPABILITIES",
        "capability is ShaderBakeCapability.CAMERA_RENDER_REQUIRED",
        'normal_route = "blocked"',
        'camera_route = "supported"',
        "material={material.name_full!r}",
        "blockers={blocker_codes}",
        "scene=unchanged",
    ):
        assert marker in source

    for stale_fixture_marker in (
        'finding.node_type == "FRESNEL"',
        'finding.node_type == "BSDF_GLOSSY"',
        'finding.output_socket == "Generated"',
        "muted_fallback=advisory",
        "capability in _ALLOWED_NORMAL_UV_CAPABILITIES",
        "blockers=none normal_mode=object-bake",
    ):
        assert stale_fixture_marker not in source
