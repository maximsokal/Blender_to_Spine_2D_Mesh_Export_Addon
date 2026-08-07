"""Keep the real coin Blender gate independent of artist-facing material names."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = (
    ROOT
    / "tests"
    / "blender_headless"
    / "run_coin_star_real_blend_shader_capability_integration.py"
)


def test_real_coin_fixture_does_not_require_a_material_display_name() -> None:
    source = RUNNER.read_text(encoding="utf-8")

    assert '_EXPECTED_OBJECT_NAME = "Game Gold Coin"' in source
    assert "_EXPECTED_MATERIAL_NAME" not in source
    assert "unexpected real coin material" not in source
    assert 'len(source.material_slots) == 1' in source
    assert 'material is not None, "real coin material slot is empty"' in source


def test_real_coin_material_is_validated_by_shader_semantics() -> None:
    source = RUNNER.read_text(encoding="utf-8")

    for marker in (
        "_MUTED_ADVISORY in graph.issues",
        "ShaderBakeCapability.CAMERA_RENDER_REQUIRED",
        'finding.code == "GRAPH_CAMERA_DEPENDENCY"',
        'finding.node_type == "FRESNEL"',
        'finding.node_type == "BSDF_GLOSSY"',
        'finding.output_socket == "Generated"',
        "_normal_uv_blocking_camera_findings(audits)",
        "material={material.name_full!r}",
    ):
        assert marker in source
