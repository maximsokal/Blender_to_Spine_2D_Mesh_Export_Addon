"""Contract for separating artist shader capability from real-coin geometry gates."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = (
    ROOT
    / "tests"
    / "blender_headless"
    / "run_coin_star_normal_geometry_publication_integration.py"
)


def test_publication_wrapper_runs_all_real_coin_normal_geometry_gates() -> None:
    source = RUNNER.read_text(encoding="utf-8")

    for marker in (
        "projection_gate._run(expected_blend)",
        "camera_root_gate._run(expected_blend)",
        "object_root_gate._run(expected_blend)",
        "ShaderBakeCapability.LOCAL_UV_SAFE",
        "_normal_uv_blocking_camera_findings(audits)",
        "source=restored",
    ):
        assert marker in source


def test_publication_wrapper_restores_original_material_in_finally() -> None:
    source = RUNNER.read_text(encoding="utf-8")

    assert "original_material = slot.material" in source
    assert "safe_material = _create_safe_material()" in source
    assert "slot.material = safe_material" in source
    assert "finally:" in source
    assert "slot.material = original_material" in source
    assert "bpy.data.materials.remove(safe_material)" in source
    assert "_object_fingerprint(source) == object_before" in source
    assert "_datablock_fingerprint() == datablocks_before" in source


def test_publication_override_has_no_displacement_path() -> None:
    source = RUNNER.read_text(encoding="utf-8")

    assert 'nodes.new(type="ShaderNodeEmission")' in source
    assert 'node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])' in source
    assert 'output.inputs["Displacement"]' not in source
