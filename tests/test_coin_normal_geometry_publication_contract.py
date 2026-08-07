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
        "ShaderBakeCapability.CAMERA_RENDER_REQUIRED",
        "_normal_uv_blocking_camera_findings(audits)",
        'finding.code == "SOURCE_OR_CAMERA_CONTEXT"',
        'finding.node_type == "FRESNEL"',
        "bake_route=CAMERA_COMBINED",
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


def test_publication_override_exercises_camera_combined_without_displacement() -> None:
    source = RUNNER.read_text(encoding="utf-8")

    assert 'nodes.new(type="ShaderNodeFresnel")' in source
    assert 'nodes.new(type="ShaderNodeBsdfDiffuse")' in source
    assert 'node_tree.links.new(fresnel.outputs["Fac"], diffuse.inputs["Color"])' in source
    assert 'node_tree.links.new(diffuse.outputs["BSDF"], output.inputs["Surface"])' in source
    assert 'output.inputs["Displacement"]' not in source
    assert 'nodes.new(type="ShaderNodeEmission")' not in source
