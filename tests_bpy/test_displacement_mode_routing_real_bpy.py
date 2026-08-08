"""Real Blender 5.2 regression for Material displacement-mode routing."""

from __future__ import annotations

import bpy

from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.production_shader_capability_displacement import (
    DISPLACEMENT_BUMP_CONTEXT_CODE,
    DISPLACEMENT_RENDER_REQUIRED_CODE,
    apply_displacement_method_boundary,
    material_displacement_method,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.production_shader_capability_routing import (
    _normal_uv_blocking_camera_findings,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import (
    MaterialCapabilityAudit,
    ShaderBakeCapability,
    ShaderCapabilityFinding,
)


def _audit() -> MaterialCapabilityAudit:
    return MaterialCapabilityAudit(
        material_name="RealDisplacementProbe",
        render_target="CYCLES",
        required_capability=ShaderBakeCapability.CAMERA_RENDER_REQUIRED,
        findings=(
            ShaderCapabilityFinding(
                capability=ShaderBakeCapability.CAMERA_RENDER_REQUIRED,
                code=DISPLACEMENT_RENDER_REQUIRED_CODE,
                reason="connected Material Output displacement",
            ),
        ),
    )


def _rna_owner(material):
    if hasattr(material, "displacement_method"):
        return material
    cycles = getattr(material, "cycles", None)
    if cycles is not None and hasattr(cycles, "displacement_method"):
        return cycles
    raise AssertionError("Blender 5.2 exposes no displacement_method RNA")


def test_real_blender_displacement_method_distinguishes_bump_from_true_geometry() -> None:
    material = bpy.data.materials.new("Spine2D_DisplacementModeProbe")
    owner = _rna_owner(material)
    original = owner.displacement_method

    try:
        prop = owner.bl_rna.properties["displacement_method"]
        identifiers = tuple(item.identifier for item in prop.enum_items)
        assert "BUMP" in identifiers

        owner.displacement_method = "BUMP"
        assert material_displacement_method(material) == "BUMP"
        bump_audit = apply_displacement_method_boundary(_audit(), material)
        assert tuple(finding.code for finding in bump_audit.findings) == (
            DISPLACEMENT_BUMP_CONTEXT_CODE,
        )
        assert _normal_uv_blocking_camera_findings((bump_audit,)) == ()

        true_modes = tuple(
            value
            for value in ("DISPLACEMENT", "BOTH", "TRUE")
            if value in identifiers
        )
        assert true_modes, (
            "Blender 5.2 displacement_method has no true-displacement enum; "
            f"identifiers={identifiers!r}"
        )
        for mode in true_modes:
            owner.displacement_method = mode
            assert material_displacement_method(material) == mode
            audit = apply_displacement_method_boundary(_audit(), material)
            assert any(
                finding.code == DISPLACEMENT_RENDER_REQUIRED_CODE
                for finding in audit.findings
            )
            assert _normal_uv_blocking_camera_findings((audit,))
    finally:
        try:
            owner.displacement_method = original
        finally:
            bpy.data.materials.remove(material)
