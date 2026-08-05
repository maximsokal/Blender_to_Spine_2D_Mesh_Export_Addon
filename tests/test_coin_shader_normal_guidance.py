"""Regression for separating full shader audit findings from Normal-mode blockers."""

from __future__ import annotations

from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.production_shader_capability_routing import (
    normal_mode_camera_requirement_message,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import (
    MaterialCapabilityAudit,
    ShaderBakeCapability,
    ShaderCapabilityFinding,
)


def _camera_finding(
    code: str,
    reason: str,
    *,
    node_id: str | None = None,
    node_type: str | None = None,
    output_socket: str | None = None,
) -> ShaderCapabilityFinding:
    return ShaderCapabilityFinding(
        capability=ShaderBakeCapability.CAMERA_RENDER_REQUIRED,
        code=code,
        reason=reason,
        node_id=node_id,
        node_type=node_type,
        output_socket=output_socket,
    )


def test_coin_guidance_lists_only_findings_normal_object_bake_cannot_reproduce() -> None:
    audit = MaterialCapabilityAudit(
        material_name="Gold coin",
        render_target="CYCLES",
        required_capability=ShaderBakeCapability.CAMERA_RENDER_REQUIRED,
        findings=(
            _camera_finding(
                "GRAPH_CAMERA_DEPENDENCY",
                "Graph depends on camera evaluation",
            ),
            _camera_finding(
                "SOURCE_OR_CAMERA_CONTEXT",
                "Fresnel needs source-object or camera context",
                node_id="Fresnel",
                node_type="FRESNEL",
            ),
            _camera_finding(
                "TEXTURE_COORD_SOURCE_CONTEXT",
                "Generated coordinates need original source-object context",
                node_id="Texture Coordinate",
                node_type="TEX_COORD",
                output_socket="Generated",
            ),
            _camera_finding(
                "SOURCE_OR_CAMERA_CONTEXT",
                "Glossy needs a render ray",
                node_id="Glossy BSDF",
                node_type="BSDF_GLOSSY",
            ),
            _camera_finding(
                "SOURCE_OR_CAMERA_CONTEXT",
                "Second Glossy needs a render ray",
                node_id="Glossy BSDF.001",
                node_type="BSDF_GLOSSY",
            ),
        ),
    )

    guidance = normal_mode_camera_requirement_message((audit,))

    assert "Camera Projection or Depth Camera Projection" in guidance
    assert guidance.count("BSDF_GLOSSY") == 2
    assert "Generated" not in guidance
    assert "FRESNEL" not in guidance
    assert "GRAPH_CAMERA_DEPENDENCY" not in guidance
