"""Regression for Normal UV routing of camera/source-context shader findings."""

from __future__ import annotations

from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.production_shader_capability_routing import (
    _normal_uv_blocking_camera_findings,
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


def _coin_audit() -> MaterialCapabilityAudit:
    return MaterialCapabilityAudit(
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
                "Glossy is resolved by Cycles COMBINED object bake",
                node_id="Glossy BSDF",
                node_type="BSDF_GLOSSY",
            ),
            _camera_finding(
                "SOURCE_OR_CAMERA_CONTEXT",
                "Second Glossy is resolved by Cycles COMBINED object bake",
                node_id="Glossy BSDF.001",
                node_type="BSDF_GLOSSY",
            ),
        ),
    )


def test_coin_glossy_fresnel_and_generated_are_normal_uv_bakeable() -> None:
    audit = _coin_audit()

    assert _normal_uv_blocking_camera_findings((audit,)) == ()


def test_unclassified_camera_surface_finding_remains_fail_closed() -> None:
    audit = MaterialCapabilityAudit(
        material_name="Unsupported refraction",
        render_target="CYCLES",
        required_capability=ShaderBakeCapability.CAMERA_RENDER_REQUIRED,
        findings=(
            _camera_finding(
                "SOURCE_OR_CAMERA_CONTEXT",
                "Refraction has no audited Normal UV object-bake route",
                node_id="Refraction BSDF",
                node_type="BSDF_REFRACTION",
            ),
        ),
    )

    blockers = _normal_uv_blocking_camera_findings((audit,))
    guidance = normal_mode_camera_requirement_message((audit,))

    assert blockers == (
        (
            "Unsupported refraction",
            (("SOURCE_OR_CAMERA_CONTEXT", "BSDF_REFRACTION", None),),
        ),
    )
    assert "Camera Projection or Depth Camera Projection" in guidance
    assert "BSDF_REFRACTION" in guidance
