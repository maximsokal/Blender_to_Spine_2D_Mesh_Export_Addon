"""Pure object-level routing from user mode and audited shader capability."""

from __future__ import annotations

from typing import Tuple

from ..domain.baking import (
    A1TextureExportMode,
    BakePlanError,
    BakeSettings,
    MaterialCapabilityAudit,
    ObjectBakeContext,
    ObjectMaterialAnalysis,
    SceneBakeContext,
    ShaderBakeCapability,
    ShaderCapabilityFinding,
    TexturePlan,
    build_camera_projection_plan,
    build_bake_plan,
    strongest_shader_capability,
)
from ..domain.baking.normal_uv_camera_context import (
    build_normal_uv_camera_context_plan,
)
from .production_shader_capability_principled import (
    PRINCIPLED_REFLECTION_CONTEXT_CODE,
    PRINCIPLED_REFLECTION_CONTEXT_OUTPUTS,
    PRINCIPLED_TRANSMISSION_CONTEXT_CODE,
    PRINCIPLED_TRANSMISSION_CONTEXT_OUTPUTS,
)
from .render_engine_contract import RenderEngineContract


# These nodes need the original source-object/camera evaluation context, but Blender's
# Cycles COMBINED object bake can still resolve them directly into the generated Spine UV
# layout. They therefore do not require the exported geometry to become Camera Projection.
_NORMAL_UV_SOURCE_CONTEXT_NODE_TYPES = frozenset(
    {
        "BSDF_GLOSSY",
        "FRESNEL",
        "LAYER_WEIGHT",
        "VECT_TRANSFORM",
        "VECTOR_TRANSFORM",
    }
)

_NORMAL_UV_TEXTURE_COORD_OUTPUTS = frozenset(
    {
        "camera",
        "generated",
        "normal",
        "object",
        "reflection",
    }
)

_NORMAL_UV_GEOMETRY_OUTPUTS = frozenset(
    {
        "pointiness",
        "random per island",
    }
)

_GRAPH_CAMERA_AGGREGATE_CODE = "GRAPH_CAMERA_DEPENDENCY"
_CAMERA_RENDER_MODES = frozenset(
    {
        A1TextureExportMode.CAMERA_PROJECTION,
        A1TextureExportMode.DEPTH_CAMERA_PROJECTION,
    }
)


def strongest_object_capability(
    audits: Tuple[MaterialCapabilityAudit, ...],
) -> ShaderBakeCapability:
    """Return the strongest audited requirement across one object's materials."""

    if not isinstance(audits, tuple):
        raise TypeError("audits must be tuple")
    if not audits:
        return ShaderBakeCapability.LOCAL_UV_SAFE
    return strongest_shader_capability(
        audit.required_capability for audit in audits
    )


def capability_failure_message(
    audits: Tuple[MaterialCapabilityAudit, ...],
    capability: ShaderBakeCapability,
) -> str:
    """Build the historical deterministic failure message for an unsafe boundary."""

    if not isinstance(capability, ShaderBakeCapability):
        raise TypeError("capability must be ShaderBakeCapability")
    details = []
    for audit in audits:
        if audit.required_capability is not capability:
            continue
        codes = tuple(
            finding.code
            for finding in audit.findings
            if finding.capability is capability
        )
        details.append((audit.material_name, codes))
    return f"shader capability {capability.value} prevents safe export: {tuple(details)}"


def _supports_normal_uv_object_bake(
    finding: ShaderCapabilityFinding,
) -> bool:
    """Return whether one camera-capability finding is reproducible by object bake."""

    if not isinstance(finding, ShaderCapabilityFinding):
        raise TypeError("finding must be ShaderCapabilityFinding")
    if finding.capability is not ShaderBakeCapability.CAMERA_RENDER_REQUIRED:
        raise ValueError("finding must use CAMERA_RENDER_REQUIRED capability")

    if finding.code == "SOURCE_OR_CAMERA_CONTEXT":
        return (finding.node_type or "").strip().upper() in (
            _NORMAL_UV_SOURCE_CONTEXT_NODE_TYPES
        )
    if finding.code == "TEXTURE_COORD_SOURCE_CONTEXT":
        return (finding.output_socket or "").strip().casefold() in (
            _NORMAL_UV_TEXTURE_COORD_OUTPUTS
        )
    if finding.code == "GEOMETRY_SOURCE_CONTEXT":
        return (finding.output_socket or "").strip().casefold() in (
            _NORMAL_UV_GEOMETRY_OUTPUTS
        )
    if finding.code == PRINCIPLED_REFLECTION_CONTEXT_CODE:
        return (
            (finding.node_type or "").strip().upper() == "BSDF_PRINCIPLED"
            and (finding.output_socket or "").strip()
            in PRINCIPLED_REFLECTION_CONTEXT_OUTPUTS
        )
    if finding.code == PRINCIPLED_TRANSMISSION_CONTEXT_CODE:
        return (
            (finding.node_type or "").strip().upper() == "BSDF_PRINCIPLED"
            and (finding.output_socket or "").strip()
            in PRINCIPLED_TRANSMISSION_CONTEXT_OUTPUTS
        )
    return False


def _normal_uv_blocking_camera_findings(
    audits: Tuple[MaterialCapabilityAudit, ...],
) -> tuple[tuple[str, tuple[tuple[str, str | None, str | None], ...]], ...]:
    """Return camera findings that cannot be reproduced by Normal object UV bake.

    ``GRAPH_CAMERA_DEPENDENCY`` is only an aggregate. It is ignored when concrete
    camera findings explain the dependency, but remains fail-closed if no concrete
    cause was classified. This prevents unknown camera dependencies from silently
    entering the Normal/UV route.
    """

    if not isinstance(audits, tuple) or not all(
        isinstance(audit, MaterialCapabilityAudit) for audit in audits
    ):
        raise TypeError("audits must contain MaterialCapabilityAudit values")

    details = []
    for audit in audits:
        camera_findings = tuple(
            finding
            for finding in audit.findings
            if finding.capability is ShaderBakeCapability.CAMERA_RENDER_REQUIRED
        )
        if not camera_findings:
            continue

        concrete = tuple(
            finding
            for finding in camera_findings
            if finding.code != _GRAPH_CAMERA_AGGREGATE_CODE
        )
        blocked = tuple(
            finding
            for finding in concrete
            if not _supports_normal_uv_object_bake(finding)
        )
        if not concrete:
            blocked = camera_findings
        if not blocked:
            continue

        details.append(
            (
                audit.material_name,
                tuple(
                    (
                        finding.code,
                        finding.node_type,
                        finding.output_socket,
                    )
                    for finding in blocked
                ),
            )
        )
    return tuple(details)


def normal_mode_camera_requirement_message(
    audits: Tuple[MaterialCapabilityAudit, ...],
) -> str:
    """Explain why specific camera findings require a camera-render mode."""

    details = _normal_uv_blocking_camera_findings(audits)
    return (
        "Normal — UV Segments can bake audited source-object surface context, but "
        "cannot reproduce these volume, displacement, unsupported source attribute, "
        "group/instance, or unclassified camera findings. Select Export Mode: Camera "
        "Projection or Depth Camera Projection. "
        f"Blocking findings: {details}"
    )


def build_capability_checked_texture_plan(
    analysis: ObjectMaterialAnalysis,
    settings: BakeSettings,
    audits: Tuple[MaterialCapabilityAudit, ...],
    renderer: RenderEngineContract,
    *,
    object_context: ObjectBakeContext,
    scene_context: SceneBakeContext,
    texture_export_mode: A1TextureExportMode = (
        A1TextureExportMode.NORMAL_UV_SEGMENTS
    ),
) -> TexturePlan:
    """Select explicit Normal UV baking or one of the camera-render representations."""

    if not isinstance(analysis, ObjectMaterialAnalysis):
        raise TypeError("analysis must be ObjectMaterialAnalysis")
    if not isinstance(settings, BakeSettings):
        raise TypeError("settings must be BakeSettings")
    if not isinstance(renderer, RenderEngineContract):
        raise TypeError("renderer must be RenderEngineContract")
    if not isinstance(object_context, ObjectBakeContext):
        raise TypeError("object_context must be ObjectBakeContext")
    if not isinstance(scene_context, SceneBakeContext):
        raise TypeError("scene_context must be SceneBakeContext")
    if not isinstance(texture_export_mode, A1TextureExportMode):
        raise TypeError("texture_export_mode must be A1TextureExportMode")

    capability = strongest_object_capability(audits)
    if capability in {
        ShaderBakeCapability.UNSUPPORTED,
        ShaderBakeCapability.GROUP_RENDER_REQUIRED,
    }:
        raise BakePlanError(capability_failure_message(audits, capability))

    if texture_export_mode in _CAMERA_RENDER_MODES:
        return build_camera_projection_plan(
            analysis,
            settings,
            object_context=object_context,
            scene_context=scene_context,
        )

    if capability is ShaderBakeCapability.CAMERA_RENDER_REQUIRED:
        if _normal_uv_blocking_camera_findings(audits):
            raise BakePlanError(normal_mode_camera_requirement_message(audits))
        return build_normal_uv_camera_context_plan(
            analysis,
            settings,
            object_context=object_context,
            scene_context=scene_context,
        )

    return build_bake_plan(
        analysis,
        settings,
        object_context=object_context,
        scene_context=scene_context,
    )


__all__ = [
    "build_capability_checked_texture_plan",
    "capability_failure_message",
    "normal_mode_camera_requirement_message",
    "strongest_object_capability",
]
