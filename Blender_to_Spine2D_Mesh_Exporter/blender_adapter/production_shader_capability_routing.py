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
    TexturePlan,
    build_camera_projection_plan,
    build_texture_plan,
    strongest_shader_capability,
)
from .render_engine_contract import RenderEngineContract


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


def normal_mode_camera_requirement_message(
    audits: Tuple[MaterialCapabilityAudit, ...],
) -> str:
    """Explain why an explicit Normal export cannot silently become B4."""

    details = []
    for audit in audits:
        codes = tuple(
            finding.code
            for finding in audit.findings
            if finding.capability is ShaderBakeCapability.CAMERA_RENDER_REQUIRED
        )
        if codes:
            details.append((audit.material_name, codes))
    return (
        "Normal — UV Segments cannot reproduce this material without changing "
        "the exported topology. Select Export Mode: Camera Projection. "
        f"Camera-dependent findings: {tuple(details)}"
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
    """Select Normal UV baking or B4 only from the explicit user mode."""

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

    if texture_export_mode is A1TextureExportMode.CAMERA_PROJECTION:
        return build_camera_projection_plan(
            analysis,
            settings,
            object_context=object_context,
            scene_context=scene_context,
        )

    if capability is ShaderBakeCapability.CAMERA_RENDER_REQUIRED:
        raise BakePlanError(normal_mode_camera_requirement_message(audits))

    return build_texture_plan(
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
