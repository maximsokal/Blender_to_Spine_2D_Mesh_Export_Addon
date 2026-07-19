"""Pure object-level routing from audited shader capability to B1-B4 planning."""

from __future__ import annotations

from typing import Tuple

from ..domain.baking import (
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


def build_capability_checked_texture_plan(
    analysis: ObjectMaterialAnalysis,
    settings: BakeSettings,
    audits: Tuple[MaterialCapabilityAudit, ...],
    renderer: RenderEngineContract,
    *,
    object_context: ObjectBakeContext,
    scene_context: SceneBakeContext,
) -> TexturePlan:
    """Select B1-B4 or fail explicitly from the strongest audited capability."""

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

    capability = strongest_object_capability(audits)
    if capability in {
        ShaderBakeCapability.UNSUPPORTED,
        ShaderBakeCapability.GROUP_RENDER_REQUIRED,
    }:
        raise BakePlanError(capability_failure_message(audits, capability))
    if renderer.uses_eevee or capability is ShaderBakeCapability.CAMERA_RENDER_REQUIRED:
        return build_camera_projection_plan(
            analysis,
            settings,
            object_context=object_context,
            scene_context=scene_context,
        )
    return build_texture_plan(
        analysis,
        settings,
        object_context=object_context,
        scene_context=scene_context,
    )


__all__ = [
    "build_capability_checked_texture_plan",
    "capability_failure_message",
    "strongest_object_capability",
]
