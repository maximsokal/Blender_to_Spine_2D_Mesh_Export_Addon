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
    build_bake_plan,
    strongest_shader_capability,
)
from ..domain.baking.normal_uv_camera_context import (
    build_normal_uv_camera_context_plan,
)
from .render_engine_contract import RenderEngineContract


_NORMAL_UV_BLOCKING_CAMERA_CODES = frozenset(
    {
        "DISPLACEMENT_RENDER_REQUIRED",
        "EEVEE_SHADER_TO_RGB",
        "SOURCE_ATTRIBUTE_NOT_MATERIALIZED",
        "VOLUME_RENDER_REQUIRED",
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


def _normal_uv_blocking_camera_findings(
    audits: Tuple[MaterialCapabilityAudit, ...],
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Return only camera-capability findings object UV baking cannot represent."""

    if not isinstance(audits, tuple) or not all(
        isinstance(audit, MaterialCapabilityAudit) for audit in audits
    ):
        raise TypeError("audits must contain MaterialCapabilityAudit values")

    details = []
    for audit in audits:
        codes = tuple(
            finding.code
            for finding in audit.findings
            if finding.capability is ShaderBakeCapability.CAMERA_RENDER_REQUIRED
            and finding.code in _NORMAL_UV_BLOCKING_CAMERA_CODES
        )
        if codes:
            details.append((audit.material_name, codes))
    return tuple(details)


def normal_mode_camera_requirement_message(
    audits: Tuple[MaterialCapabilityAudit, ...],
) -> str:
    """Explain the narrow cases that still require Camera Projection topology."""

    details = _normal_uv_blocking_camera_findings(audits)
    return (
        "Normal — UV Segments can bake source/object/camera-context surface "
        "appearance, but cannot represent volume, render displacement, Eevee "
        "Shader-to-RGB, or unavailable source attributes without changing the "
        "export boundary. Select Export Mode: Camera Projection only for these "
        f"findings: {details}"
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
    """Select explicit Normal UV baking or Camera Projection without conflating them."""

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
