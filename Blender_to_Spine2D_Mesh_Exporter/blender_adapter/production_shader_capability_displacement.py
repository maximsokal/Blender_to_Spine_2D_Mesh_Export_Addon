"""Refine connected Material Output displacement using live Blender material state.

The immutable shader graph can prove that Material Output -> Displacement is connected,
but it cannot determine whether Blender evaluates that channel as shading-only bump or as
true geometry displacement. Normal/UV can reproduce bump-only appearance through the
camera-aware Cycles COMBINED object-bake route. True displacement changes vertex positions
before rendering and therefore remains incompatible with unchanged Spine export geometry.
"""

from __future__ import annotations

from typing import Any

from ..domain.baking.capabilities import (
    MaterialCapabilityAudit,
    ShaderBakeCapability,
    strongest_shader_capability,
)
from .shader_capability_findings import build_finding, order_unique_findings


DISPLACEMENT_RENDER_REQUIRED_CODE = "DISPLACEMENT_RENDER_REQUIRED"
DISPLACEMENT_BUMP_CONTEXT_CODE = "DISPLACEMENT_BUMP_CONTEXT"

_BUMP_ONLY_METHOD = "BUMP"
_TRUE_DISPLACEMENT_METHODS = frozenset({"DISPLACEMENT", "BOTH", "TRUE"})
_SUPPORTED_METHODS = frozenset({_BUMP_ONLY_METHOD}) | _TRUE_DISPLACEMENT_METHODS


def _normalise_displacement_method(value: Any) -> str | None:
    """Return one known Blender displacement method or ``None`` fail-closed.

    Blender 5.2 exposes ``Material.displacement_method``. Older Cycles integrations used
    ``Material.cycles.displacement_method`` and some historical builds exposed ``TRUE``
    for displacement-only. Supporting both access paths costs nothing and keeps this live
    boundary robust without weakening the minimum Blender 5.2 contract.
    """

    if not isinstance(value, str):
        return None
    resolved = value.strip().upper()
    return resolved if resolved in _SUPPORTED_METHODS else None


def material_displacement_method(material: Any) -> str | None:
    """Read Blender's live displacement method without guessing unavailable RNA."""

    if material is None:
        raise TypeError("material cannot be None")

    direct = _normalise_displacement_method(
        getattr(material, "displacement_method", None)
    )
    if direct is not None:
        return direct

    cycles = getattr(material, "cycles", None)
    legacy = _normalise_displacement_method(
        getattr(cycles, "displacement_method", None)
        if cycles is not None
        else None
    )
    return legacy


def apply_displacement_method_boundary(
    audit: MaterialCapabilityAudit,
    material: Any,
) -> MaterialCapabilityAudit:
    """Replace blanket displacement blocking only for Blender ``BUMP`` materials.

    Unknown RNA and true displacement modes deliberately preserve the original
    ``DISPLACEMENT_RENDER_REQUIRED`` finding. This keeps Normal/UV fail-closed whenever
    exported source geometry would not match render-time geometry.
    """

    if not isinstance(audit, MaterialCapabilityAudit):
        raise TypeError("audit must be MaterialCapabilityAudit")
    if material is None:
        raise TypeError("material cannot be None")

    displacement_findings = tuple(
        finding
        for finding in audit.findings
        if finding.code == DISPLACEMENT_RENDER_REQUIRED_CODE
    )
    if not displacement_findings:
        return audit

    method = material_displacement_method(material)
    if method != _BUMP_ONLY_METHOD:
        return audit

    retained = tuple(
        finding
        for finding in audit.findings
        if finding.code != DISPLACEMENT_RENDER_REQUIRED_CODE
    )
    replacement = build_finding(
        ShaderBakeCapability.CAMERA_RENDER_REQUIRED,
        DISPLACEMENT_BUMP_CONTEXT_CODE,
        (
            "Material Output displacement is configured as Bump Only; Blender changes "
            "surface shading normals without moving mesh vertices, so Cycles COMBINED "
            "object bake can reproduce the appearance on unchanged Normal/UV geometry"
        ),
    )
    ordered = order_unique_findings(retained + (replacement,))
    return MaterialCapabilityAudit(
        material_name=audit.material_name,
        render_target=audit.render_target,
        required_capability=strongest_shader_capability(
            finding.capability for finding in ordered
        ),
        findings=ordered,
    )


__all__ = [
    "DISPLACEMENT_BUMP_CONTEXT_CODE",
    "DISPLACEMENT_RENDER_REQUIRED_CODE",
    "apply_displacement_method_boundary",
    "material_displacement_method",
]
