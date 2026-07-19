"""Deterministically extend immutable material capability audits."""

from __future__ import annotations

from collections.abc import Iterable

from ..domain.baking.capabilities import (
    MaterialCapabilityAudit,
    ShaderCapabilityFinding,
    strongest_shader_capability,
)
from .shader_capability_findings import order_unique_findings


def extend_material_capability_audit(
    audit: MaterialCapabilityAudit,
    additional_findings: Iterable[ShaderCapabilityFinding],
) -> MaterialCapabilityAudit:
    """Return ``audit`` enriched with findings under the shared ordering contract."""

    if not isinstance(audit, MaterialCapabilityAudit):
        raise TypeError("audit must be MaterialCapabilityAudit")
    ordered = order_unique_findings(
        tuple(audit.findings) + tuple(additional_findings)
    )
    return MaterialCapabilityAudit(
        material_name=audit.material_name,
        render_target=audit.render_target,
        required_capability=strongest_shader_capability(
            finding.capability for finding in ordered
        ),
        findings=ordered,
    )


__all__ = ["extend_material_capability_audit"]
