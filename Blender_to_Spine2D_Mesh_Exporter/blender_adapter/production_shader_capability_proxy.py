"""Alpha-proxy capability boundaries for production material preparation."""

from __future__ import annotations

from ..domain.baking.capabilities import (
    MaterialCapabilityAudit,
    ShaderBakeCapability,
    ShaderCapabilityFinding,
)
from ..domain.baking.graph import MaterialGraphSnapshot, MaterialSemanticChannel
from .production_shader_capability_merge import extend_material_capability_audit
from .shader_capability_findings import build_finding


def build_alpha_proxy_findings(
    graph: MaterialGraphSnapshot,
) -> tuple[ShaderCapabilityFinding, ...]:
    """Describe graph constructs that cannot be flattened by an Alpha proxy safely."""

    if not isinstance(graph, MaterialGraphSnapshot):
        raise TypeError("graph must be MaterialGraphSnapshot")
    if MaterialSemanticChannel.ALPHA not in graph.semantic_channels:
        return ()

    findings: list[ShaderCapabilityFinding] = []
    for node in graph.reachable_nodes:
        if node.node_type in {"GROUP", "REROUTE"}:
            findings.append(
                build_finding(
                    ShaderBakeCapability.CAMERA_RENDER_REQUIRED,
                    "ALPHA_PROXY_RECURSIVE_BOUNDARY",
                    (
                        "Straight-color/opacity extraction cannot safely flatten reachable "
                        f"{node.node_type} nodes without mutating shared node-group data"
                    ),
                    node=node,
                )
            )
        if node.muted:
            findings.append(
                build_finding(
                    ShaderBakeCapability.CAMERA_RENDER_REQUIRED,
                    "ALPHA_PROXY_MUTED_BYPASS",
                    (
                        "Alpha proxy extraction does not reproduce Blender internal_links "
                        "bypass semantics; native camera render is required"
                    ),
                    node=node,
                )
            )
    return tuple(findings)


def apply_alpha_proxy_boundary(
    audit: MaterialCapabilityAudit,
    graph: MaterialGraphSnapshot,
) -> MaterialCapabilityAudit:
    """Promote an audit when Alpha proxy extraction cannot preserve graph semantics."""

    findings = build_alpha_proxy_findings(graph)
    return (
        extend_material_capability_audit(audit, findings)
        if findings
        else audit
    )


__all__ = [
    "apply_alpha_proxy_boundary",
    "build_alpha_proxy_findings",
]
