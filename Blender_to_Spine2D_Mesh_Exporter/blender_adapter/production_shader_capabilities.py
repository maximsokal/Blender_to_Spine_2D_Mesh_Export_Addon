"""Production capability gate built from live renderer-specific Blender graphs."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Tuple

from ..domain.baking import (
    MaterialCapabilityAudit,
    MaterialGraphSnapshot,
    MaterialSemanticChannel,
    ObjectMaterialAnalysis,
    ShaderBakeCapability,
    ShaderCapabilityFinding,
    strongest_shader_capability,
)
from .render_engine_contract import render_engine_contract
from .shader_capability_audit import audit_material_graph_capabilities
from .shader_graph_analyzer import analyse_material_graph_detailed


class ProductionShaderCapabilityError(RuntimeError):
    """Raised when live Blender materials and immutable analysis disagree."""


def _enriched_graph_with_live_mute(
    graph: MaterialGraphSnapshot,
    live_nodes: Tuple[Any, ...],
) -> MaterialGraphSnapshot:
    if len(graph.reachable_nodes) != len(live_nodes):
        raise ProductionShaderCapabilityError(
            "live capability graph node count differs from material analysis"
        )
    enriched = []
    for snapshot, live_node in zip(graph.reachable_nodes, live_nodes):
        live_name = str(getattr(live_node, "name", "") or "")
        if snapshot.node_name != live_name:
            raise ProductionShaderCapabilityError(
                "live capability graph order differs from material analysis; "
                f"expected={snapshot.node_name!r}, actual={live_name!r}"
            )
        enriched.append(
            replace(snapshot, muted=bool(getattr(live_node, "mute", False)))
        )
    return replace(graph, reachable_nodes=tuple(enriched))


def _with_proxy_boundary(
    audit: MaterialCapabilityAudit,
    graph: MaterialGraphSnapshot,
) -> MaterialCapabilityAudit:
    if MaterialSemanticChannel.ALPHA not in graph.semantic_channels:
        return audit

    findings = list(audit.findings)
    for node in graph.reachable_nodes:
        if node.node_type in {"GROUP", "REROUTE"}:
            findings.append(
                ShaderCapabilityFinding(
                    capability=ShaderBakeCapability.CAMERA_RENDER_REQUIRED,
                    code="ALPHA_PROXY_RECURSIVE_BOUNDARY",
                    reason=(
                        "Straight-color/opacity extraction cannot safely flatten reachable "
                        f"{node.node_type} nodes without mutating shared node-group data"
                    ),
                    node_id=node.node_id,
                    node_type=node.node_type,
                )
            )
        if node.muted:
            findings.append(
                ShaderCapabilityFinding(
                    capability=ShaderBakeCapability.CAMERA_RENDER_REQUIRED,
                    code="ALPHA_PROXY_MUTED_BYPASS",
                    reason=(
                        "Alpha proxy extraction does not reproduce Blender internal_links "
                        "bypass semantics; native camera render is required"
                    ),
                    node_id=node.node_id,
                    node_type=node.node_type,
                )
            )

    unique = {}
    for finding in findings:
        key = (
            finding.capability.value,
            finding.code,
            finding.node_id,
            finding.node_type,
            finding.output_socket,
        )
        unique.setdefault(key, finding)
    ordered = tuple(
        unique[key]
        for key in sorted(
            unique,
            key=lambda value: tuple(
                "" if item is None else item.casefold() for item in value
            ),
        )
    )
    return MaterialCapabilityAudit(
        material_name=audit.material_name,
        render_target=audit.render_target,
        required_capability=strongest_shader_capability(
            finding.capability for finding in ordered
        ),
        findings=ordered,
    )


def audit_object_material_capabilities(
    obj: Any,
    analysis: ObjectMaterialAnalysis,
    *,
    render_target: str,
) -> Tuple[MaterialCapabilityAudit, ...]:
    """Audit every used node material against its live copied-independent graph state."""

    if obj is None or getattr(obj, "type", None) != "MESH":
        raise TypeError("obj must be a Blender MESH object")
    if not isinstance(analysis, ObjectMaterialAnalysis):
        raise TypeError("analysis must be ObjectMaterialAnalysis")
    target = render_engine_contract(render_target).shader_target
    slots = tuple(getattr(obj, "material_slots", ()))
    if len(slots) != len(analysis.slots):
        raise ProductionShaderCapabilityError(
            "live material slot count differs from immutable material analysis"
        )

    audits = []
    for slot_analysis, live_slot in zip(analysis.slots, slots):
        graph = slot_analysis.graph
        material = getattr(live_slot, "material", None)
        if graph is None:
            continue
        if material is None:
            raise ProductionShaderCapabilityError(
                f"material slot {slot_analysis.slot_index} lost its material"
            )
        detailed = analyse_material_graph_detailed(
            material,
            render_target=target,
        )
        if detailed.snapshot.active_output_node_id != graph.active_output_node_id:
            raise ProductionShaderCapabilityError(
                "renderer-specific Material Output changed between analysis and planning"
            )
        enriched_graph = _enriched_graph_with_live_mute(
            graph,
            detailed.reachable_nodes,
        )
        audit = audit_material_graph_capabilities(
            enriched_graph,
            render_target=target,
        )
        audits.append(_with_proxy_boundary(audit, enriched_graph))
    return tuple(audits)


def strongest_object_capability(
    audits: Tuple[MaterialCapabilityAudit, ...],
) -> ShaderBakeCapability:
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
