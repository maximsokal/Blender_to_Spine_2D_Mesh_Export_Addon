"""Orchestrate deterministic capability auditing of immutable shader graphs."""

from __future__ import annotations

from ..domain.baking.capabilities import (
    MaterialCapabilityAudit,
    ShaderBakeCapability,
    ShaderCapabilityFinding,
    strongest_shader_capability,
)
from ..domain.baking.graph import (
    MaterialGraphSnapshot,
    MaterialSemanticChannel,
)
from .shader_capability_findings import (
    build_finding,
    order_unique_findings,
    used_outputs,
)
from .shader_capability_node_findings import node_findings
from .shader_capability_policy import (
    CAMERA_DEPENDENCIES,
    SCENE_DEPENDENCIES,
    normalise_render_target,
)


def audit_material_graph_capabilities(
    graph: MaterialGraphSnapshot,
    *,
    render_target: str,
) -> MaterialCapabilityAudit:
    """Return a deterministic capability report for one renderer-specific graph."""

    if not isinstance(graph, MaterialGraphSnapshot):
        raise TypeError("graph must be MaterialGraphSnapshot")
    target = normalise_render_target(render_target)
    findings: list[ShaderCapabilityFinding] = []

    if graph.active_output_node_id is None:
        findings.append(
            build_finding(
                ShaderBakeCapability.UNSUPPORTED,
                "MATERIAL_OUTPUT_MISSING",
                "Renderer-specific Material Output could not be resolved",
            )
        )
    for issue in graph.issues:
        findings.append(
            build_finding(
                ShaderBakeCapability.UNSUPPORTED,
                "GRAPH_ANALYSIS_INCOMPLETE",
                issue,
            )
        )

    outputs_by_node = used_outputs(graph)
    for node in graph.reachable_nodes:
        findings.extend(
            node_findings(
                node,
                outputs_by_node.get(node.node_id, ()),
                render_target=target,
            )
        )

    dependencies = set(graph.dependencies)
    if dependencies & CAMERA_DEPENDENCIES:
        findings.append(
            build_finding(
                ShaderBakeCapability.CAMERA_RENDER_REQUIRED,
                "GRAPH_CAMERA_DEPENDENCY",
                "Graph dependencies require active-camera render evaluation",
            )
        )
    elif dependencies & SCENE_DEPENDENCIES:
        findings.append(
            build_finding(
                ShaderBakeCapability.SCENE_UV_SAFE,
                "GRAPH_SCENE_DEPENDENCY",
                "Graph dependencies require scene-aware object baking",
            )
        )

    if MaterialSemanticChannel.VOLUME in graph.semantic_channels:
        findings.append(
            build_finding(
                ShaderBakeCapability.CAMERA_RENDER_REQUIRED,
                "VOLUME_RENDER_REQUIRED",
                "Volume output cannot be represented by Blender object UV bake",
            )
        )
    if MaterialSemanticChannel.DISPLACEMENT in graph.semantic_channels:
        findings.append(
            build_finding(
                ShaderBakeCapability.CAMERA_RENDER_REQUIRED,
                "DISPLACEMENT_RENDER_REQUIRED",
                "Render-evaluated displacement requires camera projection",
            )
        )

    if not findings:
        findings.append(
            build_finding(
                ShaderBakeCapability.LOCAL_UV_SAFE,
                "LOCAL_GRAPH",
                "Reachable graph uses only audited local UV-bake-safe nodes",
            )
        )

    ordered = order_unique_findings(findings)
    return MaterialCapabilityAudit(
        material_name=graph.material_name,
        render_target=target,
        required_capability=strongest_shader_capability(
            finding.capability for finding in ordered
        ),
        findings=ordered,
    )


__all__ = ["audit_material_graph_capabilities"]
