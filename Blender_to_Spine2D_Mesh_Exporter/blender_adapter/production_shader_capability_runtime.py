"""Live Blender graph re-analysis and immutable parity validation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from ..domain.baking.graph import MaterialGraphSnapshot, ShaderNodeSnapshot
from .production_shader_capability_error import ProductionShaderCapabilityError
from .shader_graph_analysis import analyse_material_graph_detailed


NodeSignature = tuple[str, str, str, tuple[str, ...]]


@dataclass(frozen=True, slots=True)
class ProductionMaterialGraphAnalysis:
    """Validated production graph plus aligned live Blender nodes."""

    graph: MaterialGraphSnapshot
    live_nodes: tuple[Any, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.graph, MaterialGraphSnapshot):
            raise TypeError("graph must be MaterialGraphSnapshot")
        if not isinstance(self.live_nodes, tuple):
            raise TypeError("live_nodes must be tuple")
        if len(self.graph.reachable_nodes) != len(self.live_nodes):
            raise ValueError("graph and live_nodes must contain the same node count")


def graph_node_signature(node: ShaderNodeSnapshot) -> NodeSignature:
    """Return the immutable identity used to align recursive graph instances."""

    if not isinstance(node, ShaderNodeSnapshot):
        raise TypeError("node must be ShaderNodeSnapshot")
    return (
        node.node_id,
        node.node_type,
        node.node_name,
        node.group_path,
    )


def validate_graph_snapshot_parity(
    expected: MaterialGraphSnapshot,
    actual: MaterialGraphSnapshot,
) -> None:
    """Fail closed when a material graph changes between analysis and planning."""

    if not isinstance(expected, MaterialGraphSnapshot):
        raise TypeError("expected must be MaterialGraphSnapshot")
    if not isinstance(actual, MaterialGraphSnapshot):
        raise TypeError("actual must be MaterialGraphSnapshot")
    if actual.material_name != expected.material_name:
        raise ProductionShaderCapabilityError(
            "material name changed between analysis and planning"
        )
    if actual.active_output_node_id != expected.active_output_node_id:
        raise ProductionShaderCapabilityError(
            "renderer-specific Material Output changed between analysis and planning"
        )
    if len(actual.reachable_nodes) != len(expected.reachable_nodes):
        raise ProductionShaderCapabilityError(
            "reachable graph node count changed between analysis and planning"
        )

    for expected_node, actual_node in zip(
        expected.reachable_nodes,
        actual.reachable_nodes,
    ):
        if (
            expected_node.node_id != actual_node.node_id
            or expected_node.group_path != actual_node.group_path
        ):
            raise ProductionShaderCapabilityError(
                "reachable graph node identity changed between analysis and planning"
            )
        if expected_node.node_type != actual_node.node_type:
            raise ProductionShaderCapabilityError(
                "reachable graph node type changed between analysis and planning"
            )
        if expected_node.node_name != actual_node.node_name:
            raise ProductionShaderCapabilityError(
                "reachable graph node name changed between analysis and planning"
            )

    if actual.reachable_links != expected.reachable_links:
        raise ProductionShaderCapabilityError(
            "reachable graph links changed between analysis and planning"
        )
    if actual.semantic_channels != expected.semantic_channels:
        raise ProductionShaderCapabilityError(
            "shader semantic channels changed between analysis and planning"
        )
    if actual.dependencies != expected.dependencies:
        raise ProductionShaderCapabilityError(
            "shader dependencies changed between analysis and planning"
        )
    if actual.issues != expected.issues:
        raise ProductionShaderCapabilityError(
            "shader analysis issues changed between analysis and planning"
        )


def validate_live_node_alignment(
    graph: MaterialGraphSnapshot,
    live_nodes: tuple[Any, ...],
) -> None:
    """Validate the exact snapshot/live tuple alignment used by Blender preflight."""

    if not isinstance(graph, MaterialGraphSnapshot):
        raise TypeError("graph must be MaterialGraphSnapshot")
    if not isinstance(live_nodes, tuple):
        raise TypeError("live_nodes must be tuple")
    if len(graph.reachable_nodes) != len(live_nodes):
        raise ProductionShaderCapabilityError(
            "live capability graph node count differs from material analysis"
        )
    for snapshot, live_node in zip(graph.reachable_nodes, live_nodes):
        live_name = str(getattr(live_node, "name", "") or "")
        if snapshot.node_name != live_name:
            raise ProductionShaderCapabilityError(
                "live capability graph order differs from material analysis; "
                f"expected={snapshot.node_name!r}, actual={live_name!r}"
            )
        live_type = str(getattr(live_node, "type", "") or "")
        if live_type and snapshot.node_type != live_type:
            raise ProductionShaderCapabilityError(
                "live capability graph node type differs from material analysis; "
                f"expected={snapshot.node_type!r}, actual={live_type!r}"
            )


def enrich_graph_with_live_mute(
    graph: MaterialGraphSnapshot,
    live_nodes: tuple[Any, ...],
) -> MaterialGraphSnapshot:
    """Copy current live mute flags into an otherwise immutable graph snapshot."""

    validate_live_node_alignment(graph, live_nodes)
    enriched = tuple(
        replace(snapshot, muted=bool(getattr(live_node, "mute", False)))
        for snapshot, live_node in zip(graph.reachable_nodes, live_nodes)
    )
    return replace(graph, reachable_nodes=enriched)


def analyse_production_material_graph(
    expected_graph: MaterialGraphSnapshot,
    material: Any,
    *,
    render_target: str,
) -> ProductionMaterialGraphAnalysis:
    """Re-analyze one live material, validate parity, and enrich current mute state."""

    if not isinstance(expected_graph, MaterialGraphSnapshot):
        raise TypeError("expected_graph must be MaterialGraphSnapshot")
    detailed = analyse_material_graph_detailed(
        material,
        render_target=render_target,
    )
    validate_graph_snapshot_parity(expected_graph, detailed.snapshot)
    validate_live_node_alignment(detailed.snapshot, detailed.reachable_nodes)
    return ProductionMaterialGraphAnalysis(
        graph=enrich_graph_with_live_mute(
            expected_graph,
            detailed.reachable_nodes,
        ),
        live_nodes=detailed.reachable_nodes,
    )


__all__ = [
    "NodeSignature",
    "ProductionMaterialGraphAnalysis",
    "analyse_production_material_graph",
    "enrich_graph_with_live_mute",
    "graph_node_signature",
    "validate_graph_snapshot_parity",
    "validate_live_node_alignment",
]
