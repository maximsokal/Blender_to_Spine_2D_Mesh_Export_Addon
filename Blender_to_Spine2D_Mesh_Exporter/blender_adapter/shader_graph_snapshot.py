"""Deterministic immutable snapshot assembly for reachable shader graphs."""

from __future__ import annotations

from typing import Any

from ..domain.baking.graph import (
    MaterialDependencyKind,
    MaterialGraphSnapshot,
    MaterialSemanticChannel,
    ShaderLinkSnapshot,
    ShaderNodeSnapshot,
)
from .shader_graph_error import MaterialGraphAnalysisError
from .shader_graph_rna import node_name, node_type
from .shader_graph_traversal import (
    ReachableShaderNode,
    ShaderGraphTraversalResult,
)


def ordered_reachable_nodes(
    traversal: ShaderGraphTraversalResult,
) -> tuple[ReachableShaderNode, ...]:
    if not isinstance(traversal, ShaderGraphTraversalResult):
        raise TypeError("traversal must be ShaderGraphTraversalResult")
    return tuple(
        item
        for _, item in sorted(
            traversal.nodes.items(),
            key=lambda pair: pair[0].casefold(),
        )
    )


def build_shader_node_snapshots(
    ordered_nodes: tuple[ReachableShaderNode, ...],
) -> tuple[ShaderNodeSnapshot, ...]:
    if not isinstance(ordered_nodes, tuple):
        raise TypeError("ordered_nodes must be tuple")
    return tuple(
        ShaderNodeSnapshot(
            node_id=item.node_id,
            node_type=node_type(item.node),
            node_name=node_name(item.node),
            group_path=item.frame.group_path,
        )
        for item in ordered_nodes
    )


def build_shader_link_snapshots(
    traversal: ShaderGraphTraversalResult,
) -> tuple[ShaderLinkSnapshot, ...]:
    if not isinstance(traversal, ShaderGraphTraversalResult):
        raise TypeError("traversal must be ShaderGraphTraversalResult")
    return tuple(
        traversal.links[key]
        for key in sorted(
            traversal.links,
            key=lambda item: tuple(component.casefold() for component in item),
        )
    )


def build_material_graph_snapshot(
    *,
    material_name: str,
    active_output_node_id: str | None,
    traversal: ShaderGraphTraversalResult,
    semantic_channels: tuple[MaterialSemanticChannel, ...],
    dependencies: tuple[MaterialDependencyKind, ...],
) -> tuple[MaterialGraphSnapshot, tuple[Any, ...]]:
    """Build one snapshot and its exactly parallel ordered live-node tuple."""

    if not isinstance(material_name, str) or not material_name.strip():
        raise ValueError("material_name must be a non-empty string")
    if not isinstance(traversal, ShaderGraphTraversalResult):
        raise TypeError("traversal must be ShaderGraphTraversalResult")

    ordered_nodes = ordered_reachable_nodes(traversal)
    try:
        snapshot = MaterialGraphSnapshot(
            material_name=material_name,
            active_output_node_id=active_output_node_id,
            reachable_nodes=build_shader_node_snapshots(ordered_nodes),
            reachable_links=build_shader_link_snapshots(traversal),
            semantic_channels=semantic_channels,
            dependencies=dependencies,
            issues=tuple(dict.fromkeys(traversal.issues)),
        )
    except MaterialGraphAnalysisError:
        raise
    except Exception as exc:
        raise MaterialGraphAnalysisError(
            "Unable to build semantic graph snapshot for material "
            f"'{material_name}'"
        ) from exc

    live_nodes = tuple(item.node for item in ordered_nodes)
    if len(snapshot.reachable_nodes) != len(live_nodes):
        raise MaterialGraphAnalysisError(
            "Snapshot and live reachable-node counts diverged for material "
            f"'{material_name}'"
        )
    return snapshot, live_nodes


__all__ = [
    "build_material_graph_snapshot",
    "build_shader_link_snapshots",
    "build_shader_node_snapshots",
    "ordered_reachable_nodes",
]
