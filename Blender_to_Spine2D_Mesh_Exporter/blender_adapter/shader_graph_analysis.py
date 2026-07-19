"""Orchestrate renderer-specific recursive Blender shader-graph analysis."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

from ..domain.baking.graph import MaterialGraphSnapshot
from .shader_graph_error import MaterialGraphAnalysisError
from .shader_graph_rna import (
    find_material_output,
    iter_nodes,
    material_name,
    node_name,
    normalise_render_target,
)
from .shader_graph_semantics import (
    derive_material_dependencies,
    derive_semantic_channels,
)
from .shader_graph_snapshot import build_material_graph_snapshot
from .shader_graph_traversal import RecursiveShaderGraphWalker


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class MaterialGraphAnalysisResult:
    """Adapter-only graph result containing the snapshot and live nodes."""

    snapshot: MaterialGraphSnapshot
    reachable_nodes: tuple[Any, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot, MaterialGraphSnapshot):
            raise TypeError("snapshot must be MaterialGraphSnapshot")
        if not isinstance(self.reachable_nodes, tuple):
            raise TypeError("reachable_nodes must be tuple")


def analyse_material_graph_detailed(
    material: Any,
    *,
    render_target: str = "ALL",
) -> MaterialGraphAnalysisResult:
    """Analyze a material and retain adapter-private reachable live nodes."""

    if material is None:
        raise MaterialGraphAnalysisError("material cannot be None")
    resolved_material_name = material_name(material)
    node_tree = getattr(material, "node_tree", None)
    if node_tree is None:
        raise MaterialGraphAnalysisError(
            f"Material '{resolved_material_name}' has no node tree"
        )

    target = normalise_render_target(render_target)
    nodes = iter_nodes(node_tree)
    output = find_material_output(node_tree, nodes, target)
    walker = RecursiveShaderGraphWalker(
        resolved_material_name,
        node_tree,
    )
    if output is None:
        walker.issues.append(
            f"Material Output for render target '{target}' was not found; "
            "semantic analysis used all nodes"
        )
        walker.walk_all_nodes()
    else:
        walker.walk_material_output(output)

    traversal = walker.build_result()
    semantic_channels = derive_semantic_channels(traversal)
    dependencies = derive_material_dependencies(material, traversal)
    snapshot, live_nodes = build_material_graph_snapshot(
        material_name=resolved_material_name,
        active_output_node_id=(
            None if output is None else node_name(output)
        ),
        traversal=traversal,
        semantic_channels=semantic_channels,
        dependencies=dependencies,
    )

    logger.debug(
        "Analyzed recursive shader graph '%s' target=%s: nodes=%d "
        "channels=%s dependencies=%s issues=%s",
        resolved_material_name,
        target,
        len(snapshot.reachable_nodes),
        tuple(value.value for value in snapshot.semantic_channels),
        tuple(value.value for value in snapshot.dependencies),
        snapshot.issues,
    )
    return MaterialGraphAnalysisResult(
        snapshot=snapshot,
        reachable_nodes=live_nodes,
    )


def analyse_material_graph(
    material: Any,
    *,
    render_target: str = "ALL",
) -> MaterialGraphSnapshot:
    """Analyze reachable nodes, recursively expanding used Shader Node Groups."""

    return analyse_material_graph_detailed(
        material,
        render_target=render_target,
    ).snapshot


__all__ = [
    "MaterialGraphAnalysisResult",
    "analyse_material_graph",
    "analyse_material_graph_detailed",
]
