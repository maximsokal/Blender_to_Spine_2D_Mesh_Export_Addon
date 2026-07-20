"""Resolve the effective shader graph or the legacy root-node fallback."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Tuple

from ..domain.baking import MaterialGraphSnapshot
from .material_analysis_rna import is_temporary_node
from .shader_graph_analysis import analyse_material_graph_detailed
from .shader_graph_error import MaterialGraphAnalysisError


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class MaterialGraphResolution:
    """Graph snapshot, selected live nodes, and diagnostic graph issues."""

    graph: MaterialGraphSnapshot | None
    classification_nodes: Tuple[Any, ...]
    issues: Tuple[str, ...]

    def __post_init__(self) -> None:
        if self.graph is not None and not isinstance(
            self.graph,
            MaterialGraphSnapshot,
        ):
            raise TypeError("graph must be MaterialGraphSnapshot or None")
        if not isinstance(self.classification_nodes, tuple):
            raise TypeError("classification_nodes must be tuple")
        if not isinstance(self.issues, tuple) or not all(
            isinstance(value, str) and value for value in self.issues
        ):
            raise TypeError("issues must be a tuple of non-empty strings")


def resolve_material_graph(
    material: Any,
    root_nodes: tuple[Any, ...],
    *,
    render_target: str,
    material_name: str,
) -> MaterialGraphResolution:
    """Resolve reachable nodes while preserving the historical no-output fallback."""

    if not isinstance(root_nodes, tuple):
        raise TypeError("root_nodes must be tuple")
    graph: MaterialGraphSnapshot | None = None
    graph_nodes: tuple[Any, ...] | None = None
    graph_issues: list[str] = []

    try:
        detailed = analyse_material_graph_detailed(
            material,
            render_target=render_target,
        )
        graph = detailed.snapshot
        graph_issues.extend(graph.issues)
        if graph.active_output_node_id is None and not graph.semantic_channels:
            # Orphaned group content is diagnostic data, not an active material program.
            graph = None
        else:
            graph_nodes = detailed.reachable_nodes
    except MaterialGraphAnalysisError as exc:
        graph_issues.append(f"Shader graph analysis failed: {exc}")
        logger.warning(
            "Shader graph analysis failed for material '%s'",
            material_name,
            exc_info=True,
        )

    classification_nodes = (
        graph_nodes
        if graph_nodes is not None
        else tuple(node for node in root_nodes if not is_temporary_node(node))
    )
    return MaterialGraphResolution(
        graph=graph,
        classification_nodes=classification_nodes,
        issues=tuple(graph_issues),
    )


__all__ = [
    "MaterialGraphResolution",
    "resolve_material_graph",
]
