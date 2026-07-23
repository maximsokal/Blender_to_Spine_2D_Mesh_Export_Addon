"""Resolve one strict Blender 5.2+ material shader graph."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

from ..domain.baking import MaterialGraphSnapshot
from .shader_graph_analysis import analyse_material_graph_detailed


@dataclass(frozen=True, slots=True)
class MaterialGraphResolution:
    """Graph snapshot, selected live nodes, and graph diagnostics."""

    graph: MaterialGraphSnapshot
    classification_nodes: Tuple[Any, ...]
    issues: Tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.graph, MaterialGraphSnapshot):
            raise TypeError("graph must be MaterialGraphSnapshot")
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
    """Resolve only nodes reachable from the effective Material Output.

    ``root_nodes`` is retained in the public adapter signature because callers
    freeze the Blender collection once before analysis. It is validated here but
    is no longer used as a fallback when recursive analysis fails.
    """

    if not isinstance(root_nodes, tuple):
        raise TypeError("root_nodes must be tuple")
    if not isinstance(material_name, str) or not material_name.strip():
        raise ValueError("material_name must be a non-empty string")

    detailed = analyse_material_graph_detailed(
        material,
        render_target=render_target,
    )
    graph = detailed.snapshot
    if graph.material_name != material_name:
        raise ValueError(
            "Material graph name changed during analysis: "
            f"expected={material_name!r}, actual={graph.material_name!r}"
        )
    if graph.active_output_node_id is None:
        raise ValueError(
            f"Material '{material_name}' graph has no active output node"
        )

    return MaterialGraphResolution(
        graph=graph,
        classification_nodes=detailed.reachable_nodes,
        issues=graph.issues,
    )


__all__ = [
    "MaterialGraphResolution",
    "resolve_material_graph",
]
