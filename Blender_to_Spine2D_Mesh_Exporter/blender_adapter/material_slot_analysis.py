"""Analyze one Blender material slot into the immutable baking domain."""

from __future__ import annotations

from typing import Any

from ..domain.baking import MaterialAnalysis, MaterialKind
from .material_analysis_rna import (
    material_name,
    material_root_nodes,
    resolve_render_target,
)
from .material_graph_resolution import resolve_material_graph
from .material_node_classification import classify_material_nodes


def analyse_material_slot(
    slot_index: int,
    material: Any | None,
    *,
    render_target: str | None = None,
) -> MaterialAnalysis:
    """Analyze one Blender material slot without modifying its material."""

    if not isinstance(slot_index, int) or slot_index < 0:
        raise ValueError("slot_index must be a non-negative integer")
    if material is None:
        return MaterialAnalysis(
            slot_index=slot_index,
            material_name=None,
            kind=MaterialKind.EMPTY,
            issues=("Material slot is empty",),
        )

    resolved_material_name = material_name(material)
    node_tree = getattr(material, "node_tree", None)
    use_nodes = bool(getattr(material, "use_nodes", node_tree is not None))
    if not use_nodes or node_tree is None:
        return MaterialAnalysis(
            slot_index=slot_index,
            material_name=resolved_material_name,
            kind=MaterialKind.SOLID_COLOR,
            issues=("Material has no node tree; diffuse_color fallback is required",),
        )

    root_nodes = material_root_nodes(
        material,
        resolved_name=resolved_material_name,
    )
    target = resolve_render_target(render_target)
    graph_resolution = resolve_material_graph(
        material,
        root_nodes,
        render_target=target,
        material_name=resolved_material_name,
    )
    classification = classify_material_nodes(
        graph_resolution.classification_nodes
    )
    issues = tuple(
        dict.fromkeys(classification.issues + graph_resolution.issues)
    )

    return MaterialAnalysis(
        slot_index=slot_index,
        material_name=resolved_material_name,
        kind=classification.kind,
        node_types=classification.node_types,
        image_dependencies=classification.image_dependencies,
        issues=issues,
        graph=graph_resolution.graph,
    )


__all__ = ["analyse_material_slot"]
