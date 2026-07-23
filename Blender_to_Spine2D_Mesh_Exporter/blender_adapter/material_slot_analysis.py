"""Analyze one Blender 5.2 material slot into the immutable baking domain."""

from __future__ import annotations

from typing import Any

from ..domain.baking import MaterialAnalysis, MaterialKind
from .material_analysis_error import MaterialAnalysisError
from .material_analysis_rna import (
    material_name,
    material_root_nodes,
    require_render_target,
)
from .material_graph_resolution import resolve_material_graph
from .material_node_classification import classify_material_nodes
from .shader_graph_error import MaterialGraphAnalysisError


def analyse_material_slot(
    slot_index: int,
    material: Any | None,
    *,
    render_target: str,
) -> MaterialAnalysis:
    """Analyze one Blender 5.2 material slot for one explicit renderer target."""

    if not isinstance(slot_index, int) or isinstance(slot_index, bool) or slot_index < 0:
        raise ValueError("slot_index must be a non-negative integer")
    target = require_render_target(render_target)
    if material is None:
        return MaterialAnalysis(
            slot_index=slot_index,
            material_name=None,
            kind=MaterialKind.EMPTY,
            issues=("Material slot is empty",),
        )

    resolved_material_name = material_name(material)
    node_tree = getattr(material, "node_tree", None)
    if node_tree is None:
        raise MaterialAnalysisError(
            f"Material '{resolved_material_name}' has no node tree; "
            "Blender 5.2+ materials must expose a valid node graph"
        )

    try:
        root_nodes = material_root_nodes(
            material,
            resolved_name=resolved_material_name,
        )
        graph_resolution = resolve_material_graph(
            material,
            root_nodes,
            render_target=target,
            material_name=resolved_material_name,
        )
        classification = classify_material_nodes(
            graph_resolution.classification_nodes
        )
    except MaterialAnalysisError:
        raise
    except MaterialGraphAnalysisError as exc:
        raise MaterialAnalysisError(
            f"Unable to analyze material '{resolved_material_name}' for slot "
            f"{slot_index} and target '{target}': {exc}"
        ) from exc
    except Exception as exc:
        raise MaterialAnalysisError(
            f"Unexpected Blender 5.2 material analysis failure for "
            f"'{resolved_material_name}' in slot {slot_index} and target "
            f"'{target}': {exc}"
        ) from exc

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
