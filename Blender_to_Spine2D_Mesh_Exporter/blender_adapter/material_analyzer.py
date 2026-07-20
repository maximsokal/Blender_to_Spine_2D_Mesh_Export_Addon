"""Compatibility facade for the decomposed Blender material-analysis pipeline."""

from .material_analysis_error import MaterialAnalysisError
from .material_analysis_rna import (
    is_temporary_node as _is_temporary_bake_node,
    material_name as _material_name,
    node_type as _node_type,
    normalise_render_target as _normalise_render_target,
    render_target_from_engine,
    resolve_render_target as _resolve_render_target,
)
from .material_node_classification import (
    PROCEDURAL_NODE_TYPES as _PROCEDURAL_NODE_TYPES,
    classify_nodes_legacy as _classify_nodes,
    read_image_dependency as _image_dependency,
)
from .material_object_analysis import analyse_object_materials
from .material_slot_analysis import analyse_material_slot


__all__ = [
    "MaterialAnalysisError",
    "_PROCEDURAL_NODE_TYPES",
    "_classify_nodes",
    "_image_dependency",
    "_is_temporary_bake_node",
    "_material_name",
    "_node_type",
    "_normalise_render_target",
    "_resolve_render_target",
    "analyse_material_slot",
    "analyse_object_materials",
    "render_target_from_engine",
]
