"""Compatibility facade for the decomposed shader capability audit pipeline."""

from .shader_capability_analysis import audit_material_graph_capabilities
from .shader_capability_findings import (
    build_finding as _finding,
    order_unique_findings,
    used_outputs as _used_outputs,
)
from .shader_capability_node_findings import (
    geometry_findings as _geometry_findings,
    node_findings as _node_findings,
    texture_coordinate_findings as _texture_coordinate_findings,
)
from .shader_capability_policy import (
    CAMERA_DEPENDENCIES as _CAMERA_DEPENDENCIES,
    CAMERA_NODE_TYPES as _CAMERA_NODE_TYPES,
    GEOMETRY_OUTPUT_CAPABILITIES as _GEOMETRY_OUTPUT_CAPABILITIES,
    GROUP_NODE_TYPES as _GROUP_NODE_TYPES,
    LOCAL_SAFE_NODE_TYPES as _LOCAL_SAFE_NODE_TYPES,
    RENDER_TARGETS as _RENDER_TARGETS,
    SCENE_DEPENDENCIES as _SCENE_DEPENDENCIES,
    SCENE_NODE_TYPES as _SCENE_NODE_TYPES,
    SOURCE_ATTRIBUTE_NODE_TYPES as _SOURCE_ATTRIBUTE_NODE_TYPES,
    TEXTURE_COORD_CAPABILITIES as _TEXTURE_COORD_CAPABILITIES,
    normalise_render_target as _normalise_render_target,
)


__all__ = [
    "_CAMERA_DEPENDENCIES",
    "_CAMERA_NODE_TYPES",
    "_GEOMETRY_OUTPUT_CAPABILITIES",
    "_GROUP_NODE_TYPES",
    "_LOCAL_SAFE_NODE_TYPES",
    "_RENDER_TARGETS",
    "_SCENE_DEPENDENCIES",
    "_SCENE_NODE_TYPES",
    "_SOURCE_ATTRIBUTE_NODE_TYPES",
    "_TEXTURE_COORD_CAPABILITIES",
    "_finding",
    "_geometry_findings",
    "_node_findings",
    "_normalise_render_target",
    "_texture_coordinate_findings",
    "_used_outputs",
    "audit_material_graph_capabilities",
    "order_unique_findings",
]
