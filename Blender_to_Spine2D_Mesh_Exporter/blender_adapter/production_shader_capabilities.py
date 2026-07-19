"""Compatibility facade for the decomposed production shader capability gate."""

from .production_shader_capability_error import ProductionShaderCapabilityError
from .production_shader_capability_merge import (
    extend_material_capability_audit as _rebuild_audit,
)
from .production_shader_capability_object_audit import (
    audit_object_material_capabilities,
)
from .production_shader_capability_proxy import (
    apply_alpha_proxy_boundary as _with_proxy_boundary,
    build_alpha_proxy_findings,
)
from .production_shader_capability_routing import (
    build_capability_checked_texture_plan,
    capability_failure_message,
    strongest_object_capability,
)
from .production_shader_capability_runtime import (
    ProductionMaterialGraphAnalysis,
    analyse_production_material_graph,
    enrich_graph_with_live_mute as _enriched_graph_with_live_mute,
    graph_node_signature,
    validate_graph_snapshot_parity,
    validate_live_node_alignment,
)
from .production_shader_capability_uv import (
    apply_source_uv_boundary as _with_source_uv_boundary,
    build_source_uv_findings,
    graph_uses_texture_coordinate_uv as _graph_uses_texture_coordinate_uv,
    input_socket as _input_socket,
    source_render_uv_name as _source_render_uv_name,
    source_uv_layers as _source_uv_layers,
)


__all__ = [
    "ProductionMaterialGraphAnalysis",
    "ProductionShaderCapabilityError",
    "_enriched_graph_with_live_mute",
    "_graph_uses_texture_coordinate_uv",
    "_input_socket",
    "_rebuild_audit",
    "_source_render_uv_name",
    "_source_uv_layers",
    "_with_proxy_boundary",
    "_with_source_uv_boundary",
    "analyse_production_material_graph",
    "audit_object_material_capabilities",
    "build_alpha_proxy_findings",
    "build_capability_checked_texture_plan",
    "build_source_uv_findings",
    "capability_failure_message",
    "graph_node_signature",
    "strongest_object_capability",
    "validate_graph_snapshot_parity",
    "validate_live_node_alignment",
]
