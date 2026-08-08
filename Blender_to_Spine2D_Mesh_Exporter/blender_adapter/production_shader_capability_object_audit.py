"""Object/material-slot orchestration for the live production capability gate."""

from __future__ import annotations

from typing import Any, Tuple

from ..domain.baking.capabilities import MaterialCapabilityAudit
from ..domain.baking.model import ObjectMaterialAnalysis
from .production_shader_capability_displacement import (
    apply_displacement_method_boundary,
)
from .production_shader_capability_error import ProductionShaderCapabilityError
from .production_shader_capability_principled import (
    apply_principled_context_boundary,
)
from .production_shader_capability_proxy import apply_alpha_proxy_boundary
from .production_shader_capability_runtime import analyse_production_material_graph
from .production_shader_capability_uv import apply_source_uv_boundary
from .render_engine_contract import render_engine_contract
from .shader_capability_analysis import audit_material_graph_capabilities


def audit_object_material_capabilities(
    obj: Any,
    analysis: ObjectMaterialAnalysis,
    *,
    render_target: str,
) -> Tuple[MaterialCapabilityAudit, ...]:
    """Audit every used node material against its current live Blender graph state."""

    if obj is None or getattr(obj, "type", None) != "MESH":
        raise TypeError("obj must be a Blender MESH object")
    if not isinstance(analysis, ObjectMaterialAnalysis):
        raise TypeError("analysis must be ObjectMaterialAnalysis")
    target = render_engine_contract(render_target).shader_target
    slots = tuple(getattr(obj, "material_slots", ()))
    if len(slots) != len(analysis.slots):
        raise ProductionShaderCapabilityError(
            "live material slot count differs from immutable material analysis"
        )

    audits: list[MaterialCapabilityAudit] = []
    for slot_analysis, live_slot in zip(analysis.slots, slots):
        graph = slot_analysis.graph
        material = getattr(live_slot, "material", None)
        if graph is None:
            continue
        if material is None:
            raise ProductionShaderCapabilityError(
                f"material slot {slot_analysis.slot_index} lost its material"
            )

        runtime = analyse_production_material_graph(
            graph,
            material,
            render_target=target,
        )
        audit = audit_material_graph_capabilities(
            runtime.graph,
            render_target=target,
        )
        audit = apply_principled_context_boundary(
            audit,
            runtime.graph,
            runtime.live_nodes,
        )
        audit = apply_displacement_method_boundary(audit, material)
        audit = apply_alpha_proxy_boundary(audit, runtime.graph)
        audit = apply_source_uv_boundary(
            audit,
            runtime.graph,
            runtime.live_nodes,
            obj,
        )
        audits.append(audit)
    return tuple(audits)


__all__ = ["audit_object_material_capabilities"]
