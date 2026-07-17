"""Production capability gate built from live renderer-specific Blender graphs."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Iterable, Tuple

from ..domain.baking import (
    BakePlanError,
    BakeSettings,
    MaterialCapabilityAudit,
    MaterialGraphSnapshot,
    MaterialSemanticChannel,
    ObjectBakeContext,
    ObjectMaterialAnalysis,
    SceneBakeContext,
    ShaderBakeCapability,
    ShaderCapabilityFinding,
    TexturePlan,
    build_camera_projection_plan,
    build_texture_plan,
    strongest_shader_capability,
)
from .render_engine_contract import RenderEngineContract, render_engine_contract
from .shader_capability_audit import audit_material_graph_capabilities
from .shader_graph_analyzer import analyse_material_graph_detailed


class ProductionShaderCapabilityError(RuntimeError):
    """Raised when live Blender materials and immutable analysis disagree."""


def _enriched_graph_with_live_mute(
    graph: MaterialGraphSnapshot,
    live_nodes: Tuple[Any, ...],
) -> MaterialGraphSnapshot:
    if len(graph.reachable_nodes) != len(live_nodes):
        raise ProductionShaderCapabilityError(
            "live capability graph node count differs from material analysis"
        )
    enriched = []
    for snapshot, live_node in zip(graph.reachable_nodes, live_nodes):
        live_name = str(getattr(live_node, "name", "") or "")
        if snapshot.node_name != live_name:
            raise ProductionShaderCapabilityError(
                "live capability graph order differs from material analysis; "
                f"expected={snapshot.node_name!r}, actual={live_name!r}"
            )
        enriched.append(
            replace(snapshot, muted=bool(getattr(live_node, "mute", False)))
        )
    return replace(graph, reachable_nodes=tuple(enriched))


def _rebuild_audit(
    audit: MaterialCapabilityAudit,
    additional_findings: Iterable[ShaderCapabilityFinding],
) -> MaterialCapabilityAudit:
    unique = {}
    for finding in tuple(audit.findings) + tuple(additional_findings):
        key = (
            finding.capability.value,
            finding.code,
            finding.node_id,
            finding.node_type,
            finding.output_socket,
        )
        unique.setdefault(key, finding)
    ordered = tuple(
        unique[key]
        for key in sorted(
            unique,
            key=lambda value: tuple(
                "" if item is None else item.casefold() for item in value
            ),
        )
    )
    return MaterialCapabilityAudit(
        material_name=audit.material_name,
        render_target=audit.render_target,
        required_capability=strongest_shader_capability(
            finding.capability for finding in ordered
        ),
        findings=ordered,
    )


def _with_proxy_boundary(
    audit: MaterialCapabilityAudit,
    graph: MaterialGraphSnapshot,
) -> MaterialCapabilityAudit:
    if MaterialSemanticChannel.ALPHA not in graph.semantic_channels:
        return audit

    findings = []
    for node in graph.reachable_nodes:
        if node.node_type in {"GROUP", "REROUTE"}:
            findings.append(
                ShaderCapabilityFinding(
                    capability=ShaderBakeCapability.CAMERA_RENDER_REQUIRED,
                    code="ALPHA_PROXY_RECURSIVE_BOUNDARY",
                    reason=(
                        "Straight-color/opacity extraction cannot safely flatten reachable "
                        f"{node.node_type} nodes without mutating shared node-group data"
                    ),
                    node_id=node.node_id,
                    node_type=node.node_type,
                )
            )
        if node.muted:
            findings.append(
                ShaderCapabilityFinding(
                    capability=ShaderBakeCapability.CAMERA_RENDER_REQUIRED,
                    code="ALPHA_PROXY_MUTED_BYPASS",
                    reason=(
                        "Alpha proxy extraction does not reproduce Blender internal_links "
                        "bypass semantics; native camera render is required"
                    ),
                    node_id=node.node_id,
                    node_type=node.node_type,
                )
            )
    return _rebuild_audit(audit, findings) if findings else audit


def _source_uv_layers(obj: Any) -> tuple[str, ...]:
    mesh = getattr(obj, "data", None)
    layers = getattr(mesh, "uv_layers", None)
    if layers is None:
        return ()
    try:
        return tuple(str(layer.name) for layer in layers)
    except Exception as exc:
        raise ProductionShaderCapabilityError(
            "unable to inspect source mesh UV layers"
        ) from exc


def _source_render_uv_name(obj: Any) -> str | None:
    mesh = getattr(obj, "data", None)
    layers = getattr(mesh, "uv_layers", None)
    if layers is None:
        return None
    try:
        resolved = tuple(layers)
    except Exception as exc:
        raise ProductionShaderCapabilityError(
            "unable to inspect source render UV layer"
        ) from exc
    render_layers = tuple(
        layer for layer in resolved if bool(getattr(layer, "active_render", False))
    )
    if len(render_layers) > 1:
        raise ProductionShaderCapabilityError(
            "source mesh reports more than one active_render UV layer"
        )
    if render_layers:
        return str(render_layers[0].name)
    active = getattr(layers, "active", None)
    return None if active is None else str(getattr(active, "name", "") or "") or None


def _input_socket(node: Any, name: str) -> Any | None:
    inputs = getattr(node, "inputs", None)
    getter = getattr(inputs, "get", None)
    if callable(getter):
        try:
            return getter(name)
        except Exception:
            return None
    return None


def _graph_uses_texture_coordinate_uv(
    graph: MaterialGraphSnapshot,
    node_id: str,
) -> bool:
    return any(
        link.from_node_id == node_id and link.from_socket.casefold() == "uv"
        for link in graph.reachable_links
    )


def _with_source_uv_boundary(
    audit: MaterialCapabilityAudit,
    graph: MaterialGraphSnapshot,
    live_nodes: Tuple[Any, ...],
    obj: Any,
) -> MaterialCapabilityAudit:
    source_layers = set(_source_uv_layers(obj))
    render_uv = _source_render_uv_name(obj)
    findings = []

    for snapshot, live_node in zip(graph.reachable_nodes, live_nodes):
        requires_default_uv = False
        if snapshot.node_type == "TEX_IMAGE":
            vector = _input_socket(live_node, "Vector")
            requires_default_uv = vector is None or not bool(
                getattr(vector, "is_linked", False)
            )
        elif snapshot.node_type == "TEX_COORD":
            requires_default_uv = _graph_uses_texture_coordinate_uv(
                graph,
                snapshot.node_id,
            )
        elif snapshot.node_type == "NORMAL_MAP":
            uv_map = str(getattr(live_node, "uv_map", "") or "").strip()
            if uv_map and uv_map not in source_layers:
                findings.append(
                    ShaderCapabilityFinding(
                        capability=ShaderBakeCapability.CAMERA_RENDER_REQUIRED,
                        code="NAMED_NORMAL_UV_MISSING",
                        reason=(
                            f"Normal Map references missing source UV layer '{uv_map}'; "
                            "native source render is required"
                        ),
                        node_id=snapshot.node_id,
                        node_type=snapshot.node_type,
                    )
                )
            requires_default_uv = not uv_map
        elif snapshot.node_type == "TANGENT":
            direction_type = str(
                getattr(live_node, "direction_type", "") or ""
            ).upper()
            uv_map = str(getattr(live_node, "uv_map", "") or "").strip()
            if direction_type == "UV_MAP" and uv_map and uv_map not in source_layers:
                findings.append(
                    ShaderCapabilityFinding(
                        capability=ShaderBakeCapability.CAMERA_RENDER_REQUIRED,
                        code="NAMED_TANGENT_UV_MISSING",
                        reason=(
                            f"Tangent node references missing source UV layer '{uv_map}'; "
                            "native source render is required"
                        ),
                        node_id=snapshot.node_id,
                        node_type=snapshot.node_type,
                    )
                )
            requires_default_uv = direction_type == "UV_MAP" and not uv_map
        elif snapshot.node_type == "UVMAP":
            uv_map = str(getattr(live_node, "uv_map", "") or "").strip()
            if uv_map and uv_map not in source_layers:
                findings.append(
                    ShaderCapabilityFinding(
                        capability=ShaderBakeCapability.CAMERA_RENDER_REQUIRED,
                        code="NAMED_UV_MISSING",
                        reason=(
                            f"UV Map node references missing source layer '{uv_map}'; "
                            "native source render is required"
                        ),
                        node_id=snapshot.node_id,
                        node_type=snapshot.node_type,
                    )
                )

        if requires_default_uv and render_uv is None:
            findings.append(
                ShaderCapabilityFinding(
                    capability=ShaderBakeCapability.CAMERA_RENDER_REQUIRED,
                    code="SOURCE_RENDER_UV_MISSING",
                    reason=(
                        f"{snapshot.node_type} requires Blender's source render UV, but the "
                        "source mesh has no render UV layer; SpineBakeUV must remain write-only"
                    ),
                    node_id=snapshot.node_id,
                    node_type=snapshot.node_type,
                )
            )

    return _rebuild_audit(audit, findings) if findings else audit


def audit_object_material_capabilities(
    obj: Any,
    analysis: ObjectMaterialAnalysis,
    *,
    render_target: str,
) -> Tuple[MaterialCapabilityAudit, ...]:
    """Audit every used node material against its live copied-independent graph state."""

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

    audits = []
    for slot_analysis, live_slot in zip(analysis.slots, slots):
        graph = slot_analysis.graph
        material = getattr(live_slot, "material", None)
        if graph is None:
            continue
        if material is None:
            raise ProductionShaderCapabilityError(
                f"material slot {slot_analysis.slot_index} lost its material"
            )
        detailed = analyse_material_graph_detailed(
            material,
            render_target=target,
        )
        if detailed.snapshot.active_output_node_id != graph.active_output_node_id:
            raise ProductionShaderCapabilityError(
                "renderer-specific Material Output changed between analysis and planning"
            )
        enriched_graph = _enriched_graph_with_live_mute(
            graph,
            detailed.reachable_nodes,
        )
        audit = audit_material_graph_capabilities(
            enriched_graph,
            render_target=target,
        )
        audit = _with_proxy_boundary(audit, enriched_graph)
        audit = _with_source_uv_boundary(
            audit,
            enriched_graph,
            detailed.reachable_nodes,
            obj,
        )
        audits.append(audit)
    return tuple(audits)


def strongest_object_capability(
    audits: Tuple[MaterialCapabilityAudit, ...],
) -> ShaderBakeCapability:
    if not isinstance(audits, tuple):
        raise TypeError("audits must be tuple")
    if not audits:
        return ShaderBakeCapability.LOCAL_UV_SAFE
    return strongest_shader_capability(
        audit.required_capability for audit in audits
    )


def capability_failure_message(
    audits: Tuple[MaterialCapabilityAudit, ...],
    capability: ShaderBakeCapability,
) -> str:
    if not isinstance(capability, ShaderBakeCapability):
        raise TypeError("capability must be ShaderBakeCapability")
    details = []
    for audit in audits:
        if audit.required_capability is not capability:
            continue
        codes = tuple(
            finding.code
            for finding in audit.findings
            if finding.capability is capability
        )
        details.append((audit.material_name, codes))
    return f"shader capability {capability.value} prevents safe export: {tuple(details)}"


def build_capability_checked_texture_plan(
    analysis: ObjectMaterialAnalysis,
    settings: BakeSettings,
    audits: Tuple[MaterialCapabilityAudit, ...],
    renderer: RenderEngineContract,
    *,
    object_context: ObjectBakeContext,
    scene_context: SceneBakeContext,
) -> TexturePlan:
    """Select B1-B4 or fail explicitly from the strongest audited capability."""

    if not isinstance(analysis, ObjectMaterialAnalysis):
        raise TypeError("analysis must be ObjectMaterialAnalysis")
    if not isinstance(settings, BakeSettings):
        raise TypeError("settings must be BakeSettings")
    if not isinstance(renderer, RenderEngineContract):
        raise TypeError("renderer must be RenderEngineContract")
    if not isinstance(object_context, ObjectBakeContext):
        raise TypeError("object_context must be ObjectBakeContext")
    if not isinstance(scene_context, SceneBakeContext):
        raise TypeError("scene_context must be SceneBakeContext")

    capability = strongest_object_capability(audits)
    if capability in {
        ShaderBakeCapability.UNSUPPORTED,
        ShaderBakeCapability.GROUP_RENDER_REQUIRED,
    }:
        raise BakePlanError(capability_failure_message(audits, capability))
    if renderer.uses_eevee or capability is ShaderBakeCapability.CAMERA_RENDER_REQUIRED:
        return build_camera_projection_plan(
            analysis,
            settings,
            object_context=object_context,
            scene_context=scene_context,
        )
    return build_texture_plan(
        analysis,
        settings,
        object_context=object_context,
        scene_context=scene_context,
    )
