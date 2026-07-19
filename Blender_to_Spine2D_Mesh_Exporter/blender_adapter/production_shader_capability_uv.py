"""Live source-UV inspection and UV-related capability findings."""

from __future__ import annotations

from typing import Any

from ..domain.baking.capabilities import (
    MaterialCapabilityAudit,
    ShaderBakeCapability,
    ShaderCapabilityFinding,
)
from ..domain.baking.graph import MaterialGraphSnapshot
from .production_shader_capability_error import ProductionShaderCapabilityError
from .production_shader_capability_merge import extend_material_capability_audit
from .shader_capability_findings import build_finding


def source_uv_layers(obj: Any) -> tuple[str, ...]:
    """Read source mesh UV layer names without mutating Blender data."""

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


def source_render_uv_name(obj: Any) -> str | None:
    """Resolve the source render UV while rejecting inconsistent Blender state."""

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


def input_socket(node: Any, name: str) -> Any | None:
    """Read one named live Blender input socket with API/test-double tolerance."""

    inputs = getattr(node, "inputs", None)
    getter = getattr(inputs, "get", None)
    if callable(getter):
        try:
            return getter(name)
        except Exception:
            return None
    return None


def graph_uses_texture_coordinate_uv(
    graph: MaterialGraphSnapshot,
    node_id: str,
) -> bool:
    """Return whether a reachable Texture Coordinate node contributes its UV output."""

    return any(
        link.from_node_id == node_id and link.from_socket.casefold() == "uv"
        for link in graph.reachable_links
    )


def build_source_uv_findings(
    graph: MaterialGraphSnapshot,
    live_nodes: tuple[Any, ...],
    obj: Any,
) -> tuple[ShaderCapabilityFinding, ...]:
    """Build findings for live source-UV requirements absent from immutable graph data."""

    if not isinstance(graph, MaterialGraphSnapshot):
        raise TypeError("graph must be MaterialGraphSnapshot")
    if not isinstance(live_nodes, tuple):
        raise TypeError("live_nodes must be tuple")
    if len(graph.reachable_nodes) != len(live_nodes):
        raise ProductionShaderCapabilityError(
            "live capability graph node count differs from material analysis"
        )

    source_layers = set(source_uv_layers(obj))
    render_uv = source_render_uv_name(obj)
    findings: list[ShaderCapabilityFinding] = []

    for snapshot, live_node in zip(graph.reachable_nodes, live_nodes):
        requires_default_uv = False
        if snapshot.node_type == "TEX_IMAGE":
            vector = input_socket(live_node, "Vector")
            requires_default_uv = vector is None or not bool(
                getattr(vector, "is_linked", False)
            )
        elif snapshot.node_type == "TEX_COORD":
            requires_default_uv = graph_uses_texture_coordinate_uv(
                graph,
                snapshot.node_id,
            )
        elif snapshot.node_type == "NORMAL_MAP":
            uv_map = str(getattr(live_node, "uv_map", "") or "").strip()
            if uv_map and uv_map not in source_layers:
                findings.append(
                    build_finding(
                        ShaderBakeCapability.CAMERA_RENDER_REQUIRED,
                        "NAMED_NORMAL_UV_MISSING",
                        (
                            f"Normal Map references missing source UV layer '{uv_map}'; "
                            "native source render is required"
                        ),
                        node=snapshot,
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
                    build_finding(
                        ShaderBakeCapability.CAMERA_RENDER_REQUIRED,
                        "NAMED_TANGENT_UV_MISSING",
                        (
                            f"Tangent node references missing source UV layer '{uv_map}'; "
                            "native source render is required"
                        ),
                        node=snapshot,
                    )
                )
            requires_default_uv = direction_type == "UV_MAP" and not uv_map
        elif snapshot.node_type == "UVMAP":
            uv_map = str(getattr(live_node, "uv_map", "") or "").strip()
            if uv_map and uv_map not in source_layers:
                findings.append(
                    build_finding(
                        ShaderBakeCapability.CAMERA_RENDER_REQUIRED,
                        "NAMED_UV_MISSING",
                        (
                            f"UV Map node references missing source layer '{uv_map}'; "
                            "native source render is required"
                        ),
                        node=snapshot,
                    )
                )

        if requires_default_uv and render_uv is None:
            findings.append(
                build_finding(
                    ShaderBakeCapability.CAMERA_RENDER_REQUIRED,
                    "SOURCE_RENDER_UV_MISSING",
                    (
                        f"{snapshot.node_type} requires Blender's source render UV, but the "
                        "source mesh has no render UV layer; SpineBakeUV must remain write-only"
                    ),
                    node=snapshot,
                )
            )

    return tuple(findings)


def apply_source_uv_boundary(
    audit: MaterialCapabilityAudit,
    graph: MaterialGraphSnapshot,
    live_nodes: tuple[Any, ...],
    obj: Any,
) -> MaterialCapabilityAudit:
    """Promote an audit when the live source mesh cannot satisfy graph UV sampling."""

    findings = build_source_uv_findings(graph, live_nodes, obj)
    return (
        extend_material_capability_audit(audit, findings)
        if findings
        else audit
    )


__all__ = [
    "apply_source_uv_boundary",
    "build_source_uv_findings",
    "graph_uses_texture_coordinate_uv",
    "input_socket",
    "source_render_uv_name",
    "source_uv_layers",
]
