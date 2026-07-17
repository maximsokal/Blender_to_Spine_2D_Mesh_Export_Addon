"""Audit reachable Blender shader graphs without changing production routing.

The audit is deliberately diagnostic. It identifies graphs that the current temporary
UV-bake target cannot reproduce safely and records the minimum future execution boundary.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from ..domain.baking.capabilities import (
    MaterialCapabilityAudit,
    ShaderBakeCapability,
    ShaderCapabilityFinding,
    strongest_shader_capability,
)
from ..domain.baking.graph import (
    MaterialDependencyKind,
    MaterialGraphSnapshot,
    MaterialSemanticChannel,
    ShaderNodeSnapshot,
)


_RENDER_TARGETS = frozenset({"ALL", "CYCLES", "EEVEE"})
_CAMERA_DEPENDENCIES = frozenset(
    {
        MaterialDependencyKind.CAMERA,
        MaterialDependencyKind.VIEW,
        MaterialDependencyKind.REFLECTION,
        MaterialDependencyKind.TRANSMISSION,
    }
)
_SCENE_DEPENDENCIES = frozenset(
    {
        MaterialDependencyKind.WORLD,
        MaterialDependencyKind.LIGHTING,
        MaterialDependencyKind.OCCLUSION,
        MaterialDependencyKind.SCENE_OBJECTS,
    }
)

# Nodes whose reachable values are stable on the reconstructed UV-bake target when no
# stronger graph dependency is present. The list is intentionally explicit: an unknown
# node must be audited before production routing may claim it is safe.
_LOCAL_SAFE_NODE_TYPES = frozenset(
    {
        "ADD_SHADER",
        "BLACKBODY",
        "BRIGHTCONTRAST",
        "BSDF_DIFFUSE",
        "BSDF_PRINCIPLED",
        "BSDF_TRANSPARENT",
        "CHECKER",
        "CLAMP",
        "COMBHSV",
        "COMBRGB",
        "COMBXYZ",
        "COMBINE_COLOR",
        "CURVE_RGB",
        "CURVE_VEC",
        "DISPLACEMENT",
        "EMISSION",
        "GAMMA",
        "GROUP",
        "GROUP_INPUT",
        "GROUP_OUTPUT",
        "HUE_SAT",
        "INVERT",
        "MAP_RANGE",
        "MAPPING",
        "MATH",
        "MIX",
        "MIX_RGB",
        "MIX_SHADER",
        "NORMAL",
        "OUTPUT_MATERIAL",
        "PRINCIPLED_VOLUME",
        "RGB",
        "RGBTOBW",
        "REROUTE",
        "SEPARATE_COLOR",
        "SEPHSV",
        "SEPRGB",
        "SEPXYZ",
        "TEX_BRICK",
        "TEX_CHECKER",
        "TEX_GABOR",
        "TEX_GRADIENT",
        "TEX_IMAGE",
        "TEX_IES",
        "TEX_MAGIC",
        "TEX_MUSGRAVE",
        "TEX_NOISE",
        "TEX_SKY",
        "TEX_VORONOI",
        "TEX_WAVE",
        "TEX_WHITE_NOISE",
        "UVMAP",
        "VALUE",
        "VALTORGB",
        "VECT_MATH",
        "VECTOR_DISPLACEMENT",
        "VOLUME_INFO",
        "WAVELENGTH",
    }
)
_SCENE_NODE_TYPES = frozenset(
    {
        "AMBIENT_OCCLUSION",
        "BSDF_HAIR",
        "BSDF_HAIR_PRINCIPLED",
        "BSDF_TOON",
        "BSDF_TRANSLUCENT",
        "SUBSURFACE_SCATTERING",
    }
)
_CAMERA_NODE_TYPES = frozenset(
    {
        "BEVEL",
        "BSDF_GLASS",
        "BSDF_GLOSSY",
        "BSDF_REFRACTION",
        "CAMERA",
        "FRESNEL",
        "HOLDOUT",
        "LAYER_WEIGHT",
        "LIGHT_PATH",
        "OBJECT_INFO",
        "TEX_ENVIRONMENT",
    }
)
_GROUP_NODE_TYPES = frozenset(
    {
        "CURVES_INFO",
        "HAIR_INFO",
        "PARTICLE_INFO",
        "TEX_POINTDENSITY",
    }
)
_SOURCE_ATTRIBUTE_NODE_TYPES = frozenset(
    {
        "ATTRIBUTE",
        "VERTEX_COLOR",
    }
)

_TEXTURE_COORD_CAPABILITIES = {
    "uv": ShaderBakeCapability.LOCAL_UV_SAFE,
    "camera": ShaderBakeCapability.CAMERA_RENDER_REQUIRED,
    "window": ShaderBakeCapability.CAMERA_RENDER_REQUIRED,
    "reflection": ShaderBakeCapability.CAMERA_RENDER_REQUIRED,
    "object": ShaderBakeCapability.CAMERA_RENDER_REQUIRED,
    "generated": ShaderBakeCapability.CAMERA_RENDER_REQUIRED,
    "normal": ShaderBakeCapability.CAMERA_RENDER_REQUIRED,
    "from instancer": ShaderBakeCapability.GROUP_RENDER_REQUIRED,
}
_GEOMETRY_OUTPUT_CAPABILITIES = {
    "incoming": ShaderBakeCapability.CAMERA_RENDER_REQUIRED,
    "backfacing": ShaderBakeCapability.CAMERA_RENDER_REQUIRED,
    "pointiness": ShaderBakeCapability.CAMERA_RENDER_REQUIRED,
    "random per island": ShaderBakeCapability.CAMERA_RENDER_REQUIRED,
}


def _normalise_render_target(value: str) -> str:
    target = str(value or "ALL").strip().upper()
    if target in _RENDER_TARGETS:
        return target
    if "CYCLE" in target:
        return "CYCLES"
    if "EEVEE" in target:
        return "EEVEE"
    raise ValueError(f"Unsupported render_target: {value!r}")


def _used_outputs(graph: MaterialGraphSnapshot) -> dict[str, tuple[str, ...]]:
    values: dict[str, set[str]] = defaultdict(set)
    for link in graph.reachable_links:
        values[link.from_node_id].add(link.from_socket)
    return {
        node_id: tuple(sorted(sockets, key=str.casefold))
        for node_id, sockets in values.items()
    }


def _finding(
    capability: ShaderBakeCapability,
    code: str,
    reason: str,
    *,
    node: ShaderNodeSnapshot | None = None,
    output_socket: str | None = None,
) -> ShaderCapabilityFinding:
    return ShaderCapabilityFinding(
        capability=capability,
        code=code,
        reason=reason,
        node_id=None if node is None else node.node_id,
        node_type=None if node is None else node.node_type,
        output_socket=output_socket,
    )


def _texture_coordinate_findings(
    node: ShaderNodeSnapshot,
    outputs: Iterable[str],
) -> tuple[ShaderCapabilityFinding, ...]:
    resolved_outputs = tuple(outputs)
    if not resolved_outputs:
        return (
            _finding(
                ShaderBakeCapability.UNSUPPORTED,
                "TEXTURE_COORD_OUTPUT_UNKNOWN",
                "Texture Coordinate is reachable but its used output socket is unknown",
                node=node,
            ),
        )
    findings = []
    for output in resolved_outputs:
        capability = _TEXTURE_COORD_CAPABILITIES.get(output.strip().casefold())
        if capability is None:
            capability = ShaderBakeCapability.UNSUPPORTED
            code = "TEXTURE_COORD_OUTPUT_UNCLASSIFIED"
            reason = f"Texture Coordinate output '{output}' has no audited bake policy"
        elif capability is ShaderBakeCapability.LOCAL_UV_SAFE:
            code = "TEXTURE_COORD_UV_LOCAL"
            reason = "Texture Coordinate UV can be evaluated on the reconstructed UV target"
        elif capability is ShaderBakeCapability.GROUP_RENDER_REQUIRED:
            code = "TEXTURE_COORD_INSTANCER_CONTEXT"
            reason = "Texture Coordinate From Instancer requires the original instance context"
        else:
            code = "TEXTURE_COORD_SOURCE_CONTEXT"
            reason = (
                f"Texture Coordinate output '{output}' requires the original object or camera "
                "evaluation context"
            )
        findings.append(
            _finding(capability, code, reason, node=node, output_socket=output)
        )
    return tuple(findings)


def _geometry_findings(
    node: ShaderNodeSnapshot,
    outputs: Iterable[str],
) -> tuple[ShaderCapabilityFinding, ...]:
    findings = []
    for output in tuple(outputs):
        capability = _GEOMETRY_OUTPUT_CAPABILITIES.get(output.strip().casefold())
        if capability is None:
            continue
        findings.append(
            _finding(
                capability,
                "GEOMETRY_SOURCE_CONTEXT",
                f"Geometry output '{output}' is not stable on reconstructed bake topology",
                node=node,
                output_socket=output,
            )
        )
    return tuple(findings)


def _node_findings(
    node: ShaderNodeSnapshot,
    outputs: tuple[str, ...],
    *,
    render_target: str,
) -> tuple[ShaderCapabilityFinding, ...]:
    node_type = node.node_type
    if node_type == "TEX_COORD":
        return _texture_coordinate_findings(node, outputs)
    if node_type == "NEW_GEOMETRY":
        findings = _geometry_findings(node, outputs)
        return findings or (
            _finding(
                ShaderBakeCapability.LOCAL_UV_SAFE,
                "GEOMETRY_LOCAL_OUTPUT",
                "Reachable Geometry outputs use local reconstructed mesh data",
                node=node,
            ),
        )
    if node_type == "SHADER_TO_RGB":
        if render_target == "EEVEE":
            return (
                _finding(
                    ShaderBakeCapability.CAMERA_RENDER_REQUIRED,
                    "EEVEE_SHADER_TO_RGB",
                    "Shader to RGB requires an Eevee render and cannot use Cycles object bake",
                    node=node,
                ),
            )
        return (
            _finding(
                ShaderBakeCapability.UNSUPPORTED,
                "SHADER_TO_RGB_RENDERER_MISMATCH",
                "Shader to RGB is Eevee-only but the audited render target is not Eevee",
                node=node,
            ),
        )
    if node_type == "SCRIPT":
        return (
            _finding(
                ShaderBakeCapability.UNSUPPORTED,
                "OSL_PREFLIGHT_REQUIRED",
                "OSL Script requires engine, device, source, and compilation preflight",
                node=node,
            ),
        )
    if node_type in _SOURCE_ATTRIBUTE_NODE_TYPES:
        return (
            _finding(
                ShaderBakeCapability.CAMERA_RENDER_REQUIRED,
                "SOURCE_ATTRIBUTE_NOT_MATERIALIZED",
                "The reconstructed bake mesh does not preserve generic or color attributes",
                node=node,
            ),
        )
    if node_type in _GROUP_NODE_TYPES:
        return (
            _finding(
                ShaderBakeCapability.GROUP_RENDER_REQUIRED,
                "INSTANCE_OR_STRAND_CONTEXT",
                f"{node_type} requires particle, strand, curve, or instancer context",
                node=node,
            ),
        )
    if node_type in _CAMERA_NODE_TYPES:
        return (
            _finding(
                ShaderBakeCapability.CAMERA_RENDER_REQUIRED,
                "SOURCE_OR_CAMERA_CONTEXT",
                f"{node_type} requires original source-object or camera-ray evaluation",
                node=node,
            ),
        )
    if node_type in _SCENE_NODE_TYPES:
        return (
            _finding(
                ShaderBakeCapability.SCENE_UV_SAFE,
                "SCENE_EVALUATION_REQUIRED",
                f"{node_type} requires scene-aware UV baking",
                node=node,
            ),
        )
    if node_type in _LOCAL_SAFE_NODE_TYPES:
        return ()
    return (
        _finding(
            ShaderBakeCapability.UNSUPPORTED,
            "UNCLASSIFIED_REACHABLE_NODE",
            f"Reachable Blender node type '{node_type}' has no audited bake capability",
            node=node,
        ),
    )


def audit_material_graph_capabilities(
    graph: MaterialGraphSnapshot,
    *,
    render_target: str,
) -> MaterialCapabilityAudit:
    """Return a deterministic capability report without changing strategy selection."""

    if not isinstance(graph, MaterialGraphSnapshot):
        raise TypeError("graph must be MaterialGraphSnapshot")
    target = _normalise_render_target(render_target)
    findings: list[ShaderCapabilityFinding] = []

    if graph.active_output_node_id is None:
        findings.append(
            _finding(
                ShaderBakeCapability.UNSUPPORTED,
                "MATERIAL_OUTPUT_MISSING",
                "Renderer-specific Material Output could not be resolved",
            )
        )
    for issue in graph.issues:
        findings.append(
            _finding(
                ShaderBakeCapability.UNSUPPORTED,
                "GRAPH_ANALYSIS_INCOMPLETE",
                issue,
            )
        )

    used_outputs = _used_outputs(graph)
    for node in graph.reachable_nodes:
        findings.extend(
            _node_findings(
                node,
                used_outputs.get(node.node_id, ()),
                render_target=target,
            )
        )

    dependencies = set(graph.dependencies)
    if dependencies & _CAMERA_DEPENDENCIES:
        findings.append(
            _finding(
                ShaderBakeCapability.CAMERA_RENDER_REQUIRED,
                "GRAPH_CAMERA_DEPENDENCY",
                "Graph dependencies require active-camera render evaluation",
            )
        )
    elif dependencies & _SCENE_DEPENDENCIES:
        findings.append(
            _finding(
                ShaderBakeCapability.SCENE_UV_SAFE,
                "GRAPH_SCENE_DEPENDENCY",
                "Graph dependencies require scene-aware object baking",
            )
        )

    if MaterialSemanticChannel.VOLUME in graph.semantic_channels:
        findings.append(
            _finding(
                ShaderBakeCapability.CAMERA_RENDER_REQUIRED,
                "VOLUME_RENDER_REQUIRED",
                "Volume output cannot be represented by Blender object UV bake",
            )
        )
    if MaterialSemanticChannel.DISPLACEMENT in graph.semantic_channels:
        findings.append(
            _finding(
                ShaderBakeCapability.CAMERA_RENDER_REQUIRED,
                "DISPLACEMENT_RENDER_REQUIRED",
                "Render-evaluated displacement requires camera projection",
            )
        )

    if not findings:
        findings.append(
            _finding(
                ShaderBakeCapability.LOCAL_UV_SAFE,
                "LOCAL_GRAPH",
                "Reachable graph uses only audited local UV-bake-safe nodes",
            )
        )

    unique: dict[
        tuple[str, str, str | None, str | None, str | None],
        ShaderCapabilityFinding,
    ] = {}
    for finding in findings:
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
        material_name=graph.material_name,
        render_target=target,
        required_capability=strongest_shader_capability(
            finding.capability for finding in ordered
        ),
        findings=ordered,
    )
