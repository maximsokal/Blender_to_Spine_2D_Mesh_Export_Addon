"""Socket-level and node-family findings for shader capability auditing."""

from __future__ import annotations

from typing import Iterable

from ..domain.baking.capabilities import (
    ShaderBakeCapability,
    ShaderCapabilityFinding,
)
from ..domain.baking.graph import ShaderNodeSnapshot
from .shader_capability_findings import build_finding
from .shader_capability_policy import (
    CAMERA_NODE_TYPES,
    GEOMETRY_OUTPUT_CAPABILITIES,
    GROUP_NODE_TYPES,
    LOCAL_SAFE_NODE_TYPES,
    SCENE_NODE_TYPES,
    SOURCE_ATTRIBUTE_NODE_TYPES,
    TEXTURE_COORD_CAPABILITIES,
)


_INSTANCER_AFFECTED_TEXTURE_COORD_OUTPUTS = frozenset({"generated", "uv"})


def texture_coordinate_findings(
    node: ShaderNodeSnapshot,
    outputs: Iterable[str],
) -> tuple[ShaderCapabilityFinding, ...]:
    """Classify only the Texture Coordinate outputs used by the reachable graph."""

    if not isinstance(node, ShaderNodeSnapshot):
        raise TypeError("node must be ShaderNodeSnapshot")

    resolved_outputs = tuple(outputs)
    if not resolved_outputs:
        return (
            build_finding(
                ShaderBakeCapability.UNSUPPORTED,
                "TEXTURE_COORD_OUTPUT_UNKNOWN",
                "Texture Coordinate is reachable but its used output socket is unknown",
                node=node,
            ),
        )

    findings = []
    for output in resolved_outputs:
        if not isinstance(output, str) or not output.strip():
            raise TypeError("Texture Coordinate outputs must contain non-empty strings")

        output_key = output.strip().casefold()
        if node.from_instancer and output_key in _INSTANCER_AFFECTED_TEXTURE_COORD_OUTPUTS:
            capability = ShaderBakeCapability.GROUP_RENDER_REQUIRED
            code = "TEXTURE_COORD_INSTANCER_CONTEXT"
            reason = (
                f"Texture Coordinate {output} uses From Instancer and therefore requires "
                "the original instance context"
            )
        else:
            capability = TEXTURE_COORD_CAPABILITIES.get(output_key)
            if capability is None:
                capability = ShaderBakeCapability.UNSUPPORTED
                code = "TEXTURE_COORD_OUTPUT_UNCLASSIFIED"
                reason = f"Texture Coordinate output '{output}' has no audited bake policy"
            elif capability is ShaderBakeCapability.LOCAL_UV_SAFE:
                code = "TEXTURE_COORD_UV_LOCAL"
                reason = "Texture Coordinate UV uses the preserved source render UV layer"
            else:
                code = "TEXTURE_COORD_SOURCE_CONTEXT"
                reason = (
                    f"Texture Coordinate output '{output}' requires the original object or "
                    "camera evaluation context"
                )

        findings.append(
            build_finding(
                capability,
                code,
                reason,
                node=node,
                output_socket=output,
            )
        )
    return tuple(findings)


def geometry_findings(
    node: ShaderNodeSnapshot,
    outputs: Iterable[str],
) -> tuple[ShaderCapabilityFinding, ...]:
    """Classify source-sensitive Geometry outputs while ignoring audited local outputs."""

    findings = []
    for output in tuple(outputs):
        capability = GEOMETRY_OUTPUT_CAPABILITIES.get(output.strip().casefold())
        if capability is None:
            continue
        findings.append(
            build_finding(
                capability,
                "GEOMETRY_SOURCE_CONTEXT",
                f"Geometry output '{output}' is not stable on reconstructed bake topology",
                node=node,
                output_socket=output,
            )
        )
    return tuple(findings)


def node_findings(
    node: ShaderNodeSnapshot,
    outputs: tuple[str, ...],
    *,
    render_target: str,
) -> tuple[ShaderCapabilityFinding, ...]:
    """Return the exact capability findings for one reachable node."""

    node_type = node.node_type
    if node_type == "TEX_COORD":
        return texture_coordinate_findings(node, outputs)
    if node_type == "NEW_GEOMETRY":
        findings = geometry_findings(node, outputs)
        return findings or (
            build_finding(
                ShaderBakeCapability.LOCAL_UV_SAFE,
                "GEOMETRY_LOCAL_OUTPUT",
                "Reachable Geometry outputs use local reconstructed mesh data",
                node=node,
            ),
        )
    if node_type == "SHADER_TO_RGB":
        if render_target == "EEVEE":
            return (
                build_finding(
                    ShaderBakeCapability.CAMERA_RENDER_REQUIRED,
                    "EEVEE_SHADER_TO_RGB",
                    "Shader to RGB requires an Eevee render and cannot use Cycles object bake",
                    node=node,
                ),
            )
        return (
            build_finding(
                ShaderBakeCapability.UNSUPPORTED,
                "SHADER_TO_RGB_RENDERER_MISMATCH",
                "Shader to RGB is Eevee-only but the audited render target is not Eevee",
                node=node,
            ),
        )
    if node_type == "SCRIPT":
        return (
            build_finding(
                ShaderBakeCapability.UNSUPPORTED,
                "OSL_PREFLIGHT_REQUIRED",
                "OSL Script requires engine, device, source, and compilation preflight",
                node=node,
            ),
        )
    if node_type in SOURCE_ATTRIBUTE_NODE_TYPES:
        return (
            build_finding(
                ShaderBakeCapability.CAMERA_RENDER_REQUIRED,
                "SOURCE_ATTRIBUTE_NOT_MATERIALIZED",
                "The reconstructed bake mesh does not preserve generic or color attributes",
                node=node,
            ),
        )
    if node_type in GROUP_NODE_TYPES:
        return (
            build_finding(
                ShaderBakeCapability.GROUP_RENDER_REQUIRED,
                "INSTANCE_OR_STRAND_CONTEXT",
                f"{node_type} requires particle, strand, curve, or instancer context",
                node=node,
            ),
        )
    if node_type in CAMERA_NODE_TYPES:
        return (
            build_finding(
                ShaderBakeCapability.CAMERA_RENDER_REQUIRED,
                "SOURCE_OR_CAMERA_CONTEXT",
                f"{node_type} requires original source-object or camera-ray evaluation",
                node=node,
            ),
        )
    if node_type in SCENE_NODE_TYPES:
        return (
            build_finding(
                ShaderBakeCapability.SCENE_UV_SAFE,
                "SCENE_EVALUATION_REQUIRED",
                f"{node_type} requires scene-aware UV baking",
                node=node,
            ),
        )
    if node_type in LOCAL_SAFE_NODE_TYPES:
        return ()
    return (
        build_finding(
            ShaderBakeCapability.UNSUPPORTED,
            "UNCLASSIFIED_REACHABLE_NODE",
            f"Reachable Blender node type '{node_type}' has no audited bake capability",
            node=node,
        ),
    )


__all__ = [
    "geometry_findings",
    "node_findings",
    "texture_coordinate_findings",
]
