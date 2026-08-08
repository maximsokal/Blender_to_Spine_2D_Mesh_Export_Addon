"""Live Principled BSDF capability details for production bake routing.

The immutable graph records that a material depends on camera/view/reflection context,
but a Principled BSDF can reach that aggregate dependency through materially different
features. Metallic and Coat are reproducible by the Normal/UV Cycles COMBINED object-bake
route, while Transmission remains a render-ray boundary and must stay blocked there.
"""

from __future__ import annotations

from typing import Any

from ..domain.baking.capabilities import (
    MaterialCapabilityAudit,
    ShaderBakeCapability,
    ShaderCapabilityFinding,
)
from ..domain.baking.graph import MaterialGraphSnapshot
from .production_shader_capability_merge import extend_material_capability_audit
from .production_shader_capability_runtime import validate_live_node_alignment
from .shader_capability_findings import build_finding
from .shader_graph_semantics import (
    PrincipledCameraFeature,
    principled_camera_features,
)


PRINCIPLED_REFLECTION_CONTEXT_CODE = "PRINCIPLED_REFLECTION_CONTEXT"
PRINCIPLED_TRANSMISSION_RENDER_REQUIRED_CODE = (
    "PRINCIPLED_TRANSMISSION_RENDER_REQUIRED"
)
PRINCIPLED_REFLECTION_CONTEXT_OUTPUTS = frozenset({"Metallic", "Coat Weight"})

_FEATURE_OUTPUT_NAME = {
    PrincipledCameraFeature.METALLIC: "Metallic",
    PrincipledCameraFeature.COAT: "Coat Weight",
    PrincipledCameraFeature.TRANSMISSION: "Transmission Weight",
}


def build_principled_context_findings(
    graph: MaterialGraphSnapshot,
    live_nodes: tuple[Any, ...],
) -> tuple[ShaderCapabilityFinding, ...]:
    """Build concrete camera findings for reachable live Principled BSDF nodes.

    ``GRAPH_CAMERA_DEPENDENCY`` is intentionally an aggregate finding and cannot tell
    Normal/UV routing whether the dependency came from reflective surface appearance or
    from transmission/refraction. This function preserves that distinction while live
    Blender RNA is still aligned with the immutable graph snapshot.
    """

    if not isinstance(graph, MaterialGraphSnapshot):
        raise TypeError("graph must be MaterialGraphSnapshot")
    if not isinstance(live_nodes, tuple):
        raise TypeError("live_nodes must be tuple")

    validate_live_node_alignment(graph, live_nodes)

    findings: list[ShaderCapabilityFinding] = []
    for snapshot, live_node in zip(graph.reachable_nodes, live_nodes, strict=True):
        if snapshot.muted or snapshot.node_type != "BSDF_PRINCIPLED":
            continue

        features = principled_camera_features(live_node)
        for feature in sorted(features, key=lambda value: value.value):
            output_name = _FEATURE_OUTPUT_NAME[feature]
            if feature is PrincipledCameraFeature.TRANSMISSION:
                findings.append(
                    build_finding(
                        ShaderBakeCapability.CAMERA_RENDER_REQUIRED,
                        PRINCIPLED_TRANSMISSION_RENDER_REQUIRED_CODE,
                        (
                            "Principled Transmission requires camera-ray/refraction "
                            "evaluation and cannot be represented faithfully by the "
                            "Normal/UV surface object-bake route"
                        ),
                        node=snapshot,
                        output_socket=output_name,
                    )
                )
                continue

            findings.append(
                build_finding(
                    ShaderBakeCapability.CAMERA_RENDER_REQUIRED,
                    PRINCIPLED_REFLECTION_CONTEXT_CODE,
                    (
                        f"Principled {output_name} uses camera/view reflection context "
                        "that Cycles COMBINED object bake can evaluate on the original "
                        "source surface"
                    ),
                    node=snapshot,
                    output_socket=output_name,
                )
            )

    return tuple(findings)


def apply_principled_context_boundary(
    audit: MaterialCapabilityAudit,
    graph: MaterialGraphSnapshot,
    live_nodes: tuple[Any, ...],
) -> MaterialCapabilityAudit:
    """Enrich one material audit with concrete Principled camera-context causes."""

    if not isinstance(audit, MaterialCapabilityAudit):
        raise TypeError("audit must be MaterialCapabilityAudit")
    findings = build_principled_context_findings(graph, live_nodes)
    return (
        extend_material_capability_audit(audit, findings)
        if findings
        else audit
    )


__all__ = [
    "PRINCIPLED_REFLECTION_CONTEXT_CODE",
    "PRINCIPLED_REFLECTION_CONTEXT_OUTPUTS",
    "PRINCIPLED_TRANSMISSION_RENDER_REQUIRED_CODE",
    "apply_principled_context_boundary",
    "build_principled_context_findings",
]
