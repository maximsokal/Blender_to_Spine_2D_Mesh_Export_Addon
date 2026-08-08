"""Regression coverage for concrete Principled camera-context routing."""

from __future__ import annotations

from dataclasses import dataclass

from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.production_shader_capability_principled import (
    PRINCIPLED_REFLECTION_CONTEXT_CODE,
    PRINCIPLED_TRANSMISSION_RENDER_REQUIRED_CODE,
    apply_principled_context_boundary,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.production_shader_capability_routing import (
    _normal_uv_blocking_camera_findings,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import (
    MaterialCapabilityAudit,
    MaterialDependencyKind,
    MaterialGraphSnapshot,
    MaterialSemanticChannel,
    ShaderBakeCapability,
    ShaderCapabilityFinding,
    ShaderNodeSnapshot,
)


@dataclass
class _Socket:
    default_value: float = 0.0
    is_linked: bool = False


class _Inputs(dict[str, _Socket]):
    pass


@dataclass
class _LiveNode:
    name: str
    type: str
    inputs: _Inputs


def _graph() -> MaterialGraphSnapshot:
    return MaterialGraphSnapshot(
        material_name="BottleMaterial",
        active_output_node_id="Material Output",
        reachable_nodes=(
            ShaderNodeSnapshot(
                node_id="Material Output",
                node_type="OUTPUT_MATERIAL",
                node_name="Material Output",
            ),
            ShaderNodeSnapshot(
                node_id="Principled BSDF",
                node_type="BSDF_PRINCIPLED",
                node_name="Principled BSDF",
            ),
        ),
        reachable_links=(),
        semantic_channels=(MaterialSemanticChannel.SURFACE_COLOR,),
        dependencies=(
            MaterialDependencyKind.CAMERA,
            MaterialDependencyKind.VIEW,
            MaterialDependencyKind.REFLECTION,
        ),
        issues=(),
    )


def _live_nodes(
    *,
    metallic: float = 0.0,
    coat: float = 0.0,
    transmission: float = 0.0,
    coat_linked: bool = False,
    transmission_linked: bool = False,
) -> tuple[_LiveNode, ...]:
    return (
        _LiveNode(
            name="Material Output",
            type="OUTPUT_MATERIAL",
            inputs=_Inputs(),
        ),
        _LiveNode(
            name="Principled BSDF",
            type="BSDF_PRINCIPLED",
            inputs=_Inputs(
                {
                    "Metallic": _Socket(default_value=metallic),
                    "Coat Weight": _Socket(
                        default_value=coat,
                        is_linked=coat_linked,
                    ),
                    "Transmission Weight": _Socket(
                        default_value=transmission,
                        is_linked=transmission_linked,
                    ),
                }
            ),
        ),
    )


def _aggregate_audit() -> MaterialCapabilityAudit:
    return MaterialCapabilityAudit(
        material_name="BottleMaterial",
        render_target="CYCLES",
        required_capability=ShaderBakeCapability.CAMERA_RENDER_REQUIRED,
        findings=(
            ShaderCapabilityFinding(
                capability=ShaderBakeCapability.CAMERA_RENDER_REQUIRED,
                code="GRAPH_CAMERA_DEPENDENCY",
                reason="Graph dependencies require active-camera render evaluation",
            ),
        ),
    )


def _enrich(**live_kwargs: object) -> MaterialCapabilityAudit:
    return apply_principled_context_boundary(
        _aggregate_audit(),
        _graph(),
        _live_nodes(**live_kwargs),
    )


def test_principled_metallic_explains_aggregate_and_allows_normal_uv() -> None:
    audit = _enrich(metallic=0.8)

    reflection = tuple(
        finding
        for finding in audit.findings
        if finding.code == PRINCIPLED_REFLECTION_CONTEXT_CODE
    )
    assert len(reflection) == 1
    assert reflection[0].node_type == "BSDF_PRINCIPLED"
    assert reflection[0].output_socket == "Metallic"
    assert _normal_uv_blocking_camera_findings((audit,)) == ()


def test_linked_principled_coat_is_camera_combined_safe_for_normal_uv() -> None:
    audit = _enrich(coat_linked=True)

    reflection = tuple(
        finding
        for finding in audit.findings
        if finding.code == PRINCIPLED_REFLECTION_CONTEXT_CODE
    )
    assert len(reflection) == 1
    assert reflection[0].output_socket == "Coat Weight"
    assert _normal_uv_blocking_camera_findings((audit,)) == ()


def test_principled_transmission_remains_a_normal_uv_blocker() -> None:
    audit = _enrich(transmission=0.35)

    blockers = _normal_uv_blocking_camera_findings((audit,))
    assert blockers == (
        (
            "BottleMaterial",
            (
                (
                    PRINCIPLED_TRANSMISSION_RENDER_REQUIRED_CODE,
                    "BSDF_PRINCIPLED",
                    "Transmission Weight",
                ),
            ),
        ),
    )


def test_reflection_does_not_hide_principled_transmission_blocker() -> None:
    audit = _enrich(metallic=1.0, transmission_linked=True)

    assert any(
        finding.code == PRINCIPLED_REFLECTION_CONTEXT_CODE
        for finding in audit.findings
    )
    blockers = _normal_uv_blocking_camera_findings((audit,))
    assert len(blockers) == 1
    assert tuple(item[0] for item in blockers[0][1]) == (
        PRINCIPLED_TRANSMISSION_RENDER_REQUIRED_CODE,
    )


def test_unknown_aggregate_only_camera_dependency_stays_fail_closed() -> None:
    audit = _enrich()

    assert audit == _aggregate_audit()
    assert _normal_uv_blocking_camera_findings((audit,)) == (
        (
            "BottleMaterial",
            (("GRAPH_CAMERA_DEPENDENCY", None, None),),
        ),
    )
