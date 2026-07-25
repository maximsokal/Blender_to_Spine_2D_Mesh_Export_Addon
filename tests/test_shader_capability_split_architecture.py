from __future__ import annotations

import ast
from pathlib import Path

import pytest

from Blender_to_Spine2D_Mesh_Exporter.blender_adapter import (
    shader_capability_analysis,
    shader_capability_audit,
    shader_capability_findings,
    shader_capability_node_findings,
    shader_capability_policy,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import (
    MaterialDependencyKind,
    MaterialGraphSnapshot,
    MaterialSemanticChannel,
    ShaderBakeCapability,
    ShaderCapabilityFinding,
    ShaderLinkSnapshot,
    ShaderNodeSnapshot,
)


ADAPTER = (
    Path(__file__).resolve().parents[1]
    / "Blender_to_Spine2D_Mesh_Exporter"
    / "blender_adapter"
)


def _source(name: str) -> str:
    return (ADAPTER / name).read_text(encoding="utf-8")


def _top_level_definitions(name: str):
    tree = ast.parse(_source(name), filename=name)
    return tuple(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    )


def _graph(
    node_type: str,
    *,
    output_socket: str = "Value",
    dependencies=(),
    channels=(MaterialSemanticChannel.SURFACE_COLOR,),
) -> MaterialGraphSnapshot:
    source = ShaderNodeSnapshot(
        node_id="Source",
        node_type=node_type,
        node_name="Source",
    )
    output = ShaderNodeSnapshot(
        node_id="Material Output",
        node_type="OUTPUT_MATERIAL",
        node_name="Material Output",
    )
    return MaterialGraphSnapshot(
        material_name="Material",
        active_output_node_id=output.node_id,
        reachable_nodes=(source, output),
        reachable_links=(
            ShaderLinkSnapshot(
                from_node_id=source.node_id,
                from_socket=output_socket,
                to_node_id=output.node_id,
                to_socket="Surface",
            ),
        ),
        semantic_channels=tuple(channels),
        dependencies=tuple(dependencies),
    )


def _codes(audit) -> tuple[str, ...]:
    return tuple(finding.code for finding in audit.findings)


def test_shader_capability_audit_is_compatibility_only():
    assert _top_level_definitions("shader_capability_audit.py") == ()
    source = _source("shader_capability_audit.py")
    for owner in (
        "shader_capability_analysis",
        "shader_capability_findings",
        "shader_capability_node_findings",
        "shader_capability_policy",
    ):
        assert owner in source


def test_policy_owner_contains_only_immutable_tables_and_target_normalization():
    source = _source("shader_capability_policy.py")
    assert "MappingProxyType" in source
    assert "LOCAL_SAFE_NODE_TYPES" in source
    assert "TEXTURE_COORD_CAPABILITIES" in source
    assert "GEOMETRY_OUTPUT_CAPABILITIES" in source
    assert "def normalise_render_target" in source
    for forbidden in (
        "MaterialGraphSnapshot",
        "MaterialCapabilityAudit",
        "ShaderCapabilityFinding",
        "ShaderNodeSnapshot",
        "strongest_shader_capability",
    ):
        assert forbidden not in source


def test_generic_finding_owner_does_not_apply_node_or_graph_policy():
    source = _source("shader_capability_findings.py")
    assert "def used_outputs" in source
    assert "def build_finding" in source
    assert "def order_unique_findings" in source
    for forbidden in (
        "LOCAL_SAFE_NODE_TYPES",
        "CAMERA_NODE_TYPES",
        "MaterialCapabilityAudit",
        "strongest_shader_capability",
        "GRAPH_CAMERA_DEPENDENCY",
        "TEXTURE_COORD_SOURCE_CONTEXT",
    ):
        assert forbidden not in source


def test_node_finding_owner_does_not_assemble_final_audit():
    source = _source("shader_capability_node_findings.py")
    assert "def texture_coordinate_findings" in source
    assert "def geometry_findings" in source
    assert "def node_findings" in source
    assert "TEXTURE_COORD_CAPABILITIES" in source
    for forbidden in (
        "MaterialCapabilityAudit",
        "strongest_shader_capability",
        "GRAPH_CAMERA_DEPENDENCY",
        "GRAPH_SCENE_DEPENDENCY",
        "VOLUME_RENDER_REQUIRED",
        "DISPLACEMENT_RENDER_REQUIRED",
    ):
        assert forbidden not in source


def test_analysis_owner_coordinates_graph_node_and_ordering_layers():
    source = _source("shader_capability_analysis.py")
    assert "used_outputs(graph)" in source
    assert "node_findings(" in source
    assert "order_unique_findings(findings)" in source
    assert "strongest_shader_capability(" in source
    assert "GRAPH_CAMERA_DEPENDENCY" in source
    assert "VOLUME_RENDER_REQUIRED" in source
    for forbidden in (
        "LOCAL_SAFE_NODE_TYPES",
        "CAMERA_NODE_TYPES",
        "TEXTURE_COORD_CAPABILITIES",
        "GEOMETRY_OUTPUT_CAPABILITIES",
    ):
        assert forbidden not in source


def test_facade_retains_historical_private_aliases():
    assert (
        shader_capability_audit.audit_material_graph_capabilities
        is shader_capability_analysis.audit_material_graph_capabilities
    )
    assert shader_capability_audit._finding is shader_capability_findings.build_finding
    assert shader_capability_audit._used_outputs is shader_capability_findings.used_outputs
    assert (
        shader_capability_audit._texture_coordinate_findings
        is shader_capability_node_findings.texture_coordinate_findings
    )
    assert (
        shader_capability_audit._geometry_findings
        is shader_capability_node_findings.geometry_findings
    )
    assert shader_capability_audit._node_findings is shader_capability_node_findings.node_findings
    assert (
        shader_capability_audit._normalise_render_target
        is shader_capability_policy.normalise_render_target
    )


def test_policy_mappings_are_read_only_and_target_normalization_is_compatible():
    assert shader_capability_policy.normalise_render_target("cycles") == "CYCLES"
    assert (
        shader_capability_policy.normalise_render_target("BLENDER_EEVEE")
        == "EEVEE"
    )
    for unsupported in ("BLENDER_EEVEE_NEXT", "MY_CYCLES", "WORKBENCH"):
        with pytest.raises(ValueError, match="Unsupported render_target"):
            shader_capability_policy.normalise_render_target(unsupported)
    with pytest.raises(TypeError):
        shader_capability_policy.TEXTURE_COORD_CAPABILITIES["future"] = (
            ShaderBakeCapability.UNSUPPORTED
        )


def test_finding_deduplication_and_order_preserve_historical_key():
    first = ShaderCapabilityFinding(
        capability=ShaderBakeCapability.CAMERA_RENDER_REQUIRED,
        code="SAME",
        reason="first reason",
        output_socket="Z",
    )
    duplicate = ShaderCapabilityFinding(
        capability=ShaderBakeCapability.CAMERA_RENDER_REQUIRED,
        code="SAME",
        reason="second reason",
        output_socket="Z",
    )
    local = ShaderCapabilityFinding(
        capability=ShaderBakeCapability.LOCAL_UV_SAFE,
        code="AAA",
        reason="local",
    )

    ordered = shader_capability_findings.order_unique_findings(
        (first, duplicate, local)
    )

    assert ordered == (first, local)
    assert ordered[0].reason == "first reason"


def test_graph_level_precedence_and_socket_policy_remain_unchanged():
    camera = shader_capability_analysis.audit_material_graph_capabilities(
        _graph(
            "TEX_IMAGE",
            dependencies=(
                MaterialDependencyKind.CAMERA,
                MaterialDependencyKind.LIGHTING,
            ),
        ),
        render_target="CYCLES",
    )
    texture_coord = shader_capability_analysis.audit_material_graph_capabilities(
        _graph("TEX_COORD", output_socket="From Instancer"),
        render_target="CYCLES",
    )

    assert camera.required_capability is ShaderBakeCapability.CAMERA_RENDER_REQUIRED
    assert "GRAPH_CAMERA_DEPENDENCY" in _codes(camera)
    assert "GRAPH_SCENE_DEPENDENCY" not in _codes(camera)
    assert texture_coord.required_capability is ShaderBakeCapability.GROUP_RENDER_REQUIRED
    assert "TEXTURE_COORD_INSTANCER_CONTEXT" in _codes(texture_coord)
