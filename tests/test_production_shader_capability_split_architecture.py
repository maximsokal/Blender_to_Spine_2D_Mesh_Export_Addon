import ast
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from Blender_to_Spine2D_Mesh_Exporter.blender_adapter import (
    production_shader_capabilities as facade,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.production_shader_capability_error import (
    ProductionShaderCapabilityError,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.production_shader_capability_merge import (
    extend_material_capability_audit,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.production_shader_capability_proxy import (
    apply_alpha_proxy_boundary,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.production_shader_capability_runtime import (
    enrich_graph_with_live_mute,
    validate_graph_snapshot_parity,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.production_shader_capability_uv import (
    build_source_uv_findings,
    source_render_uv_name,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import (
    MaterialCapabilityAudit,
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


def _definitions(name: str):
    tree = ast.parse(_source(name), filename=name)
    return tuple(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    )


def _graph(*, nodes=None, links=(), channels=None, dependencies=(), issues=()):
    resolved_nodes = nodes or (
        ShaderNodeSnapshot(
            node_id="First Group::Image",
            node_type="TEX_IMAGE",
            node_name="Image",
            group_path=("First Group",),
        ),
        ShaderNodeSnapshot(
            node_id="Second Group::Image",
            node_type="TEX_IMAGE",
            node_name="Image",
            group_path=("Second Group",),
        ),
        ShaderNodeSnapshot(
            node_id="Material Output",
            node_type="OUTPUT_MATERIAL",
            node_name="Material Output",
        ),
    )
    return MaterialGraphSnapshot(
        material_name="Material",
        active_output_node_id="Material Output",
        reachable_nodes=tuple(resolved_nodes),
        reachable_links=tuple(links),
        semantic_channels=(
            (MaterialSemanticChannel.SURFACE_COLOR,)
            if channels is None
            else tuple(channels)
        ),
        dependencies=tuple(dependencies),
        issues=tuple(issues),
    )


def _audit():
    finding = ShaderCapabilityFinding(
        capability=ShaderBakeCapability.LOCAL_UV_SAFE,
        code="LOCAL_GRAPH",
        reason="local",
    )
    return MaterialCapabilityAudit(
        material_name="Material",
        render_target="CYCLES",
        required_capability=ShaderBakeCapability.LOCAL_UV_SAFE,
        findings=(finding,),
    )


class _UvLayers(list):
    def __init__(self, values=(), *, active=None):
        super().__init__(values)
        self.active = active


def test_public_module_is_compatibility_only():
    assert _definitions("production_shader_capabilities.py") == ()
    source = _source("production_shader_capabilities.py")
    for owner in (
        "production_shader_capability_error",
        "production_shader_capability_merge",
        "production_shader_capability_object_audit",
        "production_shader_capability_proxy",
        "production_shader_capability_routing",
        "production_shader_capability_runtime",
        "production_shader_capability_uv",
    ):
        assert owner in source


def test_physical_owners_do_not_cross_boundaries():
    runtime = _source("production_shader_capability_runtime.py")
    proxy = _source("production_shader_capability_proxy.py")
    uv = _source("production_shader_capability_uv.py")
    object_audit = _source("production_shader_capability_object_audit.py")
    routing = _source("production_shader_capability_routing.py")

    assert "build_texture_plan" not in runtime
    assert "build_camera_projection_plan" not in runtime
    assert "uv_layers" not in proxy
    assert "build_texture_plan" not in uv
    assert "build_camera_projection_plan" not in object_audit
    assert "analyse_material_graph_detailed" not in routing
    assert "live_nodes" not in routing


def test_texture_planning_uses_physical_owners():
    source = _source("a1_texture_planning.py")
    assert "from .production_shader_capability_object_audit import" in source
    assert "from .production_shader_capability_routing import" in source
    assert "from .production_shader_capabilities import" not in source


def test_recursive_equal_names_cannot_swap_group_instances():
    expected = _graph()
    actual = _graph(
        nodes=(
            expected.reachable_nodes[1],
            expected.reachable_nodes[0],
            expected.reachable_nodes[2],
        )
    )

    with pytest.raises(ProductionShaderCapabilityError, match="identity changed"):
        validate_graph_snapshot_parity(expected, actual)


def test_snapshot_parity_rejects_type_links_channels_dependencies_and_issues():
    expected = _graph()
    variants = (
        replace(
            expected,
            reachable_nodes=(
                replace(expected.reachable_nodes[0], node_type="VALUE"),
                *expected.reachable_nodes[1:],
            ),
        ),
        replace(
            expected,
            reachable_links=(
                ShaderLinkSnapshot(
                    from_node_id="First Group::Image",
                    from_socket="Color",
                    to_node_id="Material Output",
                    to_socket="Surface",
                ),
            ),
        ),
        replace(expected, semantic_channels=(MaterialSemanticChannel.ALPHA,)),
        replace(expected, dependencies=(MaterialDependencyKind.CAMERA,)),
        replace(expected, issues=("changed",)),
    )
    for actual in variants:
        with pytest.raises(ProductionShaderCapabilityError):
            validate_graph_snapshot_parity(expected, actual)


def test_live_count_fails_before_enrichment_and_uv_inspection():
    graph = _graph()
    live_nodes = tuple(
        SimpleNamespace(name=node.node_name, type=node.node_type, mute=False)
        for node in graph.reachable_nodes[:-1]
    )
    obj = SimpleNamespace(data=SimpleNamespace(uv_layers=_UvLayers()))

    with pytest.raises(ProductionShaderCapabilityError, match="node count"):
        enrich_graph_with_live_mute(graph, live_nodes)
    with pytest.raises(ProductionShaderCapabilityError, match="node count"):
        build_source_uv_findings(graph, live_nodes, obj)


def test_named_uv_findings_keep_existing_codes():
    output = ShaderNodeSnapshot(
        node_id="Material Output",
        node_type="OUTPUT_MATERIAL",
        node_name="Material Output",
    )
    cases = (
        (
            ShaderNodeSnapshot("Normal", "NORMAL_MAP", "Normal"),
            SimpleNamespace(name="Normal", type="NORMAL_MAP", uv_map="Missing"),
            "NAMED_NORMAL_UV_MISSING",
        ),
        (
            ShaderNodeSnapshot("Tangent", "TANGENT", "Tangent"),
            SimpleNamespace(
                name="Tangent",
                type="TANGENT",
                direction_type="UV_MAP",
                uv_map="Missing",
            ),
            "NAMED_TANGENT_UV_MISSING",
        ),
        (
            ShaderNodeSnapshot("UV Map", "UVMAP", "UV Map"),
            SimpleNamespace(name="UV Map", type="UVMAP", uv_map="Missing"),
            "NAMED_UV_MISSING",
        ),
    )
    obj = SimpleNamespace(data=SimpleNamespace(uv_layers=_UvLayers()))
    for snapshot, live_node, code in cases:
        findings = build_source_uv_findings(
            _graph(nodes=(snapshot, output)),
            (
                live_node,
                SimpleNamespace(
                    name="Material Output",
                    type="OUTPUT_MATERIAL",
                ),
            ),
            obj,
        )
        assert code in {finding.code for finding in findings}


def test_multiple_active_render_layers_fail_explicitly():
    layers = _UvLayers(
        (
            SimpleNamespace(name="First", active_render=True),
            SimpleNamespace(name="Second", active_render=True),
        )
    )
    obj = SimpleNamespace(data=SimpleNamespace(uv_layers=layers))

    with pytest.raises(ProductionShaderCapabilityError, match="more than one"):
        source_render_uv_name(obj)


def test_shared_finding_order_keeps_first_reason():
    first = ShaderCapabilityFinding(
        capability=ShaderBakeCapability.CAMERA_RENDER_REQUIRED,
        code="DUPLICATE",
        reason="first",
    )
    result = extend_material_capability_audit(
        _audit(),
        (first, replace(first, reason="second")),
    )
    matches = tuple(
        finding for finding in result.findings if finding.code == "DUPLICATE"
    )

    assert len(matches) == 1
    assert matches[0].reason == "first"
    assert result.required_capability is ShaderBakeCapability.CAMERA_RENDER_REQUIRED


def test_alpha_proxy_and_private_aliases_remain_compatible():
    graph = _graph(
        nodes=(
            ShaderNodeSnapshot("Group", "GROUP", "Group", muted=True),
            ShaderNodeSnapshot(
                "Material Output",
                "OUTPUT_MATERIAL",
                "Material Output",
            ),
        ),
        channels=(MaterialSemanticChannel.ALPHA,),
    )
    result = apply_alpha_proxy_boundary(_audit(), graph)

    assert {
        "ALPHA_PROXY_RECURSIVE_BOUNDARY",
        "ALPHA_PROXY_MUTED_BYPASS",
    }.issubset({finding.code for finding in result.findings})
    assert facade._rebuild_audit is extend_material_capability_audit
    assert facade._with_proxy_boundary is apply_alpha_proxy_boundary
    assert facade._source_render_uv_name is source_render_uv_name
    assert facade._enriched_graph_with_live_mute is enrich_graph_with_live_mute
