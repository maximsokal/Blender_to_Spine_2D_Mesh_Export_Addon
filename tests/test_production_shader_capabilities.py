from pathlib import Path
from types import SimpleNamespace

import pytest

from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.production_shader_capabilities import (
    _with_proxy_boundary,
    _with_source_uv_boundary,
    build_capability_checked_texture_plan,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.render_engine_contract import (
    render_engine_contract,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import (
    BakePlan,
    BakePlanError,
    BakeSettings,
    CameraBakeSnapshot,
    CameraProjectionPlan,
    ColorManagementSnapshot,
    MaterialAnalysis,
    MaterialCapabilityAudit,
    MaterialGraphSnapshot,
    MaterialKind,
    MaterialSemanticChannel,
    ObjectBakeContext,
    ObjectMaterialAnalysis,
    SceneBakeContext,
    ShaderBakeCapability,
    ShaderCapabilityFinding,
    ShaderLinkSnapshot,
    ShaderNodeSnapshot,
)


IDENTITY = (
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
)


class FakeUvLayers(list):
    def __init__(self, values=(), active=None):
        super().__init__(values)
        self.active = active


class FakeInputs(list):
    def get(self, name):
        for item in self:
            if item.name == name:
                return item
        return None


def _graph(*, alpha=False, node_type="BSDF_PRINCIPLED", muted=False, links=()):
    node = ShaderNodeSnapshot(
        node_id="Shader",
        node_type=node_type,
        node_name="Shader",
        muted=muted,
    )
    output = ShaderNodeSnapshot(
        node_id="Material Output",
        node_type="OUTPUT_MATERIAL",
        node_name="Material Output",
    )
    channels = [MaterialSemanticChannel.SURFACE_COLOR]
    if alpha:
        channels.append(MaterialSemanticChannel.ALPHA)
    return MaterialGraphSnapshot(
        material_name="Material",
        active_output_node_id=output.node_id,
        reachable_nodes=(node, output),
        reachable_links=tuple(links),
        semantic_channels=tuple(channels),
        dependencies=(),
    )


def _analysis(graph):
    return ObjectMaterialAnalysis(
        "Object",
        (
            MaterialAnalysis(
                slot_index=0,
                material_name="Material",
                kind=MaterialKind.SOLID_COLOR,
                graph=graph,
            ),
        ),
    )


def _audit(capability):
    return MaterialCapabilityAudit(
        material_name="Material",
        render_target="CYCLES",
        required_capability=capability,
        findings=(
            ShaderCapabilityFinding(
                capability=capability,
                code=f"TEST_{capability.value}",
                reason="test capability",
            ),
        ),
    )


def _settings(tmp_path: Path):
    return BakeSettings(
        width=32,
        height=32,
        output_directory=tmp_path,
        output_stem="Capability",
    )


def _object_context():
    return ObjectBakeContext(
        source_object_id="Object",
        object_type="MESH",
        world_matrix=IDENTITY,
    )


def _scene_context():
    return SceneBakeContext(
        scene_name="Scene",
        render_engine="CYCLES",
        analysis_frame=1,
        world=None,
        camera=CameraBakeSnapshot(
            object_id="Camera",
            camera_type="PERSP",
            world_matrix=IDENTITY,
            lens=50.0,
            ortho_scale=6.0,
            clip_start=0.1,
            clip_end=100.0,
        ),
        lights=(),
        visible_object_ids=("Camera", "Object"),
        shadow_caster_ids=("Object",),
        color_management=ColorManagementSnapshot(
            view_transform="Standard",
            look="",
            exposure=0.0,
            gamma=1.0,
        ),
    )


def _fake_object(render_uv=True):
    if render_uv:
        layer = SimpleNamespace(name="SourceUV", active_render=True)
        uv_layers = FakeUvLayers((layer,), active=layer)
    else:
        uv_layers = FakeUvLayers()
    return SimpleNamespace(data=SimpleNamespace(uv_layers=uv_layers))


def _fake_live_image_node(vector_linked=False):
    vector = SimpleNamespace(name="Vector", is_linked=vector_linked)
    node = SimpleNamespace(inputs=FakeInputs((vector,)))
    output = SimpleNamespace(inputs=FakeInputs())
    return node, output


def test_local_capability_keeps_object_bake(tmp_path: Path):
    graph = _graph()
    plan = build_capability_checked_texture_plan(
        _analysis(graph),
        _settings(tmp_path),
        (_audit(ShaderBakeCapability.LOCAL_UV_SAFE),),
        render_engine_contract("CYCLES"),
        object_context=_object_context(),
        scene_context=_scene_context(),
    )

    assert isinstance(plan, BakePlan)
    assert not isinstance(plan, CameraProjectionPlan)


def test_camera_capability_routes_whole_object_to_b4(tmp_path: Path):
    graph = _graph()
    plan = build_capability_checked_texture_plan(
        _analysis(graph),
        _settings(tmp_path),
        (_audit(ShaderBakeCapability.CAMERA_RENDER_REQUIRED),),
        render_engine_contract("CYCLES"),
        object_context=_object_context(),
        scene_context=_scene_context(),
    )

    assert isinstance(plan, CameraProjectionPlan)


@pytest.mark.parametrize(
    "capability",
    (
        ShaderBakeCapability.GROUP_RENDER_REQUIRED,
        ShaderBakeCapability.UNSUPPORTED,
    ),
)
def test_unrepresentable_capabilities_fail_explicitly(tmp_path: Path, capability):
    graph = _graph()

    with pytest.raises(BakePlanError, match=capability.value):
        build_capability_checked_texture_plan(
            _analysis(graph),
            _settings(tmp_path),
            (_audit(capability),),
            render_engine_contract("CYCLES"),
            object_context=_object_context(),
            scene_context=_scene_context(),
        )


def test_alpha_group_is_promoted_to_camera_render():
    graph = _graph(alpha=True, node_type="GROUP")
    base = _audit(ShaderBakeCapability.LOCAL_UV_SAFE)

    resolved = _with_proxy_boundary(base, graph)

    assert resolved.required_capability is ShaderBakeCapability.CAMERA_RENDER_REQUIRED
    assert "ALPHA_PROXY_RECURSIVE_BOUNDARY" in {
        finding.code for finding in resolved.findings
    }


def test_alpha_muted_bypass_is_promoted_to_camera_render():
    graph = _graph(alpha=True, muted=True)
    base = _audit(ShaderBakeCapability.LOCAL_UV_SAFE)

    resolved = _with_proxy_boundary(base, graph)

    assert resolved.required_capability is ShaderBakeCapability.CAMERA_RENDER_REQUIRED
    assert "ALPHA_PROXY_MUTED_BYPASS" in {
        finding.code for finding in resolved.findings
    }


def test_unlinked_image_without_source_render_uv_is_promoted_to_b4():
    graph = _graph(node_type="TEX_IMAGE")
    base = _audit(ShaderBakeCapability.LOCAL_UV_SAFE)
    live_nodes = _fake_live_image_node(vector_linked=False)

    resolved = _with_source_uv_boundary(
        base,
        graph,
        live_nodes,
        _fake_object(render_uv=False),
    )

    assert resolved.required_capability is ShaderBakeCapability.CAMERA_RENDER_REQUIRED
    assert "SOURCE_RENDER_UV_MISSING" in {
        finding.code for finding in resolved.findings
    }


def test_unlinked_image_with_source_render_uv_remains_local_safe():
    graph = _graph(node_type="TEX_IMAGE")
    base = _audit(ShaderBakeCapability.LOCAL_UV_SAFE)
    live_nodes = _fake_live_image_node(vector_linked=False)

    resolved = _with_source_uv_boundary(
        base,
        graph,
        live_nodes,
        _fake_object(render_uv=True),
    )

    assert resolved.required_capability is ShaderBakeCapability.LOCAL_UV_SAFE


def test_texture_coordinate_uv_without_source_uv_is_promoted_to_b4():
    link = ShaderLinkSnapshot(
        from_node_id="Shader",
        from_socket="UV",
        to_node_id="Material Output",
        to_socket="Surface",
    )
    graph = _graph(node_type="TEX_COORD", links=(link,))
    base = _audit(ShaderBakeCapability.LOCAL_UV_SAFE)
    live_nodes = (SimpleNamespace(inputs=FakeInputs()), SimpleNamespace(inputs=FakeInputs()))

    resolved = _with_source_uv_boundary(
        base,
        graph,
        live_nodes,
        _fake_object(render_uv=False),
    )

    assert resolved.required_capability is ShaderBakeCapability.CAMERA_RENDER_REQUIRED
    assert "SOURCE_RENDER_UV_MISSING" in {
        finding.code for finding in resolved.findings
    }
