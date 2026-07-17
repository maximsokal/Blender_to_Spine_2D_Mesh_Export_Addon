from pathlib import Path

import pytest

from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.production_shader_capabilities import (
    _with_proxy_boundary,
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


def _graph(*, alpha=False, node_type="BSDF_PRINCIPLED", muted=False):
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
        reachable_links=(),
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
