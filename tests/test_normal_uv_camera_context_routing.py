"""Regression contracts for camera/source-context materials in Normal UV mode."""

from __future__ import annotations

from pathlib import Path

import pytest

from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.production_shader_capability_routing import (
    build_capability_checked_texture_plan,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.render_engine_contract import (
    RenderEngineContract,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import (
    A1TextureExportMode,
    BakeCompositeMode,
    BakeEvaluationScope,
    BakeMode,
    BakePlan,
    BakePlanError,
    BakeSettings,
    BakeStrategyId,
    CameraBakeSnapshot,
    CameraProjectionPlan,
    ColorManagementSnapshot,
    MaterialAnalysis,
    MaterialCapabilityAudit,
    MaterialDependencyKind,
    MaterialGraphSnapshot,
    MaterialKind,
    MaterialPreparationMode,
    MaterialSemanticChannel,
    ObjectBakeContext,
    ObjectMaterialAnalysis,
    SceneBakeContext,
    ShaderBakeCapability,
    ShaderCapabilityFinding,
    ShaderNodeSnapshot,
)


_IDENTITY = (
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


def _analysis(
    *,
    channels: tuple[MaterialSemanticChannel, ...] = (
        MaterialSemanticChannel.SURFACE_COLOR,
    ),
) -> ObjectMaterialAnalysis:
    graph = MaterialGraphSnapshot(
        material_name="Crystal",
        active_output_node_id="output",
        reachable_nodes=(
            ShaderNodeSnapshot(
                node_id="output",
                node_type="OUTPUT_MATERIAL",
                node_name="Material Output",
            ),
            ShaderNodeSnapshot(
                node_id="fresnel",
                node_type="FRESNEL",
                node_name="Fresnel",
            ),
            ShaderNodeSnapshot(
                node_id="coordinates",
                node_type="TEX_COORD",
                node_name="Texture Coordinate",
            ),
        ),
        reachable_links=(),
        semantic_channels=channels,
        dependencies=(
            MaterialDependencyKind.CAMERA,
            MaterialDependencyKind.VIEW,
            MaterialDependencyKind.REFLECTION,
            MaterialDependencyKind.GEOMETRY,
        ),
    )
    return ObjectMaterialAnalysis(
        source_object_id="CrystalObject",
        slots=(
            MaterialAnalysis(
                slot_index=0,
                material_name="Crystal",
                kind=MaterialKind.PROCEDURAL,
                node_types=("OUTPUT_MATERIAL", "FRESNEL", "TEX_COORD"),
                graph=graph,
            ),
        ),
    )


def _object_context() -> ObjectBakeContext:
    return ObjectBakeContext(
        source_object_id="CrystalObject",
        object_type="MESH",
        world_matrix=_IDENTITY,
    )


def _scene_context(*, with_camera: bool = True) -> SceneBakeContext:
    camera = (
        CameraBakeSnapshot(
            object_id="Camera",
            camera_type="PERSP",
            world_matrix=_IDENTITY,
            lens=50.0,
            ortho_scale=6.0,
            clip_start=0.1,
            clip_end=1000.0,
        )
        if with_camera
        else None
    )
    return SceneBakeContext(
        scene_name="Scene",
        render_engine="CYCLES",
        analysis_frame=1,
        world=None,
        camera=camera,
        lights=(),
        visible_object_ids=("CrystalObject",),
        shadow_caster_ids=(),
        color_management=ColorManagementSnapshot(
            view_transform="Standard",
            look="",
            exposure=0.0,
            gamma=1.0,
        ),
    )


def _settings() -> BakeSettings:
    return BakeSettings(
        width=128,
        height=128,
        output_directory=Path("output"),
        output_stem="Crystal",
        uv_layer_name="SpineBakeUV",
        sequence_start_frame=1,
        sequence_frame_count=3,
    )


def _finding(
    code: str,
    *,
    node_type: str | None = None,
    output_socket: str | None = None,
) -> ShaderCapabilityFinding:
    return ShaderCapabilityFinding(
        capability=ShaderBakeCapability.CAMERA_RENDER_REQUIRED,
        code=code,
        reason=f"test finding: {code}",
        node_type=node_type,
        output_socket=output_socket,
    )


def _camera_audit(
    *extra_findings: ShaderCapabilityFinding,
) -> tuple[MaterialCapabilityAudit, ...]:
    findings = (
        _finding("GRAPH_CAMERA_DEPENDENCY"),
        _finding("SOURCE_OR_CAMERA_CONTEXT", node_type="FRESNEL"),
        _finding(
            "TEXTURE_COORD_SOURCE_CONTEXT",
            node_type="TEX_COORD",
            output_socket="Camera",
        ),
        _finding(
            "TEXTURE_COORD_SOURCE_CONTEXT",
            node_type="TEX_COORD",
            output_socket="Reflection",
        ),
        _finding(
            "TEXTURE_COORD_SOURCE_CONTEXT",
            node_type="TEX_COORD",
            output_socket="Reflection",
        ),
        *extra_findings,
    )
    return (
        MaterialCapabilityAudit(
            material_name="Crystal",
            render_target="CYCLES",
            required_capability=ShaderBakeCapability.CAMERA_RENDER_REQUIRED,
            findings=findings,
        ),
    )


def _build(
    *,
    audits: tuple[MaterialCapabilityAudit, ...] | None = None,
    mode: A1TextureExportMode = A1TextureExportMode.NORMAL_UV_SEGMENTS,
    analysis: ObjectMaterialAnalysis | None = None,
    scene_context: SceneBakeContext | None = None,
):
    return build_capability_checked_texture_plan(
        analysis or _analysis(),
        _settings(),
        audits or _camera_audit(),
        RenderEngineContract("CYCLES", "CYCLES"),
        object_context=_object_context(),
        scene_context=scene_context or _scene_context(),
        texture_export_mode=mode,
    )


def test_normal_uv_mode_keeps_camera_and_reflection_coordinates_on_object_bake() -> None:
    plan = _build()

    assert type(plan) is BakePlan
    assert not isinstance(plan, CameraProjectionPlan)
    assert len(plan.frame_tasks) == 3
    assert tuple(task.timeline_frame for task in plan.frame_tasks) == (1, 2, 3)

    assert len(plan.passes) == 2
    surface_pass, alpha_pass = plan.passes

    assert surface_pass.pass_index == 0
    assert surface_pass.strategy_id is BakeStrategyId.CAMERA_SURFACE_COLOR
    assert surface_pass.evaluation_scope is BakeEvaluationScope.CAMERA
    assert surface_pass.bake_mode is BakeMode.EMIT
    assert surface_pass.semantic_channels == (MaterialSemanticChannel.SURFACE_COLOR,)

    assert alpha_pass.pass_index == 1
    assert alpha_pass.strategy_id is BakeStrategyId.ALPHA
    assert alpha_pass.evaluation_scope is BakeEvaluationScope.AUXILIARY
    assert alpha_pass.bake_mode is BakeMode.EMIT
    assert alpha_pass.semantic_channels == (MaterialSemanticChannel.ALPHA,)
    assert tuple(
        preparation.mode for preparation in alpha_pass.material_preparations
    ) == (MaterialPreparationMode.OPAQUE_ALPHA_TO_EMISSION,)

    assert plan.composite.mode is BakeCompositeMode.ADD_RGB_REPLACE_ALPHA
    assert plan.composite.color_pass_indices == (0,)
    assert plan.composite.alpha_pass_index == 1
    assert plan.composite.unpremultiply_color_by_alpha is False
    assert plan.scene_aware is True


def test_explicit_camera_projection_mode_still_builds_projection_plan() -> None:
    plan = _build(mode=A1TextureExportMode.CAMERA_PROJECTION)

    assert isinstance(plan, CameraProjectionPlan)


@pytest.mark.parametrize(
    "blocking_finding",
    (
        _finding("DISPLACEMENT_RENDER_REQUIRED"),
        _finding("EEVEE_SHADER_TO_RGB", node_type="SHADER_TO_RGB"),
        _finding("SOURCE_ATTRIBUTE_NOT_MATERIALIZED", node_type="ATTRIBUTE"),
        _finding("VOLUME_RENDER_REQUIRED"),
        _finding(
            "TEXTURE_COORD_SOURCE_CONTEXT",
            node_type="TEX_COORD",
            output_socket="Window",
        ),
        _finding("SOURCE_OR_CAMERA_CONTEXT", node_type="OBJECT_INFO"),
        _finding("SOURCE_OR_CAMERA_CONTEXT", node_type="LIGHT_PATH"),
    ),
)
def test_normal_uv_mode_rejects_unrepresentable_camera_findings(
    blocking_finding: ShaderCapabilityFinding,
) -> None:
    with pytest.raises(BakePlanError, match="Select Export Mode: Camera Projection"):
        _build(audits=_camera_audit(blocking_finding))


def test_aggregate_camera_finding_without_concrete_evidence_fails_closed() -> None:
    audit = MaterialCapabilityAudit(
        material_name="Crystal",
        render_target="CYCLES",
        required_capability=ShaderBakeCapability.CAMERA_RENDER_REQUIRED,
        findings=(_finding("GRAPH_CAMERA_DEPENDENCY"),),
    )
    with pytest.raises(BakePlanError, match="Select Export Mode: Camera Projection"):
        _build(audits=(audit,))


def test_normal_uv_camera_context_requires_active_camera_snapshot() -> None:
    with pytest.raises(BakePlanError, match="active scene camera"):
        _build(scene_context=_scene_context(with_camera=False))


def test_normal_uv_builder_rejects_render_displacement_channel() -> None:
    analysis = _analysis(
        channels=(
            MaterialSemanticChannel.SURFACE_COLOR,
            MaterialSemanticChannel.DISPLACEMENT,
        )
    )
    with pytest.raises(BakePlanError, match="render displacement"):
        _build(analysis=analysis)
