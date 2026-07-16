from array import array
from pathlib import Path

import pytest

from Blender_to_Spine2D_Mesh_Exporter.blender_adapter import (
    BakePixelBuffer,
    compose_bake_passes,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import (
    BakeCompositeMode,
    BakeCompositePlan,
    BakeEvaluationScope,
    BakeMode,
    BakePlanError,
    BakeSettings,
    BakeStrategyId,
    CameraBakeSnapshot,
    ColorManagementSnapshot,
    LightBakeSnapshot,
    MaterialAnalysis,
    MaterialDependencyKind,
    MaterialGraphSnapshot,
    MaterialKind,
    MaterialPreparationMode,
    MaterialSemanticChannel,
    ObjectBakeContext,
    ObjectMaterialAnalysis,
    SceneBakeContext,
    ShaderNodeSnapshot,
    WorldBakeSnapshot,
    build_bake_plan,
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


def _graph(*channels, dependencies=()):
    output = ShaderNodeSnapshot(
        node_id="Material Output",
        node_type="OUTPUT_MATERIAL",
        node_name="Material Output",
    )
    return MaterialGraphSnapshot(
        material_name="Material",
        active_output_node_id=output.node_id,
        reachable_nodes=(output,),
        reachable_links=(),
        semantic_channels=tuple(channels),
        dependencies=tuple(dependencies),
    )


def _slot(index, name, *channels, dependencies=()):
    return MaterialAnalysis(
        slot_index=index,
        material_name=name,
        kind=MaterialKind.SOLID_COLOR,
        graph=_graph(*channels, dependencies=dependencies),
    )


def _settings(tmp_path: Path):
    return BakeSettings(
        width=32,
        height=32,
        output_directory=tmp_path,
        output_stem="SceneObject",
        diffuse_mode=BakeMode.DIFFUSE,
        procedural_mode=BakeMode.DIFFUSE,
    )


def _object_context():
    return ObjectBakeContext(
        source_object_id="Object",
        object_type="MESH",
        world_matrix=IDENTITY,
        collection_names=("Collection",),
    )


def _scene_context(*, camera=True, lights=True):
    resolved_camera = (
        CameraBakeSnapshot(
            object_id="Camera",
            camera_type="PERSP",
            world_matrix=IDENTITY,
            lens=50.0,
            ortho_scale=6.0,
            clip_start=0.1,
            clip_end=1000.0,
        )
        if camera
        else None
    )
    resolved_lights = (
        (
            LightBakeSnapshot(
                object_id="Key",
                light_type="AREA",
                energy=1000.0,
                color=(1.0, 1.0, 1.0),
                world_matrix=IDENTITY,
            ),
        )
        if lights
        else ()
    )
    return SceneBakeContext(
        scene_name="Scene",
        render_engine="CYCLES",
        analysis_frame=1,
        world=WorldBakeSnapshot(
            world_name="World",
            color=(0.05, 0.05, 0.05),
            use_nodes=True,
            node_types=("BACKGROUND", "OUTPUT_WORLD"),
            background_strength=0.5,
        ),
        camera=resolved_camera,
        lights=resolved_lights,
        visible_object_ids=("Camera", "Key", "Object") if camera and lights else ("Object",),
        shadow_caster_ids=("Object",),
        color_management=ColorManagementSnapshot(
            view_transform="Standard",
            look="",
            exposure=0.0,
            gamma=1.0,
        ),
    )


def test_scene_context_reports_effective_resources():
    context = _scene_context()

    assert context.has_camera
    assert context.has_effective_lighting
    assert context.animated_dependency_ids == ()


def test_local_surface_keeps_legacy_local_strategy_without_context(tmp_path: Path):
    plan = build_bake_plan(
        ObjectMaterialAnalysis(
            "Object",
            (_slot(0, "Local", MaterialSemanticChannel.SURFACE_COLOR),),
        ),
        _settings(tmp_path),
    )

    assert tuple(item.strategy_id for item in plan.passes) == (
        BakeStrategyId.SURFACE_COLOR,
    )
    assert plan.passes[0].evaluation_scope is BakeEvaluationScope.LOCAL
    assert not plan.scene_aware


def test_scene_dependency_requires_immutable_scene_context(tmp_path: Path):
    analysis = ObjectMaterialAnalysis(
        "Object",
        (
            _slot(
                0,
                "Toon",
                MaterialSemanticChannel.SURFACE_COLOR,
                dependencies=(MaterialDependencyKind.LIGHTING,),
            ),
        ),
    )

    with pytest.raises(BakePlanError, match="SceneBakeContext"):
        build_bake_plan(analysis, _settings(tmp_path))


def test_scene_dependency_selects_combined_strategy(tmp_path: Path):
    analysis = ObjectMaterialAnalysis(
        "Object",
        (
            _slot(
                0,
                "Toon",
                MaterialSemanticChannel.SURFACE_COLOR,
                dependencies=(MaterialDependencyKind.LIGHTING,),
            ),
        ),
    )
    plan = build_bake_plan(
        analysis,
        _settings(tmp_path),
        object_context=_object_context(),
        scene_context=_scene_context(),
    )

    assert tuple(item.strategy_id for item in plan.passes) == (
        BakeStrategyId.SCENE_COMBINED,
    )
    assert plan.passes[0].bake_mode is BakeMode.COMBINED
    assert plan.passes[0].evaluation_scope is BakeEvaluationScope.SCENE
    assert plan.scene_aware


def test_camera_dependency_requires_active_camera(tmp_path: Path):
    analysis = ObjectMaterialAnalysis(
        "Object",
        (
            _slot(
                0,
                "Fresnel",
                MaterialSemanticChannel.SURFACE_COLOR,
                dependencies=(MaterialDependencyKind.VIEW,),
            ),
        ),
    )

    with pytest.raises(BakePlanError, match="active scene camera"):
        build_bake_plan(
            analysis,
            _settings(tmp_path),
            object_context=_object_context(),
            scene_context=_scene_context(camera=False, lights=False),
        )


def test_camera_dependency_requires_render_projection_even_with_camera(tmp_path: Path):
    analysis = ObjectMaterialAnalysis(
        "Object",
        (
            _slot(
                0,
                "Fresnel",
                MaterialSemanticChannel.SURFACE_COLOR,
                dependencies=(MaterialDependencyKind.VIEW,),
            ),
        ),
    )

    with pytest.raises(BakePlanError, match="camera-render projection"):
        build_bake_plan(
            analysis,
            _settings(tmp_path),
            object_context=_object_context(),
            scene_context=_scene_context(),
        )


def test_mixed_scene_and_local_slots_receive_explicit_masks(tmp_path: Path):
    analysis = ObjectMaterialAnalysis(
        "Object",
        (
            _slot(0, "Local", MaterialSemanticChannel.SURFACE_COLOR),
            _slot(
                1,
                "Scene",
                MaterialSemanticChannel.SURFACE_COLOR,
                dependencies=(MaterialDependencyKind.LIGHTING,),
            ),
        ),
    )
    plan = build_bake_plan(
        analysis,
        _settings(tmp_path),
        object_context=_object_context(),
        scene_context=_scene_context(),
    )

    assert tuple(item.strategy_id for item in plan.passes) == (
        BakeStrategyId.SCENE_COMBINED,
        BakeStrategyId.SURFACE_COLOR,
    )
    assert tuple(
        (item.slot_index, item.mode)
        for item in plan.passes[0].material_preparations
    ) == (
        (0, MaterialPreparationMode.ZERO_TO_EMISSION),
        (1, MaterialPreparationMode.PRESERVE),
    )
    assert tuple(
        (item.slot_index, item.mode)
        for item in plan.passes[1].material_preparations
    ) == (
        (0, MaterialPreparationMode.PRESERVE),
        (1, MaterialPreparationMode.ZERO_TO_EMISSION),
    )


def test_scene_alpha_plan_requests_straight_rgb_unpremultiplication(tmp_path: Path):
    analysis = ObjectMaterialAnalysis(
        "Object",
        (
            _slot(
                0,
                "Transparent Toon",
                MaterialSemanticChannel.SURFACE_COLOR,
                MaterialSemanticChannel.ALPHA,
                dependencies=(MaterialDependencyKind.LIGHTING,),
            ),
        ),
    )
    plan = build_bake_plan(
        analysis,
        _settings(tmp_path),
        object_context=_object_context(),
        scene_context=_scene_context(),
    )

    assert tuple(item.strategy_id for item in plan.passes) == (
        BakeStrategyId.SCENE_COMBINED,
        BakeStrategyId.ALPHA,
    )
    assert plan.composite.mode is BakeCompositeMode.ADD_RGB_REPLACE_ALPHA
    assert plan.composite.unpremultiply_color_by_alpha


def test_pure_transparent_remains_alpha_only(tmp_path: Path):
    plan = build_bake_plan(
        ObjectMaterialAnalysis(
            "Object",
            (_slot(0, "Transparent", MaterialSemanticChannel.ALPHA),),
        ),
        _settings(tmp_path),
    )

    assert tuple(item.strategy_id for item in plan.passes) == (BakeStrategyId.ALPHA,)
    assert plan.composite.mode is BakeCompositeMode.ADD_RGB_REPLACE_ALPHA


def test_scene_compositor_unpremultiplies_color_by_explicit_alpha():
    color = BakePixelBuffer(
        width=1,
        height=1,
        channels=4,
        pixels=array("f", (0.2, 0.1, 0.05, 1.0)),
    )
    alpha = BakePixelBuffer(
        width=1,
        height=1,
        channels=4,
        pixels=array("f", (0.25, 0.25, 0.25, 1.0)),
    )
    result = compose_bake_passes(
        (color, alpha),
        BakeCompositePlan(
            mode=BakeCompositeMode.ADD_RGB_REPLACE_ALPHA,
            color_pass_indices=(0,),
            alpha_pass_index=1,
            unpremultiply_color_by_alpha=True,
        ),
    )

    assert tuple(round(float(value), 4) for value in result.pixels) == (
        0.8,
        0.4,
        0.2,
        0.25,
    )
