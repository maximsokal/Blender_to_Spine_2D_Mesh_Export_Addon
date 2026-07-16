from array import array
from pathlib import Path

from Blender_to_Spine2D_Mesh_Exporter.blender_adapter import (
    BakePixelBuffer,
    compose_bake_passes,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import (
    BakeCompositeMode,
    BakeCompositePlan,
    BakeMode,
    BakeSettings,
    BakeStrategyId,
    MaterialAnalysis,
    MaterialGraphSnapshot,
    MaterialKind,
    MaterialPreparationMode,
    MaterialSemanticChannel,
    ObjectMaterialAnalysis,
    ShaderNodeSnapshot,
    build_bake_plan,
)


def _graph(*channels: MaterialSemanticChannel) -> MaterialGraphSnapshot:
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
        dependencies=(),
    )


def _slot(index: int, name: str, *channels: MaterialSemanticChannel) -> MaterialAnalysis:
    return MaterialAnalysis(
        slot_index=index,
        material_name=name,
        kind=MaterialKind.SOLID_COLOR,
        graph=_graph(*channels),
    )


def _settings(tmp_path: Path, *, procedural_mode: BakeMode = BakeMode.DIFFUSE):
    return BakeSettings(
        width=2,
        height=1,
        output_directory=tmp_path,
        output_stem="AlphaObject",
        diffuse_mode=BakeMode.DIFFUSE,
        procedural_mode=procedural_mode,
    )


def _buffer(values) -> BakePixelBuffer:
    return BakePixelBuffer(
        width=2,
        height=1,
        channels=4,
        pixels=array("f", values),
    )


def test_surface_alpha_adds_straight_color_and_alpha_passes(tmp_path: Path):
    plan = build_bake_plan(
        ObjectMaterialAnalysis(
            "Object",
            (
                _slot(
                    0,
                    "Cutout",
                    MaterialSemanticChannel.SURFACE_COLOR,
                    MaterialSemanticChannel.ALPHA,
                ),
            ),
        ),
        _settings(tmp_path),
    )

    assert tuple(item.strategy_id for item in plan.passes) == (
        BakeStrategyId.SURFACE_COLOR,
        BakeStrategyId.ALPHA,
    )
    assert plan.passes[0].bake_mode is BakeMode.EMIT
    assert plan.composite.mode is BakeCompositeMode.ADD_RGB_REPLACE_ALPHA
    assert plan.composite.color_pass_indices == (0,)
    assert plan.composite.alpha_pass_index == 1
    assert plan.requires_composition

    alpha_pass = plan.passes[1]
    assert tuple(
        (item.slot_index, item.mode) for item in alpha_pass.material_preparations
    ) == ((0, MaterialPreparationMode.EXTRACT_ALPHA_TO_EMISSION),)


def test_alpha_pass_marks_non_alpha_slots_opaque(tmp_path: Path):
    plan = build_bake_plan(
        ObjectMaterialAnalysis(
            "Object",
            (
                _slot(0, "Opaque", MaterialSemanticChannel.SURFACE_COLOR),
                _slot(
                    1,
                    "Cutout",
                    MaterialSemanticChannel.SURFACE_COLOR,
                    MaterialSemanticChannel.ALPHA,
                ),
            ),
        ),
        _settings(tmp_path),
    )

    assert plan.passes[0].bake_mode is BakeMode.EMIT
    alpha_pass = plan.passes[-1]
    assert tuple(
        (item.slot_index, item.mode) for item in alpha_pass.material_preparations
    ) == (
        (0, MaterialPreparationMode.OPAQUE_ALPHA_TO_EMISSION),
        (1, MaterialPreparationMode.EXTRACT_ALPHA_TO_EMISSION),
    )


def test_pure_transparent_material_uses_one_alpha_pass_but_still_composes(tmp_path: Path):
    plan = build_bake_plan(
        ObjectMaterialAnalysis(
            "Object",
            (_slot(0, "Transparent", MaterialSemanticChannel.ALPHA),),
        ),
        _settings(tmp_path),
    )

    assert len(plan.passes) == 1
    assert not plan.multipass
    assert plan.requires_composition
    assert plan.passes[0].strategy_id is BakeStrategyId.ALPHA
    assert plan.passes[0].bake_mode is BakeMode.EMIT
    assert plan.composite.mode is BakeCompositeMode.ADD_RGB_REPLACE_ALPHA
    assert plan.composite.color_pass_indices == ()
    assert plan.composite.alpha_pass_index == 0


def test_alpha_surface_uses_emission_proxy_even_when_combined_was_requested(tmp_path: Path):
    plan = build_bake_plan(
        ObjectMaterialAnalysis(
            "Object",
            (
                _slot(
                    0,
                    "Transparent Surface",
                    MaterialSemanticChannel.SURFACE_COLOR,
                    MaterialSemanticChannel.ALPHA,
                ),
            ),
        ),
        _settings(tmp_path, procedural_mode=BakeMode.COMBINED),
    )

    assert plan.passes[0].bake_mode is BakeMode.EMIT
    assert plan.passes[1].bake_mode is BakeMode.EMIT


def test_alpha_compositor_uses_red_mask_and_ignores_pass_alpha():
    color = _buffer(
        (
            0.8,
            0.2,
            0.1,
            1.0,
            0.1,
            0.6,
            0.3,
            1.0,
        )
    )
    alpha = _buffer(
        (
            0.25,
            0.25,
            0.25,
            1.0,
            0.75,
            0.75,
            0.75,
            0.0,
        )
    )
    result = compose_bake_passes(
        (color, alpha),
        BakeCompositePlan(
            mode=BakeCompositeMode.ADD_RGB_REPLACE_ALPHA,
            color_pass_indices=(0,),
            alpha_pass_index=1,
        ),
    )

    assert tuple(round(float(value), 4) for value in result.pixels) == (
        0.8,
        0.2,
        0.1,
        0.25,
        0.1,
        0.6,
        0.3,
        0.75,
    )


def test_alpha_only_compositor_outputs_black_rgb_with_mask_alpha():
    alpha = _buffer(
        (
            0.0,
            0.0,
            0.0,
            1.0,
            0.6,
            0.6,
            0.6,
            1.0,
        )
    )
    result = compose_bake_passes(
        (alpha,),
        BakeCompositePlan(
            mode=BakeCompositeMode.ADD_RGB_REPLACE_ALPHA,
            color_pass_indices=(),
            alpha_pass_index=0,
        ),
    )

    assert tuple(round(float(value), 4) for value in result.pixels) == (
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.6,
    )
