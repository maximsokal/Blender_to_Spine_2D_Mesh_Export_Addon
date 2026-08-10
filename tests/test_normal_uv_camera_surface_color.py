from __future__ import annotations

from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.scene_material_preparation import (
    _material_preparation_pass,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import (
    BakeCompositeMode,
    BakeCompositePlan,
    BakeEvaluationScope,
    BakeMode,
    BakePassPlan,
    BakeStrategyId,
    MaterialAnalysis,
    MaterialKind,
    MaterialPreparationMode,
    MaterialSemanticChannel,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.baking.normal_uv_camera_context import (
    NormalUvCoverageAlphaBakeStrategy,
    _normalize_camera_texture_composite,
)


def _pass(
    *,
    strategy: BakeStrategyId,
    mode: BakeMode,
    scope: BakeEvaluationScope,
    channel: MaterialSemanticChannel = MaterialSemanticChannel.SURFACE_COLOR,
    pass_index: int = 0,
) -> BakePassPlan:
    return BakePassPlan(
        pass_index=pass_index,
        strategy_id=strategy,
        bake_mode=mode,
        material_slot_indices=(0,),
        semantic_channels=(channel,),
        evaluation_scope=scope,
    )


def _material(
    slot_index: int,
    *channels: MaterialSemanticChannel,
) -> MaterialAnalysis:
    # Synthetic material analysis intentionally omits graph data. The strategy contract
    # depends only on semantic channels and slot identity; Blender graph extraction is
    # covered by the material-preparation regression suite.
    node_types = ["BSDF_PRINCIPLED"]
    if MaterialSemanticChannel.ALPHA in channels:
        node_types.append("BSDF_TRANSPARENT")
    return MaterialAnalysis(
        slot_index=slot_index,
        material_name=f"Material_{slot_index}",
        kind=MaterialKind.SOLID_COLOR,
        node_types=tuple(node_types),
    )


def test_camera_surface_color_uses_existing_proxy_without_changing_scope():
    original = _pass(
        strategy=BakeStrategyId.CAMERA_SURFACE_COLOR,
        mode=BakeMode.EMIT,
        scope=BakeEvaluationScope.CAMERA,
    )

    prepared = _material_preparation_pass(original)

    assert prepared is not original
    assert original.strategy_id is BakeStrategyId.CAMERA_SURFACE_COLOR
    assert prepared.strategy_id is BakeStrategyId.SURFACE_COLOR
    assert prepared.bake_mode is BakeMode.EMIT
    assert prepared.evaluation_scope is BakeEvaluationScope.CAMERA
    assert prepared.material_slot_indices == original.material_slot_indices
    assert prepared.semantic_channels == original.semantic_channels


def test_camera_emission_remains_original_emit_material_in_camera_scope():
    original = _pass(
        strategy=BakeStrategyId.CAMERA_EMISSION,
        mode=BakeMode.EMIT,
        scope=BakeEvaluationScope.CAMERA,
        channel=MaterialSemanticChannel.SURFACE_EMISSION,
    )

    assert _material_preparation_pass(original) is original


def test_true_camera_combined_pass_is_not_retagged_as_surface_color():
    original = _pass(
        strategy=BakeStrategyId.CAMERA_COMBINED,
        mode=BakeMode.COMBINED,
        scope=BakeEvaluationScope.CAMERA,
    )

    assert _material_preparation_pass(original) is original


def test_camera_coverage_alpha_distinguishes_opaque_and_real_alpha_slots():
    strategy = NormalUvCoverageAlphaBakeStrategy()
    opaque = _material(0, MaterialSemanticChannel.SURFACE_COLOR)
    transparent = _material(
        1,
        MaterialSemanticChannel.SURFACE_COLOR,
        MaterialSemanticChannel.ALPHA,
    )
    usable = (opaque, transparent)

    assert strategy.supports(opaque)
    assert strategy.supports(transparent)
    assert strategy.semantic_channels(usable) == (MaterialSemanticChannel.ALPHA,)

    preparations = strategy.material_preparations(usable, usable)

    assert tuple(item.slot_index for item in preparations) == (0, 1)
    assert preparations[0].mode is MaterialPreparationMode.OPAQUE_ALPHA_TO_EMISSION
    assert preparations[1].mode is MaterialPreparationMode.EXTRACT_ALPHA_TO_EMISSION


def test_camera_texture_data_color_is_not_unpremultiplied_as_render_appearance():
    surface = _pass(
        strategy=BakeStrategyId.CAMERA_SURFACE_COLOR,
        mode=BakeMode.EMIT,
        scope=BakeEvaluationScope.CAMERA,
        pass_index=0,
    )
    emission = _pass(
        strategy=BakeStrategyId.CAMERA_EMISSION,
        mode=BakeMode.EMIT,
        scope=BakeEvaluationScope.CAMERA,
        channel=MaterialSemanticChannel.SURFACE_EMISSION,
        pass_index=1,
    )
    alpha = _pass(
        strategy=BakeStrategyId.ALPHA,
        mode=BakeMode.EMIT,
        scope=BakeEvaluationScope.AUXILIARY,
        channel=MaterialSemanticChannel.ALPHA,
        pass_index=2,
    )
    composite = BakeCompositePlan(
        mode=BakeCompositeMode.ADD_RGB_REPLACE_ALPHA,
        clamp_rgb=True,
        color_pass_indices=(0, 1),
        alpha_pass_index=2,
        unpremultiply_color_by_alpha=True,
    )

    normalized = _normalize_camera_texture_composite(
        (surface, emission, alpha),
        composite,
    )

    assert normalized.unpremultiply_color_by_alpha is False


def test_true_camera_combined_color_keeps_render_appearance_unpremultiplication():
    color = _pass(
        strategy=BakeStrategyId.CAMERA_COMBINED,
        mode=BakeMode.COMBINED,
        scope=BakeEvaluationScope.CAMERA,
        pass_index=0,
    )
    alpha = _pass(
        strategy=BakeStrategyId.ALPHA,
        mode=BakeMode.EMIT,
        scope=BakeEvaluationScope.AUXILIARY,
        channel=MaterialSemanticChannel.ALPHA,
        pass_index=1,
    )
    composite = BakeCompositePlan(
        mode=BakeCompositeMode.ADD_RGB_REPLACE_ALPHA,
        clamp_rgb=True,
        color_pass_indices=(0,),
        alpha_pass_index=1,
        unpremultiply_color_by_alpha=True,
    )

    normalized = _normalize_camera_texture_composite((color, alpha), composite)

    assert normalized is composite
    assert normalized.unpremultiply_color_by_alpha is True
