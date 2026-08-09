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
    MaterialSemanticChannel,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.baking.normal_uv_camera_context import (
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


def test_camera_context_emit_uses_existing_surface_color_proxy_without_changing_scope():
    original = _pass(
        strategy=BakeStrategyId.CAMERA_COMBINED,
        mode=BakeMode.EMIT,
        scope=BakeEvaluationScope.CAMERA,
    )

    prepared = _material_preparation_pass(original)

    assert prepared is not original
    assert original.strategy_id is BakeStrategyId.CAMERA_COMBINED
    assert prepared.strategy_id is BakeStrategyId.SURFACE_COLOR
    assert prepared.bake_mode is BakeMode.EMIT
    assert prepared.evaluation_scope is BakeEvaluationScope.CAMERA
    assert prepared.material_slot_indices == original.material_slot_indices
    assert prepared.semantic_channels == original.semantic_channels


def test_true_camera_combined_pass_is_not_retagged_as_surface_color():
    original = _pass(
        strategy=BakeStrategyId.CAMERA_COMBINED,
        mode=BakeMode.COMBINED,
        scope=BakeEvaluationScope.CAMERA,
    )

    assert _material_preparation_pass(original) is original


def test_camera_context_emit_color_is_not_unpremultiplied_as_render_appearance():
    color = _pass(
        strategy=BakeStrategyId.CAMERA_COMBINED,
        mode=BakeMode.EMIT,
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
