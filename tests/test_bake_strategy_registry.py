from pathlib import Path

import pytest

from Blender_to_Spine2D_Mesh_Exporter.domain.baking import (
    BakeCompositeMode,
    BakeMode,
    BakePlanError,
    BakeSettings,
    BakeStrategyId,
    MaterialAnalysis,
    MaterialGraphSnapshot,
    MaterialKind,
    MaterialSemanticChannel,
    ObjectMaterialAnalysis,
    ShaderNodeSnapshot,
    build_bake_plan,
)


def graph(*channels):
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


def slot(index, name, *channels):
    return MaterialAnalysis(
        slot_index=index,
        material_name=name,
        kind=MaterialKind.SOLID_COLOR,
        graph=graph(*channels),
    )


def settings(tmp_path: Path):
    return BakeSettings(
        width=64,
        height=64,
        output_directory=tmp_path,
        output_stem="Object",
        diffuse_mode=BakeMode.DIFFUSE,
        procedural_mode=BakeMode.DIFFUSE,
    )


def test_surface_color_resolves_one_registered_pass(tmp_path: Path):
    plan = build_bake_plan(
        ObjectMaterialAnalysis(
            "Object",
            (slot(0, "Body", MaterialSemanticChannel.SURFACE_COLOR),),
        ),
        settings(tmp_path),
    )

    assert not plan.multipass
    assert plan.bake_mode is BakeMode.DIFFUSE
    assert plan.passes[0].strategy_id is BakeStrategyId.SURFACE_COLOR
    assert plan.passes[0].material_slot_indices == (0,)
    assert plan.composite.mode is BakeCompositeMode.SINGLE


def test_pure_emission_resolves_emit_pass(tmp_path: Path):
    plan = build_bake_plan(
        ObjectMaterialAnalysis(
            "Object",
            (slot(0, "Glow", MaterialSemanticChannel.SURFACE_EMISSION),),
        ),
        settings(tmp_path),
    )

    assert not plan.multipass
    assert plan.bake_mode is BakeMode.EMIT
    assert plan.passes[0].strategy_id is BakeStrategyId.EMISSION


def test_surface_and_emission_slots_resolve_deterministic_multipass(tmp_path: Path):
    plan = build_bake_plan(
        ObjectMaterialAnalysis(
            "Object",
            (
                slot(0, "Body", MaterialSemanticChannel.SURFACE_COLOR),
                slot(1, "Glow", MaterialSemanticChannel.SURFACE_EMISSION),
            ),
        ),
        settings(tmp_path),
    )

    assert plan.multipass
    assert tuple(item.strategy_id for item in plan.passes) == (
        BakeStrategyId.SURFACE_COLOR,
        BakeStrategyId.EMISSION,
    )
    assert tuple(item.bake_mode for item in plan.passes) == (
        BakeMode.DIFFUSE,
        BakeMode.EMIT,
    )
    assert plan.passes[0].material_slot_indices == (0,)
    assert plan.passes[1].material_slot_indices == (1,)
    assert plan.composite.mode is BakeCompositeMode.ADD_RGB_MAX_ALPHA


def test_one_material_with_surface_and_emission_uses_two_passes(tmp_path: Path):
    plan = build_bake_plan(
        ObjectMaterialAnalysis(
            "Object",
            (
                slot(
                    0,
                    "Lit Glow",
                    MaterialSemanticChannel.SURFACE_COLOR,
                    MaterialSemanticChannel.SURFACE_EMISSION,
                ),
            ),
        ),
        settings(tmp_path),
    )

    assert plan.multipass
    assert plan.passes[0].material_slot_indices == (0,)
    assert plan.passes[1].material_slot_indices == (0,)


def test_volume_requires_future_camera_projection_strategy(tmp_path: Path):
    with pytest.raises(BakePlanError, match="camera-projection"):
        build_bake_plan(
            ObjectMaterialAnalysis(
                "Object",
                (slot(0, "Smoke", MaterialSemanticChannel.VOLUME),),
            ),
            settings(tmp_path),
        )
