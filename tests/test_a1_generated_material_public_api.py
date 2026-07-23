from pathlib import Path

from Blender_to_Spine2D_Mesh_Exporter.application import (
    build_generated_material_plan,
    generated_palette_color,
)
from Blender_to_Spine2D_Mesh_Exporter.application.a1_generated_materials import (
    build_generated_material_plan as private_build_generated_material_plan,
    generated_palette_color as private_generated_palette_color,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import (
    A1GeneratedMaterialPattern,
    A1MaterialSourcePolicy,
    BakeMode,
    BakeSettings,
    GeneratedBakePlan,
    GeneratedMaterialPlan,
    MaterialAnalysis,
    MaterialKind,
    ObjectMaterialAnalysis,
    build_bake_plan,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.baking.generated_materials import (
    A1GeneratedMaterialPattern as PrivateGeneratedMaterialPattern,
    A1MaterialSourcePolicy as PrivateMaterialSourcePolicy,
    GeneratedBakePlan as PrivateGeneratedBakePlan,
    GeneratedMaterialPlan as PrivateGeneratedMaterialPlan,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.geometry import MeshSnapshot


_EMPTY_WORLD_MATRIX = (
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


def _empty_generated_target() -> MeshSnapshot:
    return MeshSnapshot(
        snapshot_id="Hero:generated-material",
        source_object_id="Hero",
        object_name="Hero_GeneratedMaterial",
        vertices=(),
        edges=(),
        loops=(),
        faces=(),
        uv_layer_names=("SpineBakeUV",),
        active_uv_layer="SpineBakeUV",
        render_uv_layer="SpineBakeUV",
        world_matrix=_EMPTY_WORLD_MATRIX,
    )


def test_generated_material_contracts_are_public_identity_aliases():
    assert A1GeneratedMaterialPattern is PrivateGeneratedMaterialPattern
    assert A1MaterialSourcePolicy is PrivateMaterialSourcePolicy
    assert GeneratedBakePlan is PrivateGeneratedBakePlan
    assert GeneratedMaterialPlan is PrivateGeneratedMaterialPlan
    assert build_generated_material_plan is private_build_generated_material_plan
    assert generated_palette_color is private_generated_palette_color


def test_generated_bake_plan_preserves_base_plan_contract(tmp_path: Path):
    analysis = ObjectMaterialAnalysis(
        source_object_id="Hero",
        slots=(
            MaterialAnalysis(
                slot_index=0,
                material_name="__Spine2D_GeneratedMaterial",
                kind=MaterialKind.SOLID_COLOR,
                node_types=("EMISSION",),
            ),
        ),
    )
    base = build_bake_plan(
        analysis,
        BakeSettings(
            width=64,
            height=64,
            output_directory=tmp_path,
            output_stem="Hero",
            diffuse_mode=BakeMode.EMIT,
            procedural_mode=BakeMode.EMIT,
        ),
    )
    material = GeneratedMaterialPlan(
        source_policy=A1MaterialSourcePolicy.FORCE_GENERATED,
        pattern=A1GeneratedMaterialPattern.SOLID_GRAY,
        target_snapshot=_empty_generated_target(),
        face_colors=(),
    )

    generated = GeneratedBakePlan.from_bake_plan(base, material)

    assert isinstance(generated, GeneratedBakePlan)
    assert generated.generated_material is material
    assert generated.source_object_id == base.source_object_id
    assert generated.settings is base.settings
    assert generated.material_analysis is base.material_analysis
    assert generated.frame_tasks is base.frame_tasks
    assert generated.passes is base.passes
    assert generated.bake_mode is BakeMode.EMIT
    assert generated.passes[0].material_slot_indices == (0,)
