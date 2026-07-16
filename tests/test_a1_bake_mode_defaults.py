from pathlib import Path

import pytest

from Blender_to_Spine2D_Mesh_Exporter.application import (
    A1SingleObjectExportSettings,
    ExportSettings,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import (
    BakeMode,
    BakePlanError,
    BakeSettings,
    MaterialAnalysis,
    MaterialKind,
    ObjectMaterialAnalysis,
    build_bake_plan,
)


def test_a1_defaults_to_lighting_independent_diffuse_color_bake(tmp_path: Path):
    settings = A1SingleObjectExportSettings(
        export=ExportSettings(
            texture_width=64,
            texture_height=64,
            output_directory=tmp_path,
        )
    )

    assert settings.diffuse_mode is BakeMode.DIFFUSE
    assert settings.procedural_mode is BakeMode.DIFFUSE


def test_pure_emission_materials_use_emit_even_with_diffuse_a1_defaults(
    tmp_path: Path,
):
    emission = MaterialAnalysis(
        slot_index=0,
        material_name="Glow",
        kind=MaterialKind.SOLID_COLOR,
        node_types=("EMISSION", "OUTPUT_MATERIAL"),
    )
    plan = build_bake_plan(
        ObjectMaterialAnalysis("GlowObject", (emission,)),
        BakeSettings(
            width=64,
            height=64,
            output_directory=tmp_path,
            output_stem="GlowObject",
            procedural_mode=BakeMode.DIFFUSE,
        ),
    )

    assert plan.bake_mode is BakeMode.EMIT


def test_mixed_emission_and_surface_slots_fail_instead_of_writing_partial_black(
    tmp_path: Path,
):
    emission = MaterialAnalysis(
        slot_index=0,
        material_name="Glow",
        kind=MaterialKind.SOLID_COLOR,
        node_types=("EMISSION", "OUTPUT_MATERIAL"),
    )
    surface = MaterialAnalysis(
        slot_index=1,
        material_name="Body",
        kind=MaterialKind.SOLID_COLOR,
        node_types=("BSDF_PRINCIPLED", "OUTPUT_MATERIAL"),
    )

    with pytest.raises(BakePlanError, match="emission-only and surface"):
        build_bake_plan(
            ObjectMaterialAnalysis("MixedObject", (emission, surface)),
            BakeSettings(
                width=64,
                height=64,
                output_directory=tmp_path,
                output_stem="MixedObject",
                procedural_mode=BakeMode.DIFFUSE,
            ),
        )
