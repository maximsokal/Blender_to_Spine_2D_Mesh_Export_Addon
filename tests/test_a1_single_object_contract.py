from dataclasses import replace
from pathlib import Path

import pytest

from Blender_to_Spine2D_Mesh_Exporter.application import (
    A1SingleObjectExportSettings,
    A1SingleObjectStage,
    ExportSettings,
    build_a1_attachment_path,
    build_a1_attachment_sequence,
    build_a1_bake_settings,
    calculate_a1_main_position_pixels,
    calculate_a1_mesh_bounds,
    resolve_a1_output_paths,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import (
    BakeMode,
    MaterialAnalysis,
    MaterialKind,
    ObjectMaterialAnalysis,
    build_bake_plan,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.geometry import SegmentationSettings

from test_geometry_domain import build_square_snapshot


def build_settings(tmp_path, **overrides):
    values = {
        "export": ExportSettings(
            texture_width=128,
            texture_height=64,
            output_directory=tmp_path,
            images_relative_path="textures/spine",
            angle_limit_degrees=42.0,
        )
    }
    values.update(overrides)
    return A1SingleObjectExportSettings(**values)


def test_auto_and_custom_modes_resolve_segmentation_without_mutating_source(tmp_path):
    base = build_settings(tmp_path)
    original = base.geometry.segmentation

    auto = base.resolved_geometry_settings().segmentation
    custom = replace(
        base,
        export=replace(base.export, seam_mode="CUSTOM"),
    ).resolved_geometry_settings().segmentation

    assert original == SegmentationSettings()
    assert auto.split_by_angle
    assert auto.angle_limit_degrees == 42.0
    assert auto.respect_seams
    assert not custom.split_by_angle
    assert custom.respect_seams
    assert base.geometry.segmentation == original


def test_output_paths_are_normalized_and_remain_under_export_root(tmp_path):
    settings = build_settings(tmp_path, output_stem="My:Mesh")
    paths = resolve_a1_output_paths("Object", settings)

    assert paths.output_stem == "My_Mesh"
    assert paths.json_path == tmp_path.resolve() / "My_Mesh.json"
    assert paths.image_directory == tmp_path.resolve() / "textures" / "spine"
    assert paths.image_relative_directory == "textures/spine"


@pytest.mark.parametrize("value", ("../escape", "safe/../../escape", "/absolute"))
def test_output_paths_reject_unsafe_image_directories(tmp_path, value):
    settings = A1SingleObjectExportSettings(
        export=ExportSettings(
            texture_width=64,
            texture_height=64,
            output_directory=tmp_path,
            images_relative_path=value,
        )
    )
    with pytest.raises(ValueError):
        resolve_a1_output_paths("Object", settings)


def test_world_translation_is_converted_with_legacy_uniform_scale(tmp_path):
    snapshot = build_square_snapshot()
    translated = replace(
        snapshot,
        world_matrix=(
            1.0,
            0.0,
            0.0,
            2.0,
            0.0,
            1.0,
            0.0,
            -3.0,
            0.0,
            0.0,
            1.0,
            5.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ),
    )
    settings = build_settings(tmp_path)

    # AVERAGE scale for 128x64 is 96.
    assert calculate_a1_main_position_pixels(translated, settings) == (192.0, -288.0)
    assert calculate_a1_main_position_pixels(
        translated,
        replace(settings, use_world_location_for_main_bone=False),
    ) is None


def test_bounds_are_calculated_from_geometry_not_uv(tmp_path):
    bounds = calculate_a1_mesh_bounds(build_square_snapshot())
    assert bounds.minimum_x == 0.0
    assert bounds.maximum_x == 1.0
    assert bounds.minimum_y == 0.0
    assert bounds.maximum_y == 1.0
    assert bounds.center_x == 0.5
    assert bounds.center_y == 0.5


def test_bake_and_attachment_paths_share_one_deterministic_stem(tmp_path):
    settings = build_settings(tmp_path, output_stem="Hero")
    paths = resolve_a1_output_paths("Object", settings)
    analysis = ObjectMaterialAnalysis(
        source_object_id="Object",
        slots=(
            MaterialAnalysis(
                slot_index=0,
                material_name="Material",
                kind=MaterialKind.SOLID_COLOR,
            ),
        ),
    )
    plan = build_bake_plan(
        analysis,
        build_a1_bake_settings("Object", settings),
    )

    assert plan.bake_mode is BakeMode.COMBINED
    assert plan.representative_task.output_path == (
        tmp_path.resolve() / "textures" / "spine" / "Hero_Baked.png"
    )
    assert build_a1_attachment_path(plan, paths) == "textures/spine/Hero_Baked"
    assert build_a1_attachment_sequence(plan) is None


def test_sequence_path_uses_common_prefix_and_spine_metadata(tmp_path):
    base = build_settings(tmp_path, output_stem="Hero")
    settings = replace(
        base,
        export=replace(
            base.export,
            sequence_start_frame=7,
            sequence_frame_count=3,
        ),
    )
    paths = resolve_a1_output_paths("Object", settings)
    analysis = ObjectMaterialAnalysis(
        source_object_id="Object",
        slots=(
            MaterialAnalysis(
                slot_index=0,
                material_name="Material",
                kind=MaterialKind.SOLID_COLOR,
            ),
        ),
    )
    plan = build_bake_plan(
        analysis,
        build_a1_bake_settings("Object", settings),
    )
    sequence = build_a1_attachment_sequence(plan)

    assert build_a1_attachment_path(plan, paths) == "textures/spine/Hero_Baked_"
    assert tuple(task.timeline_frame for task in plan.frame_tasks) == (7, 8, 9)
    assert sequence is not None
    assert sequence.count == 3
    assert sequence.start == 7
    assert sequence.digits == 4
    assert sequence.resolved_setup == 1


def test_stage_error_codes_are_unique_and_stable():
    codes = tuple(stage.error_code for stage in A1SingleObjectStage)
    assert len(codes) == len(set(codes))
    assert A1SingleObjectStage.UNWRAP_TEXTURE_UV.error_code == (
        "A1_UNWRAP_TEXTURE_UV_FAILED"
    )
