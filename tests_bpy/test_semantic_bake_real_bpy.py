"""Real Cycles bake regressions executed through the official Blender 5.2 bpy wheel."""

from __future__ import annotations

from pathlib import Path

import bpy
import pytest

from Blender_to_Spine2D_Mesh_Exporter.blender_adapter import execute_bake_plan
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter import (
    semantic_bake_execution,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.bake_execution_error import (
    BakeExecutionError,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import (
    BakeCompositeMode,
    BakeExecutionSettings,
    BakeMode,
    BakeStrategyId,
    MaterialSemanticChannel,
    TextureFormat,
)

from bake_test_support import (
    PNG_SIGNATURE,
    activate_only,
    capture_context,
    capture_scene_bake_state,
    create_quad,
    create_sentinel,
    create_two_quad_object,
    datablock_signature,
    dominant_pixel_count,
    load_image,
    material_fingerprint,
    mean_rgba,
    new_emission_material,
    new_image_alpha_material,
    new_principled_material,
    new_transparent_material,
    new_transparent_mix_material,
    prepare_bake_plan,
    temporary_datablock_names,
)


def _assert_runtime_restored(
    *,
    context_before,
    scene_before,
    datablocks_before,
    material_fingerprints_before: tuple[tuple[object, ...], ...],
    materials: tuple[object, ...],
) -> None:
    assert capture_context() == context_before
    assert capture_scene_bake_state() == scene_before
    assert datablock_signature() == datablocks_before
    assert tuple(material_fingerprint(material) for material in materials) == (
        material_fingerprints_before
    )
    assert temporary_datablock_names() == ()


def test_emit_bake_from_edit_mode_writes_valid_png_and_restores_everything(
    clean_blender_data,
    tmp_path: Path,
):
    source = create_quad("EditModeEmission")
    material, _color_socket = new_emission_material(
        "EditModeEmissionMaterial",
        (0.8, 0.15, 0.05),
    )
    source.data.materials.append(material)
    sentinel = create_sentinel()

    target, analysis, plan = prepare_bake_plan(
        source,
        tmp_path,
        "EditModeEmission",
        width=32,
        height=32,
    )
    assert analysis.slots[0].semantic_channels == (
        MaterialSemanticChannel.SURFACE_EMISSION,
    )
    assert tuple(item.strategy_id for item in plan.passes) == (
        BakeStrategyId.EMISSION,
    )

    bpy.context.scene.frame_set(37)
    activate_only(sentinel, mode="EDIT")
    source.select_set(False)
    context_before = capture_context()
    scene_before = capture_scene_bake_state()
    datablocks_before = datablock_signature()
    material_before = (material_fingerprint(material),)

    result = execute_bake_plan(
        source,
        target,
        plan,
        BakeExecutionSettings(samples=1),
    )

    assert len(result.artifacts) == 1
    output_path = result.representative_artifact.output_path
    assert output_path.is_file()
    assert output_path.stat().st_size > len(PNG_SIGNATURE)
    assert output_path.read_bytes()[: len(PNG_SIGNATURE)] == PNG_SIGNATURE

    image = load_image(output_path)
    assert (image.width, image.height, image.channels) == (32, 32, 4)
    assert len(image.rgba_pixels) == 32 * 32
    mean = mean_rgba(image.rgba_pixels)
    assert mean[0] > mean[1] > mean[2] > 0.0
    assert mean[3] == pytest.approx(1.0, abs=0.01)

    _assert_runtime_restored(
        context_before=context_before,
        scene_before=scene_before,
        datablocks_before=datablocks_before,
        material_fingerprints_before=material_before,
        materials=(material,),
    )


def test_forced_bake_failure_preserves_existing_file_and_has_no_false_completion(
    clean_blender_data,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    source = create_quad("FailureRollback")
    material, _color_socket = new_emission_material(
        "FailureRollbackMaterial",
        (0.7, 0.1, 0.05),
    )
    source.data.materials.append(material)
    sentinel = create_sentinel()
    target, _analysis, plan = prepare_bake_plan(
        source,
        tmp_path,
        "FailureRollback",
    )

    final_path = plan.representative_task.output_path
    final_path.parent.mkdir(parents=True, exist_ok=True)
    previous_content = b"previous-production-output"
    final_path.write_bytes(previous_content)

    activate_only(sentinel)
    source.select_set(False)
    context_before = capture_context()
    scene_before = capture_scene_bake_state()
    datablocks_before = datablock_signature()
    material_before = (material_fingerprint(material),)
    progress_updates = []

    def _fail_bake(
        _bpy_module,
        _bake_type: str,
        *,
        uv_layer_name: str,
    ) -> None:
        assert uv_layer_name == "SpineBakeUV"
        raise BakeExecutionError("forced real-bpy bake failure")

    monkeypatch.setattr(
        semantic_bake_execution,
        "_call_bake_operator",
        _fail_bake,
    )

    with pytest.raises(BakeExecutionError, match="forced real-bpy bake failure"):
        execute_bake_plan(
            source,
            target,
            plan,
            BakeExecutionSettings(samples=1),
            progress_callback=progress_updates.append,
        )

    assert final_path.read_bytes() == previous_content
    assert tuple(
        sorted(
            str(path.relative_to(tmp_path))
            for path in tmp_path.rglob("*")
            if path.is_file()
        )
    ) == (final_path.name,)
    assert tuple(update.message for update in progress_updates) == (
        "Baking frame 1/1",
    )

    _assert_runtime_restored(
        context_before=context_before,
        scene_before=scene_before,
        datablocks_before=datablocks_before,
        material_fingerprints_before=material_before,
        materials=(material,),
    )


def test_surface_and_emission_material_slots_are_composed_into_one_texture(
    clean_blender_data,
    tmp_path: Path,
):
    source = create_two_quad_object("MultipassSlots")
    surface = new_principled_material(
        "MultipassSurface",
        (0.8, 0.03, 0.02),
    )
    emission, _color_socket = new_emission_material(
        "MultipassEmission",
        (0.01, 0.05, 0.9),
    )
    source.data.materials.append(surface)
    source.data.materials.append(emission)
    source.data.polygons[0].material_index = 0
    source.data.polygons[1].material_index = 1
    sentinel = create_sentinel()

    target, analysis, plan = prepare_bake_plan(
        source,
        tmp_path,
        "MultipassSlots",
        width=64,
        height=64,
        diffuse_mode=BakeMode.DIFFUSE,
        procedural_mode=BakeMode.DIFFUSE,
    )
    assert tuple(slot.semantic_channels for slot in analysis.slots) == (
        (MaterialSemanticChannel.SURFACE_COLOR,),
        (MaterialSemanticChannel.SURFACE_EMISSION,),
    )
    assert tuple(item.strategy_id for item in plan.passes) == (
        BakeStrategyId.SURFACE_COLOR,
        BakeStrategyId.EMISSION,
    )
    assert plan.composite.mode is BakeCompositeMode.ADD_RGB_MAX_ALPHA

    activate_only(sentinel)
    source.select_set(False)
    context_before = capture_context()
    scene_before = capture_scene_bake_state()
    datablocks_before = datablock_signature()
    materials = (surface, emission)
    materials_before = tuple(material_fingerprint(item) for item in materials)

    result = execute_bake_plan(
        source,
        target,
        plan,
        BakeExecutionSettings(samples=1),
    )
    image = load_image(result.representative_artifact.output_path)
    minimum_region_pixels = int(image.width * image.height * 0.20)
    assert dominant_pixel_count(image, 0) > minimum_region_pixels
    assert dominant_pixel_count(image, 2) > minimum_region_pixels

    _assert_runtime_restored(
        context_before=context_before,
        scene_before=scene_before,
        datablocks_before=datablocks_before,
        material_fingerprints_before=materials_before,
        materials=materials,
    )


def test_principled_constant_alpha_is_preserved_in_committed_png(
    clean_blender_data,
    tmp_path: Path,
):
    source = create_quad("PrincipledAlpha")
    material = new_principled_material(
        "PrincipledAlphaMaterial",
        (0.8, 0.05, 0.02),
        alpha=0.35,
    )
    source.data.materials.append(material)
    sentinel = create_sentinel()

    target, analysis, plan = prepare_bake_plan(
        source,
        tmp_path,
        "PrincipledAlpha",
        width=48,
        height=48,
        diffuse_mode=BakeMode.DIFFUSE,
        procedural_mode=BakeMode.DIFFUSE,
    )
    assert set(analysis.slots[0].semantic_channels) == {
        MaterialSemanticChannel.SURFACE_COLOR,
        MaterialSemanticChannel.ALPHA,
    }
    assert tuple(item.strategy_id for item in plan.passes) == (
        BakeStrategyId.SURFACE_COLOR,
        BakeStrategyId.ALPHA,
    )
    assert plan.composite.mode is BakeCompositeMode.ADD_RGB_REPLACE_ALPHA

    activate_only(sentinel)
    source.select_set(False)
    context_before = capture_context()
    scene_before = capture_scene_bake_state()
    datablocks_before = datablock_signature()
    material_before = (material_fingerprint(material),)

    result = execute_bake_plan(
        source,
        target,
        plan,
        BakeExecutionSettings(samples=1),
    )
    image = load_image(result.representative_artifact.output_path)
    alpha_values = tuple(pixel[3] for pixel in image.rgba_pixels)
    matching = tuple(value for value in alpha_values if abs(value - 0.35) <= 0.08)
    assert len(matching) >= int(len(alpha_values) * 0.80)
    assert max(pixel[0] for pixel in image.rgba_pixels) > 0.35

    _assert_runtime_restored(
        context_before=context_before,
        scene_before=scene_before,
        datablocks_before=datablocks_before,
        material_fingerprints_before=material_before,
        materials=(material,),
    )


def test_image_texture_alpha_link_is_baked_without_removing_source_image(
    clean_blender_data,
    tmp_path: Path,
):
    source = create_quad("ImageAlpha")
    material, source_image = new_image_alpha_material(
        "ImageAlphaMaterial",
        alpha=0.62,
    )
    source.data.materials.append(material)
    sentinel = create_sentinel()

    target, analysis, plan = prepare_bake_plan(
        source,
        tmp_path,
        "ImageAlpha",
        width=40,
        height=40,
        diffuse_mode=BakeMode.DIFFUSE,
        procedural_mode=BakeMode.DIFFUSE,
    )
    assert set(analysis.slots[0].semantic_channels) == {
        MaterialSemanticChannel.SURFACE_COLOR,
        MaterialSemanticChannel.ALPHA,
    }
    assert plan.passes[-1].strategy_id is BakeStrategyId.ALPHA

    activate_only(sentinel)
    source.select_set(False)
    context_before = capture_context()
    scene_before = capture_scene_bake_state()
    datablocks_before = datablock_signature()
    material_before = (material_fingerprint(material),)

    result = execute_bake_plan(
        source,
        target,
        plan,
        BakeExecutionSettings(samples=1),
    )
    image = load_image(result.representative_artifact.output_path)
    alpha_values = tuple(pixel[3] for pixel in image.rgba_pixels)
    matching = tuple(value for value in alpha_values if abs(value - 0.62) <= 0.08)
    assert len(matching) >= int(len(alpha_values) * 0.80)
    assert max(pixel[1] for pixel in image.rgba_pixels) > 0.35
    assert source_image.name_full in bpy.data.images

    _assert_runtime_restored(
        context_before=context_before,
        scene_before=scene_before,
        datablocks_before=datablocks_before,
        material_fingerprints_before=material_before,
        materials=(material,),
    )


def test_mix_shader_socket_order_produces_both_expected_opacity_bands(
    clean_blender_data,
    tmp_path: Path,
):
    source = create_two_quad_object("MixShaderAlpha")
    low_opacity = new_transparent_mix_material(
        "TransparentFirst",
        factor=0.25,
        transparent_first=True,
    )
    high_opacity = new_transparent_mix_material(
        "TransparentSecond",
        factor=0.25,
        transparent_first=False,
    )
    source.data.materials.append(low_opacity)
    source.data.materials.append(high_opacity)
    source.data.polygons[0].material_index = 0
    source.data.polygons[1].material_index = 1
    sentinel = create_sentinel()

    target, analysis, plan = prepare_bake_plan(
        source,
        tmp_path,
        "MixShaderAlpha",
        width=64,
        height=64,
        diffuse_mode=BakeMode.DIFFUSE,
        procedural_mode=BakeMode.DIFFUSE,
    )
    assert all(
        MaterialSemanticChannel.ALPHA in slot.semantic_channels
        for slot in analysis.slots
    )

    activate_only(sentinel)
    source.select_set(False)
    context_before = capture_context()
    scene_before = capture_scene_bake_state()
    datablocks_before = datablock_signature()
    materials = (low_opacity, high_opacity)
    materials_before = tuple(material_fingerprint(item) for item in materials)

    result = execute_bake_plan(
        source,
        target,
        plan,
        BakeExecutionSettings(samples=1),
    )
    image = load_image(result.representative_artifact.output_path)
    alpha_values = tuple(pixel[3] for pixel in image.rgba_pixels)
    low_band = tuple(value for value in alpha_values if abs(value - 0.25) <= 0.10)
    high_band = tuple(value for value in alpha_values if abs(value - 0.75) <= 0.10)
    minimum_region_pixels = int(image.width * image.height * 0.20)
    assert len(low_band) > minimum_region_pixels
    assert len(high_band) > minimum_region_pixels

    _assert_runtime_restored(
        context_before=context_before,
        scene_before=scene_before,
        datablocks_before=datablocks_before,
        material_fingerprints_before=materials_before,
        materials=materials,
    )


def test_pure_transparent_material_produces_zero_straight_rgba(
    clean_blender_data,
    tmp_path: Path,
):
    source = create_quad("PureTransparent")
    material = new_transparent_material("PureTransparentMaterial")
    source.data.materials.append(material)
    target, analysis, plan = prepare_bake_plan(
        source,
        tmp_path,
        "PureTransparent",
        width=32,
        height=32,
        diffuse_mode=BakeMode.DIFFUSE,
        procedural_mode=BakeMode.DIFFUSE,
    )
    assert analysis.slots[0].semantic_channels == (
        MaterialSemanticChannel.ALPHA,
    )
    assert tuple(item.strategy_id for item in plan.passes) == (
        BakeStrategyId.ALPHA,
    )
    assert plan.requires_composition

    context_before = capture_context()
    scene_before = capture_scene_bake_state()
    datablocks_before = datablock_signature()
    material_before = (material_fingerprint(material),)

    result = execute_bake_plan(
        source,
        target,
        plan,
        BakeExecutionSettings(samples=1),
    )
    image = load_image(result.representative_artifact.output_path)
    for channel in range(4):
        assert max(pixel[channel] for pixel in image.rgba_pixels) <= 0.01

    _assert_runtime_restored(
        context_before=context_before,
        scene_before=scene_before,
        datablocks_before=datablocks_before,
        material_fingerprints_before=material_before,
        materials=(material,),
    )


def test_sequence_bake_writes_distinct_frames_restores_timeline_and_reports_progress(
    clean_blender_data,
    tmp_path: Path,
):
    source = create_quad("AnimatedEmission")
    material, color_socket = new_emission_material(
        "AnimatedEmissionMaterial",
        (1.0, 0.0, 0.0),
    )
    color_socket.default_value = (1.0, 0.0, 0.0, 1.0)
    color_socket.keyframe_insert(data_path="default_value", frame=1)
    color_socket.default_value = (0.0, 0.0, 1.0, 1.0)
    color_socket.keyframe_insert(data_path="default_value", frame=2)
    source.data.materials.append(material)
    sentinel = create_sentinel()

    target, analysis, plan = prepare_bake_plan(
        source,
        tmp_path,
        "AnimatedEmission",
        width=32,
        height=32,
        sequence_start_frame=1,
        sequence_frame_count=2,
    )
    assert analysis.has_animated_dependencies
    assert tuple(task.timeline_frame for task in plan.frame_tasks) == (1, 2)

    bpy.context.scene.frame_set(37)
    activate_only(sentinel)
    source.select_set(False)
    context_before = capture_context()
    scene_before = capture_scene_bake_state()
    datablocks_before = datablock_signature()
    material_before = (material_fingerprint(material),)
    progress_updates = []

    result = execute_bake_plan(
        source,
        target,
        plan,
        BakeExecutionSettings(samples=1),
        progress_callback=progress_updates.append,
    )
    assert tuple(artifact.timeline_frame for artifact in result.artifacts) == (1, 2)
    assert tuple(artifact.output_path.name for artifact in result.artifacts) == (
        "AnimatedEmission_Baked_0001.png",
        "AnimatedEmission_Baked_0002.png",
    )

    first = load_image(result.artifacts[0].output_path)
    second = load_image(result.artifacts[1].output_path)
    first_mean = mean_rgba(first.rgba_pixels)
    second_mean = mean_rgba(second.rgba_pixels)
    assert first_mean[0] > 0.9 and first_mean[2] < 0.05
    assert second_mean[2] > 0.9 and second_mean[0] < 0.05
    assert tuple(update.message for update in progress_updates) == (
        "Baking frame 1/2",
        "Baked frame 1/2",
        "Baking frame 2/2",
        "Baked frame 2/2",
    )
    assert tuple(update.percent for update in progress_updates) == (0, 50, 50, 100)

    _assert_runtime_restored(
        context_before=context_before,
        scene_before=scene_before,
        datablocks_before=datablocks_before,
        material_fingerprints_before=material_before,
        materials=(material,),
    )


@pytest.mark.parametrize(
    ("texture_format", "expected_suffix", "signature"),
    (
        (TextureFormat.JPEG, ".jpg", b"\xff\xd8\xff"),
        (TextureFormat.WEBP, ".webp", b"RIFF"),
        (TextureFormat.OPEN_EXR, ".exr", b"v/1\x01"),
    ),
)
def test_real_codec_outputs_are_saved_reloaded_and_restore_scene_format_state(
    clean_blender_data,
    tmp_path: Path,
    texture_format: TextureFormat,
    expected_suffix: str,
    signature: bytes,
):
    source = create_quad(f"Codec{texture_format.value}")
    material, _color_socket = new_emission_material(
        f"Codec{texture_format.value}Material",
        (0.8, 0.15, 0.05),
    )
    source.data.materials.append(material)
    sentinel = create_sentinel()
    target, _analysis, plan = prepare_bake_plan(
        source,
        tmp_path,
        f"Codec{texture_format.value}",
        width=20,
        height=20,
        texture_format=texture_format,
    )

    bpy.context.scene.render.image_settings.file_format = "PNG"
    bpy.context.scene.render.image_settings.color_mode = "RGBA"
    activate_only(sentinel)
    source.select_set(False)
    context_before = capture_context()
    scene_before = capture_scene_bake_state()
    datablocks_before = datablock_signature()
    material_before = (material_fingerprint(material),)

    result = execute_bake_plan(
        source,
        target,
        plan,
        BakeExecutionSettings(samples=1),
    )
    output_path = result.representative_artifact.output_path
    assert output_path.suffix == expected_suffix
    assert output_path.read_bytes().startswith(signature)
    image = load_image(output_path)
    assert (image.width, image.height, image.channels) == (20, 20, 4)
    mean = mean_rgba(image.rgba_pixels)
    assert mean[0] > mean[1] > mean[2] > 0.0
    if texture_format is not TextureFormat.JPEG:
        assert mean[3] == pytest.approx(1.0, abs=0.01)

    _assert_runtime_restored(
        context_before=context_before,
        scene_before=scene_before,
        datablocks_before=datablocks_before,
        material_fingerprints_before=material_before,
        materials=(material,),
    )


def test_selected_to_active_emit_bake_restores_selection_and_cage_settings(
    clean_blender_data,
    tmp_path: Path,
):
    source = create_quad("SelectedToActive")
    material, _color_socket = new_emission_material(
        "SelectedToActiveMaterial",
        (0.2, 0.7, 0.1),
    )
    source.data.materials.append(material)
    sentinel = create_sentinel()
    target, _analysis, plan = prepare_bake_plan(
        source,
        tmp_path,
        "SelectedToActive",
        width=24,
        height=24,
        selected_to_active=True,
    )

    activate_only(sentinel)
    source.select_set(False)
    context_before = capture_context()
    scene_before = capture_scene_bake_state()
    datablocks_before = datablock_signature()
    material_before = (material_fingerprint(material),)

    result = execute_bake_plan(
        source,
        target,
        plan,
        BakeExecutionSettings(samples=1),
    )
    image = load_image(result.representative_artifact.output_path)
    mean = mean_rgba(image.rgba_pixels)
    assert mean[1] > mean[0] > mean[2]
    assert mean[3] == pytest.approx(1.0, abs=0.01)

    _assert_runtime_restored(
        context_before=context_before,
        scene_before=scene_before,
        datablocks_before=datablocks_before,
        material_fingerprints_before=material_before,
        materials=(material,),
    )
