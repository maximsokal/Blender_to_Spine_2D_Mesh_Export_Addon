"""Extended real Blender 5.2 fixtures for B3 scene dependencies."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import traceback
from unittest import mock

import bpy

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIRECTORY.parents[1]
for path in (SCRIPT_DIRECTORY, REPOSITORY_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from Blender_to_Spine2D_Mesh_Exporter.blender_adapter import (  # noqa: E402
    BakeExecutionError,
    analyse_bake_contexts,
    analyse_object_materials,
    execute_bake_plan,
    read_source_mesh_snapshot,
)
import Blender_to_Spine2D_Mesh_Exporter.blender_adapter.semantic_bake_execution as bake_module  # noqa: E402
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import (  # noqa: E402
    BakeCompositeMode,
    BakeExecutionSettings,
    BakeMode,
    BakeSettings,
    BakeStrategyId,
    build_bake_plan,
)
from run_bake_integration import (  # noqa: E402
    _activate_only,
    _assert,
    _capture_context,
    _capture_scene_bake_state,
    _clear_scene,
    _create_mesh_object,
    _create_sentinel,
    _material_fingerprint,
    _temporary_datablock_names,
)
from run_scene_bake_integration import (  # noqa: E402
    _configure_cycles_scene,
    _create_area_light,
    _create_subsurface_material,
    _mean_luminance,
    _read_pixels,
)


def _build_plan(
    obj,
    output_directory: Path,
    output_stem: str,
    *,
    sequence_start: int = 0,
    sequence_count: int = 0,
):
    snapshot = read_source_mesh_snapshot(obj)
    analysis = analyse_object_materials(
        obj,
        source_object_id=snapshot.source_object_id,
    )
    object_context, scene_context = analyse_bake_contexts(
        obj,
        scene=bpy.context.scene,
        context=bpy.context,
    )
    plan = build_bake_plan(
        analysis,
        BakeSettings(
            width=64,
            height=64,
            output_directory=output_directory,
            output_stem=output_stem,
            uv_layer_name="UVMap",
            margin_pixels=1,
            diffuse_mode=BakeMode.DIFFUSE,
            procedural_mode=BakeMode.DIFFUSE,
            sequence_start_frame=sequence_start,
            sequence_frame_count=sequence_count,
        ),
        object_context=object_context,
        scene_context=scene_context,
    )
    return snapshot, analysis, plan


def _world_background_node():
    world = bpy.context.scene.world
    if world is None or world.node_tree is None:
        raise AssertionError("Scene World node tree is missing")
    for node in world.node_tree.nodes:
        if node.type == "BACKGROUND":
            return node
    raise AssertionError("Scene World has no Background node")


def _create_ao_emission_material(name: str):
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new(type="ShaderNodeOutputMaterial")
    ambient_occlusion = nodes.new(type="ShaderNodeAmbientOcclusion")
    ambient_occlusion.samples = 32
    ambient_occlusion.inputs["Distance"].default_value = 3.0
    emission = nodes.new(type="ShaderNodeEmission")
    emission.inputs["Strength"].default_value = 1.0
    material.node_tree.links.new(
        ambient_occlusion.outputs["AO"],
        emission.inputs["Color"],
    )
    material.node_tree.links.new(
        emission.outputs["Emission"],
        output.inputs["Surface"],
    )
    return material


def _create_cube_mesh(name: str, *, center=(0.0, 0.0, 0.5), size=1.2):
    cx, cy, cz = center
    half = size * 0.5
    vertices = (
        (cx - half, cy - half, cz - half),
        (cx + half, cy - half, cz - half),
        (cx + half, cy + half, cz - half),
        (cx - half, cy + half, cz - half),
        (cx - half, cy - half, cz + half),
        (cx + half, cy - half, cz + half),
        (cx + half, cy + half, cz + half),
        (cx - half, cy + half, cz + half),
    )
    faces = (
        (0, 1, 2, 3),
        (4, 7, 6, 5),
        (0, 4, 5, 1),
        (1, 5, 6, 2),
        (2, 6, 7, 3),
        (4, 0, 3, 7),
    )
    return _create_mesh_object(name, vertices, faces)


def _median_alpha(pixels: tuple[float, ...]) -> float:
    values = sorted(float(pixels[offset + 3]) for offset in range(0, len(pixels), 4))
    visible = [value for value in values if value > 0.01]
    if not visible:
        return 0.0
    return visible[len(visible) // 2]


def test_world_illumination_changes_scene_combined_output() -> None:
    _clear_scene()
    _configure_cycles_scene()
    with tempfile.TemporaryDirectory(prefix="spine2d-world-bake-") as directory:
        output_directory = Path(directory)
        source = _create_mesh_object(
            "WorldSurface",
            ((-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (1.0, 1.0, 0.0), (-1.0, 1.0, 0.0)),
            ((0, 1, 2, 3),),
        )
        material = _create_subsurface_material("WorldSubsurface", (0.8, 0.15, 0.03))
        source.data.materials.append(material)
        sentinel = _create_sentinel()
        sentinel.location.x = 20.0
        _activate_only(sentinel)
        source.select_set(False)
        background = _world_background_node()
        background.inputs["Color"].default_value = (0.8, 0.8, 0.8, 1.0)
        background.inputs["Strength"].default_value = 1.5

        snapshot, _, bright_plan = _build_plan(source, output_directory, "WorldBright")
        bright = execute_bake_plan(
            source,
            snapshot,
            bright_plan,
            BakeExecutionSettings(samples=8),
        )
        bright_luminance = _mean_luminance(
            _read_pixels(bright.representative_artifact.output_path)
        )

        background.inputs["Strength"].default_value = 0.0
        bpy.context.view_layer.update()
        _, _, dark_plan = _build_plan(source, output_directory, "WorldDark")
        dark = execute_bake_plan(
            source,
            snapshot,
            dark_plan,
            BakeExecutionSettings(samples=8),
        )
        dark_luminance = _mean_luminance(
            _read_pixels(dark.representative_artifact.output_path)
        )

        _assert(
            bright_luminance > dark_luminance + 0.03,
            f"World strength did not affect output: bright={bright_luminance}, dark={dark_luminance}",
        )
        _assert(not _temporary_datablock_names(), "World bake leaked temporary data")


def test_other_object_changes_ambient_occlusion_bake() -> None:
    _clear_scene()
    _configure_cycles_scene()
    with tempfile.TemporaryDirectory(prefix="spine2d-ao-bake-") as directory:
        output_directory = Path(directory)
        source = _create_mesh_object(
            "AoSurface",
            ((-2.0, -2.0, 0.0), (2.0, -2.0, 0.0), (2.0, 2.0, 0.0), (-2.0, 2.0, 0.0)),
            ((0, 1, 2, 3),),
        )
        material = _create_ao_emission_material("SceneAO")
        source.data.materials.append(material)
        occluder = _create_cube_mesh("AoOccluder", center=(0.0, 0.0, 0.55), size=1.4)
        sentinel = _create_sentinel()
        sentinel.location.x = 20.0
        _activate_only(sentinel)
        source.select_set(False)

        snapshot, analysis, occluded_plan = _build_plan(
            source,
            output_directory,
            "AoOccluded",
        )
        dependencies = {item.value for item in analysis.slots[0].dependencies}
        _assert("OCCLUSION" in dependencies, "AO node did not report OCCLUSION")
        _assert("SCENE_OBJECTS" in dependencies, "AO node did not report SCENE_OBJECTS")
        occluded = execute_bake_plan(
            source,
            snapshot,
            occluded_plan,
            BakeExecutionSettings(samples=4),
        )
        occluded_luminance = _mean_luminance(
            _read_pixels(occluded.representative_artifact.output_path)
        )

        occluder.hide_render = True
        bpy.context.view_layer.update()
        _, _, clear_plan = _build_plan(source, output_directory, "AoClear")
        clear = execute_bake_plan(
            source,
            snapshot,
            clear_plan,
            BakeExecutionSettings(samples=4),
        )
        clear_luminance = _mean_luminance(
            _read_pixels(clear.representative_artifact.output_path)
        )

        _assert(
            clear_luminance > occluded_luminance + 0.01,
            f"Other object did not affect AO: clear={clear_luminance}, occluded={occluded_luminance}",
        )
        _assert(not _temporary_datablock_names(), "AO bake leaked temporary data")


def test_animated_light_produces_distinct_sequence_frames() -> None:
    _clear_scene()
    _configure_cycles_scene()
    with tempfile.TemporaryDirectory(prefix="spine2d-light-sequence-") as directory:
        output_directory = Path(directory)
        source = _create_mesh_object(
            "AnimatedLightSurface",
            ((-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (1.0, 1.0, 0.0), (-1.0, 1.0, 0.0)),
            ((0, 1, 2, 3),),
        )
        material = _create_subsurface_material("AnimatedLightMaterial", (0.05, 0.7, 0.1))
        source.data.materials.append(material)
        light = _create_area_light("AnimatedKey", energy=0.0)
        light.data.energy = 0.0
        light.data.keyframe_insert(data_path="energy", frame=1)
        light.data.energy = 1800.0
        light.data.keyframe_insert(data_path="energy", frame=2)
        light.data.energy = 0.0
        light.data.keyframe_insert(data_path="energy", frame=3)
        bpy.context.scene.frame_set(1)
        sentinel = _create_sentinel()
        sentinel.location.x = 20.0
        _activate_only(sentinel)
        source.select_set(False)
        frame_before = int(bpy.context.scene.frame_current)

        snapshot, _, plan = _build_plan(
            source,
            output_directory,
            "AnimatedLight",
            sequence_start=1,
            sequence_count=3,
        )
        _assert(plan.sequence, "animated scene plan did not create sequence tasks")
        _assert(
            "AnimatedKey" in plan.scene_context.animated_dependency_ids,
            "animated light missing from scene context",
        )
        result = execute_bake_plan(
            source,
            snapshot,
            plan,
            BakeExecutionSettings(samples=4),
        )
        luminance = tuple(
            _mean_luminance(_read_pixels(artifact.output_path))
            for artifact in result.artifacts
        )

        _assert(
            luminance[1] > luminance[0] + 0.03 and luminance[1] > luminance[2] + 0.03,
            f"animated light frames are not distinct: {luminance}",
        )
        _assert(
            int(bpy.context.scene.frame_current) == frame_before,
            "animated scene bake did not restore frame",
        )
        _assert(not _temporary_datablock_names(), "animated scene bake leaked data")


def test_scene_alpha_composes_straight_rgba() -> None:
    _clear_scene()
    _configure_cycles_scene()
    with tempfile.TemporaryDirectory(prefix="spine2d-scene-alpha-") as directory:
        output_directory = Path(directory)
        source = _create_mesh_object(
            "SceneAlphaSurface",
            ((-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (1.0, 1.0, 0.0), (-1.0, 1.0, 0.0)),
            ((0, 1, 2, 3),),
        )
        material = _create_subsurface_material("SceneAlphaMaterial", (0.05, 0.1, 0.9))
        principled = next(node for node in material.node_tree.nodes if node.type == "BSDF_PRINCIPLED")
        principled.inputs["Alpha"].default_value = 0.4
        source.data.materials.append(material)
        _create_area_light(energy=1600.0)
        sentinel = _create_sentinel()
        sentinel.location.x = 20.0
        _activate_only(sentinel)
        source.select_set(False)

        snapshot, _, plan = _build_plan(source, output_directory, "SceneAlpha")
        _assert(
            tuple(item.strategy_id for item in plan.passes)
            == (BakeStrategyId.SCENE_COMBINED, BakeStrategyId.ALPHA),
            f"unexpected scene alpha plan: {plan.passes}",
        )
        _assert(
            plan.composite.mode is BakeCompositeMode.ADD_RGB_REPLACE_ALPHA,
            "scene alpha compositor mode is wrong",
        )
        _assert(plan.composite.unpremultiply_color_by_alpha, "scene RGB is not straightened")
        result = execute_bake_plan(
            source,
            snapshot,
            plan,
            BakeExecutionSettings(samples=8),
        )
        pixels = _read_pixels(result.representative_artifact.output_path)
        alpha = _median_alpha(pixels)
        blue_values = [
            pixels[offset + 2]
            for offset in range(0, len(pixels), 4)
            if pixels[offset + 3] > 0.05
        ]
        mean_blue = sum(blue_values) / len(blue_values)

        _assert(0.3 < alpha < 0.5, f"scene alpha was not preserved: {alpha}")
        _assert(mean_blue > 0.15, f"scene straight RGB is too dark: {mean_blue}")
        _assert(not _temporary_datablock_names(), "scene alpha bake leaked data")


def test_scene_bake_failure_restores_source_and_existing_output() -> None:
    _clear_scene()
    _configure_cycles_scene()
    with tempfile.TemporaryDirectory(prefix="spine2d-scene-rollback-") as directory:
        output_directory = Path(directory)
        source = _create_mesh_object(
            "SceneRollbackSurface",
            ((-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (1.0, 1.0, 0.0), (-1.0, 1.0, 0.0)),
            ((0, 1, 2, 3),),
        )
        material = _create_subsurface_material("SceneRollbackMaterial", (0.8, 0.05, 0.02))
        source.data.materials.append(material)
        _create_area_light(energy=1200.0)
        sentinel = _create_sentinel()
        sentinel.location.x = 20.0
        _activate_only(sentinel)
        source.select_set(False)
        snapshot, _, plan = _build_plan(source, output_directory, "SceneRollback")
        final_path = plan.representative_task.output_path
        final_path.parent.mkdir(parents=True, exist_ok=True)
        previous = b"previous-scene-output"
        final_path.write_bytes(previous)
        context_before = _capture_context()
        scene_before = _capture_scene_bake_state()
        material_before = _material_fingerprint(material)
        hide_before = bool(source.hide_render)

        with mock.patch.object(
            bake_module,
            "_call_bake_operator",
            side_effect=BakeExecutionError("forced scene bake failure"),
        ):
            try:
                execute_bake_plan(
                    source,
                    snapshot,
                    plan,
                    BakeExecutionSettings(samples=1),
                )
            except BakeExecutionError as exc:
                _assert("forced scene bake failure" in str(exc), "primary error was hidden")
            else:
                raise AssertionError("forced scene bake failure did not propagate")

        _assert(final_path.read_bytes() == previous, "existing scene output was corrupted")
        leftovers = tuple(path.name for path in output_directory.iterdir())
        _assert(leftovers == (final_path.name,), f"rollback left files: {leftovers}")
        _assert(bool(source.hide_render) == hide_before, "rollback did not restore hide_render")
        _assert(_capture_context() == context_before, "rollback changed context")
        _assert(_capture_scene_bake_state() == scene_before, "rollback changed scene")
        _assert(_material_fingerprint(material) == material_before, "rollback mutated material")
        _assert(not _temporary_datablock_names(), "rollback leaked temporary data")


def main() -> None:
    print(f"Blender version: {bpy.app.version_string}")
    tests = (
        test_world_illumination_changes_scene_combined_output,
        test_other_object_changes_ambient_occlusion_bake,
        test_animated_light_produces_distinct_sequence_frames,
        test_scene_alpha_composes_straight_rgba,
        test_scene_bake_failure_restores_source_and_existing_output,
    )
    for test in tests:
        print(f"[SCENE-EXTENDED] RUN {test.__name__}")
        test()
        print(f"[SCENE-EXTENDED] PASS {test.__name__}")
    print(f"[SCENE-EXTENDED] PASS {len(tests)} integration tests")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
