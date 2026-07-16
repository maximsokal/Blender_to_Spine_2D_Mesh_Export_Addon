"""Real Blender 4.4 coverage for B3 scene- and camera-aware bake strategies."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import traceback

import bpy
from mathutils import Vector

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIRECTORY.parents[1]
for path in (SCRIPT_DIRECTORY, REPOSITORY_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from Blender_to_Spine2D_Mesh_Exporter.blender_adapter import (  # noqa: E402
    analyse_bake_contexts,
    analyse_object_materials,
    execute_bake_plan,
    read_source_mesh_snapshot,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import (  # noqa: E402
    BakeEvaluationScope,
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


def _configure_cycles_scene() -> None:
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 1
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "Medium High Contrast"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0

    world = bpy.data.worlds.new("SceneBakeWorld")
    world.use_nodes = True
    nodes = world.node_tree.nodes
    nodes.clear()
    output = nodes.new(type="ShaderNodeOutputWorld")
    background = nodes.new(type="ShaderNodeBackground")
    background.inputs["Color"].default_value = (0.02, 0.02, 0.02, 1.0)
    background.inputs["Strength"].default_value = 0.0
    world.node_tree.links.new(background.outputs["Background"], output.inputs["Surface"])
    scene.world = world


def _create_area_light(name: str = "SceneKey", energy: float = 1200.0):
    data = bpy.data.lights.new(name=f"{name}_Data", type="AREA")
    data.energy = energy
    data.color = (1.0, 1.0, 1.0)
    data.shape = "DISK"
    data.size = 5.0
    obj = bpy.data.objects.new(name, data)
    bpy.context.scene.collection.objects.link(obj)
    obj.location = (0.0, 0.0, 4.0)
    return obj


def _create_camera(name: str = "SceneCamera"):
    data = bpy.data.cameras.new(name=f"{name}_Data")
    data.type = "PERSP"
    data.lens = 50.0
    data.clip_start = 0.1
    data.clip_end = 100.0
    obj = bpy.data.objects.new(name, data)
    bpy.context.scene.collection.objects.link(obj)
    obj.location = (0.0, 0.0, 5.0)
    _aim_at(obj, Vector((0.0, 0.0, 0.0)))
    bpy.context.scene.camera = obj
    return obj


def _aim_at(obj, target: Vector) -> None:
    direction = target - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def _create_subsurface_material(name: str, color):
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new(type="ShaderNodeOutputMaterial")
    principled = nodes.new(type="ShaderNodeBsdfPrincipled")
    principled.inputs["Base Color"].default_value = (*color, 1.0)
    principled.inputs["Roughness"].default_value = 1.0
    subsurface = principled.inputs.get("Subsurface Weight") or principled.inputs.get(
        "Subsurface"
    )
    if subsurface is None:
        raise AssertionError("Blender 4.4 Principled has no Subsurface input")
    subsurface.default_value = 0.35
    material.node_tree.links.new(principled.outputs["BSDF"], output.inputs["Surface"])
    return material


def _create_local_principled_material(name: str, color):
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new(type="ShaderNodeOutputMaterial")
    principled = nodes.new(type="ShaderNodeBsdfPrincipled")
    principled.inputs["Base Color"].default_value = (*color, 1.0)
    principled.inputs["Roughness"].default_value = 1.0
    material.node_tree.links.new(principled.outputs["BSDF"], output.inputs["Surface"])
    return material


def _create_layer_weight_emission_material(name: str):
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new(type="ShaderNodeOutputMaterial")
    layer_weight = nodes.new(type="ShaderNodeLayerWeight")
    ramp = nodes.new(type="ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.0
    ramp.color_ramp.elements[0].color = (0.01, 0.02, 0.1, 1.0)
    ramp.color_ramp.elements[1].position = 1.0
    ramp.color_ramp.elements[1].color = (0.9, 0.2, 0.05, 1.0)
    emission = nodes.new(type="ShaderNodeEmission")
    emission.inputs["Strength"].default_value = 1.0
    material.node_tree.links.new(layer_weight.outputs["Facing"], ramp.inputs["Fac"])
    material.node_tree.links.new(ramp.outputs["Color"], emission.inputs["Color"])
    material.node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return material


def _create_two_quad_object(name: str):
    return _create_mesh_object(
        name,
        (
            (-2.0, -1.0, 0.0),
            (-0.2, -1.0, 0.0),
            (-0.2, 1.0, 0.0),
            (-2.0, 1.0, 0.0),
            (0.2, -1.0, 0.0),
            (2.0, -1.0, 0.0),
            (2.0, 1.0, 0.0),
            (0.2, 1.0, 0.0),
        ),
        ((0, 1, 2, 3), (4, 5, 6, 7)),
    )


def _prepare_plan(obj, output_directory: Path, output_stem: str):
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
        ),
        object_context=object_context,
        scene_context=scene_context,
    )
    return snapshot, analysis, plan


def _read_pixels(path: Path) -> tuple[float, ...]:
    image = bpy.data.images.load(str(path), check_existing=False)
    try:
        return tuple(float(value) for value in image.pixels[:])
    finally:
        bpy.data.images.remove(image)


def _mean_luminance(pixels: tuple[float, ...]) -> float:
    values = []
    for offset in range(0, len(pixels), 4):
        red, green, blue, alpha = pixels[offset : offset + 4]
        if alpha <= 0.05:
            continue
        values.append(0.2126 * red + 0.7152 * green + 0.0722 * blue)
    if not values:
        raise AssertionError("Decoded image has no visible pixels")
    return sum(values) / len(values)


def _dominant_count(pixels: tuple[float, ...], channel: int) -> int:
    count = 0
    for offset in range(0, len(pixels), 4):
        red, green, blue, alpha = pixels[offset : offset + 4]
        if alpha <= 0.05:
            continue
        values = (red, green, blue)
        others = tuple(values[index] for index in range(3) if index != channel)
        if values[channel] > 0.08 and values[channel] > max(others) * 1.25:
            count += 1
    return count


def test_scene_combined_responds_to_light_energy() -> None:
    _clear_scene()
    _configure_cycles_scene()
    with tempfile.TemporaryDirectory(prefix="spine2d-scene-combined-") as directory:
        output_directory = Path(directory)
        source = _create_mesh_object(
            "SceneSurface",
            ((-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (1.0, 1.0, 0.0), (-1.0, 1.0, 0.0)),
            ((0, 1, 2, 3),),
        )
        material = _create_subsurface_material("SceneSubsurface", (0.05, 0.15, 0.9))
        source.data.materials.append(material)
        light = _create_area_light()
        sentinel = _create_sentinel()
        sentinel.location.x = 20.0
        _activate_only(sentinel)
        source.select_set(False)

        snapshot, analysis, bright_plan = _prepare_plan(
            source,
            output_directory,
            "SceneBright",
        )
        _assert(
            tuple(item.strategy_id for item in bright_plan.passes)
            == (BakeStrategyId.SCENE_COMBINED,),
            f"unexpected scene plan: {bright_plan.passes}",
        )
        _assert(
            bright_plan.passes[0].evaluation_scope is BakeEvaluationScope.SCENE,
            "scene material did not receive SCENE scope",
        )
        _assert(
            "LIGHTING" in {item.value for item in analysis.slots[0].dependencies},
            "subsurface material did not report LIGHTING dependency",
        )

        context_before = _capture_context()
        scene_before = _capture_scene_bake_state()
        material_before = _material_fingerprint(material)
        hide_before = bool(source.hide_render)
        bright = execute_bake_plan(
            source,
            snapshot,
            bright_plan,
            BakeExecutionSettings(samples=4),
        )
        bright_luminance = _mean_luminance(
            _read_pixels(bright.representative_artifact.output_path)
        )

        light.data.energy = 0.0
        bpy.context.view_layer.update()
        _, _, dark_plan = _prepare_plan(source, output_directory, "SceneDark")
        dark = execute_bake_plan(
            source,
            snapshot,
            dark_plan,
            BakeExecutionSettings(samples=4),
        )
        dark_luminance = _mean_luminance(
            _read_pixels(dark.representative_artifact.output_path)
        )

        _assert(
            bright_luminance > dark_luminance + 0.03,
            f"scene bake ignored light energy: bright={bright_luminance}, dark={dark_luminance}",
        )
        _assert(bool(source.hide_render) == hide_before, "source hide_render was not restored")
        _assert(_capture_context() == context_before, "scene bake changed Blender context")
        _assert(_capture_scene_bake_state() == scene_before, "scene bake changed bake settings")
        _assert(_material_fingerprint(material) == material_before, "scene bake mutated material")
        _assert(not _temporary_datablock_names(), "scene bake leaked temporary datablocks")


def test_active_camera_bake_executes_view_dependent_graph() -> None:
    _clear_scene()
    _configure_cycles_scene()
    with tempfile.TemporaryDirectory(prefix="spine2d-camera-combined-") as directory:
        output_directory = Path(directory)
        source = _create_mesh_object(
            "CameraSurface",
            ((-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (1.0, 1.0, 0.0), (-1.0, 1.0, 0.0)),
            ((0, 1, 2, 3),),
        )
        material = _create_layer_weight_emission_material("ViewEmission")
        source.data.materials.append(material)
        camera = _create_camera()
        sentinel = _create_sentinel()
        sentinel.location.x = 20.0
        _activate_only(sentinel)
        source.select_set(False)

        snapshot, analysis, plan = _prepare_plan(
            source,
            output_directory,
            "CameraFacing",
        )
        _assert(
            tuple(item.strategy_id for item in plan.passes)
            == (BakeStrategyId.CAMERA_COMBINED,),
            f"unexpected camera plan: {plan.passes}",
        )
        _assert(plan.passes[0].bake_mode is BakeMode.ACTIVE_CAMERA, "camera mode missing")
        _assert(
            plan.passes[0].evaluation_scope is BakeEvaluationScope.CAMERA,
            "camera graph did not receive CAMERA scope",
        )
        dependency_values = {item.value for item in analysis.slots[0].dependencies}
        _assert("VIEW" in dependency_values, "Layer Weight did not report VIEW")
        _assert("CAMERA" in dependency_values, "Layer Weight did not report CAMERA")

        context_before = _capture_context()
        scene_before = _capture_scene_bake_state()
        material_before = _material_fingerprint(material)
        hide_before = bool(source.hide_render)
        result = execute_bake_plan(
            source,
            snapshot,
            plan,
            BakeExecutionSettings(samples=1),
        )
        luminance = _mean_luminance(_read_pixels(result.representative_artifact.output_path))

        _assert(luminance > 0.02, f"ACTIVE_CAMERA output is unusable: {luminance}")
        _assert(bpy.context.scene.camera is camera, "active camera changed")
        _assert(bool(source.hide_render) == hide_before, "camera bake did not restore source")
        _assert(_capture_context() == context_before, "camera bake changed context")
        _assert(_capture_scene_bake_state() == scene_before, "camera bake changed scene settings")
        _assert(_material_fingerprint(material) == material_before, "camera bake mutated material")
        _assert(not _temporary_datablock_names(), "camera bake leaked temporary datablocks")


def test_mixed_local_and_scene_slots_are_composed_without_double_counting() -> None:
    _clear_scene()
    _configure_cycles_scene()
    with tempfile.TemporaryDirectory(prefix="spine2d-mixed-scopes-") as directory:
        output_directory = Path(directory)
        source = _create_two_quad_object("MixedScopes")
        local = _create_local_principled_material("LocalRed", (0.85, 0.02, 0.01))
        scene_material = _create_subsurface_material("SceneBlue", (0.01, 0.05, 0.9))
        source.data.materials.append(local)
        source.data.materials.append(scene_material)
        source.data.polygons[0].material_index = 0
        source.data.polygons[1].material_index = 1
        _create_area_light(energy=1500.0)
        sentinel = _create_sentinel()
        sentinel.location.x = 20.0
        _activate_only(sentinel)
        source.select_set(False)

        snapshot, _, plan = _prepare_plan(source, output_directory, "MixedScopes")
        _assert(
            tuple(item.strategy_id for item in plan.passes)
            == (BakeStrategyId.SCENE_COMBINED, BakeStrategyId.SURFACE_COLOR),
            f"unexpected mixed-scope plan: {plan.passes}",
        )
        context_before = _capture_context()
        scene_before = _capture_scene_bake_state()
        local_before = _material_fingerprint(local)
        scene_before_material = _material_fingerprint(scene_material)
        hide_before = bool(source.hide_render)

        result = execute_bake_plan(
            source,
            snapshot,
            plan,
            BakeExecutionSettings(samples=4),
        )
        pixels = _read_pixels(result.representative_artifact.output_path)

        _assert(_dominant_count(pixels, 0) > 20, "local red contribution is missing")
        _assert(_dominant_count(pixels, 2) > 20, "scene blue contribution is missing")
        _assert(bool(source.hide_render) == hide_before, "mixed bake did not restore source")
        _assert(_capture_context() == context_before, "mixed bake changed context")
        _assert(_capture_scene_bake_state() == scene_before, "mixed bake changed scene settings")
        _assert(_material_fingerprint(local) == local_before, "local material mutated")
        _assert(
            _material_fingerprint(scene_material) == scene_before_material,
            "scene material mutated",
        )
        _assert(not _temporary_datablock_names(), "mixed bake leaked temporary datablocks")


def main() -> None:
    print(f"Blender version: {bpy.app.version_string}")
    tests = (
        test_scene_combined_responds_to_light_energy,
        test_active_camera_bake_executes_view_dependent_graph,
        test_mixed_local_and_scene_slots_are_composed_without_double_counting,
    )
    for test in tests:
        print(f"[SCENE-BAKE] RUN {test.__name__}")
        test()
        print(f"[SCENE-BAKE] PASS {test.__name__}")
    print(f"[SCENE-BAKE] PASS {len(tests)} integration tests")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
