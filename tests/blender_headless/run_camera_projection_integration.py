"""Real Blender 5.2 end-to-end tests for B4 camera-render projection."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import traceback
from unittest import mock

import bpy
from mathutils import Vector

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIRECTORY.parents[1]
for path in (SCRIPT_DIRECTORY, REPOSITORY_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from Blender_to_Spine2D_Mesh_Exporter.application import (  # noqa: E402
    A1SingleObjectExportSettings,
    A1SourceGeometryMode,
    ExportSettings,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter import (  # noqa: E402
    CameraProjectionExecutionError,
    export_a1_single_object,
    prepare_a1_object,
)
import Blender_to_Spine2D_Mesh_Exporter.blender_adapter.camera_projection_execution as render_module  # noqa: E402
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import (  # noqa: E402
    A1TextureExportMode,
    BakeExecutionSettings,
    CameraProjectionPlan,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.uv import UvUnwrapSettings  # noqa: E402
from run_bake_integration import (  # noqa: E402
    _activate_only,
    _assert,
    _capture_context,
    _clear_scene,
    _create_mesh_object,
    _create_sentinel,
    _material_fingerprint,
    _temporary_datablock_names,
)

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _purge_orphan_scene_data() -> None:
    for collection in (bpy.data.cameras, bpy.data.lights, bpy.data.worlds):
        for datablock in tuple(collection):
            if datablock.users == 0:
                collection.remove(datablock)


def _configure_scene() -> None:
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 2
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    scene.render.film_transparent = False
    scene.render.resolution_x = 37
    scene.render.resolution_y = 29
    scene.render.resolution_percentage = 73
    scene.render.filepath = "//user-render-path.png"
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "Medium High Contrast"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0

    world = bpy.data.worlds.new("ProjectionWorld")
    world.use_nodes = True
    nodes = world.node_tree.nodes
    nodes.clear()
    output = nodes.new(type="ShaderNodeOutputWorld")
    background = nodes.new(type="ShaderNodeBackground")
    background.inputs["Color"].default_value = (0.03, 0.05, 0.12, 1.0)
    background.inputs["Strength"].default_value = 0.35
    world.node_tree.links.new(background.outputs["Background"], output.inputs["Surface"])
    scene.world = world


def _aim_at(obj, target: Vector) -> None:
    obj.rotation_euler = (target - obj.location).to_track_quat("-Z", "Y").to_euler()


def _create_camera(name: str = "ProjectionCamera"):
    data = bpy.data.cameras.new(name=f"{name}_Data")
    data.type = "PERSP"
    data.lens = 52.0
    data.clip_start = 0.1
    data.clip_end = 100.0
    obj = bpy.data.objects.new(name, data)
    bpy.context.scene.collection.objects.link(obj)
    obj.location = (0.0, 0.0, 5.5)
    _aim_at(obj, Vector((0.0, 0.0, 0.0)))
    bpy.context.scene.camera = obj
    return obj


def _create_area_light(name: str = "ProjectionKey", energy: float = 700.0):
    data = bpy.data.lights.new(name=f"{name}_Data", type="AREA")
    data.energy = energy
    data.shape = "DISK"
    data.size = 4.0
    obj = bpy.data.objects.new(name, data)
    bpy.context.scene.collection.objects.link(obj)
    obj.location = (2.5, -2.0, 4.0)
    _aim_at(obj, Vector((0.0, 0.0, 0.0)))
    return obj


def _create_layer_weight_material(name: str):
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new(type="ShaderNodeOutputMaterial")
    layer_weight = nodes.new(type="ShaderNodeLayerWeight")
    ramp = nodes.new(type="ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (0.01, 0.03, 0.2, 1.0)
    ramp.color_ramp.elements[1].color = (1.0, 0.08, 0.01, 1.0)
    emission = nodes.new(type="ShaderNodeEmission")
    emission.inputs["Strength"].default_value = 2.0
    material.node_tree.links.new(layer_weight.outputs["Facing"], ramp.inputs["Fac"])
    material.node_tree.links.new(ramp.outputs["Color"], emission.inputs["Color"])
    material.node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return material


def _create_glass_material(name: str):
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new(type="ShaderNodeOutputMaterial")
    glass = nodes.new(type="ShaderNodeBsdfGlass")
    glass.inputs["Color"].default_value = (0.1, 0.7, 1.0, 1.0)
    glass.inputs["Roughness"].default_value = 0.08
    glass.inputs["IOR"].default_value = 1.35
    material.node_tree.links.new(glass.outputs["BSDF"], output.inputs["Surface"])
    return material


def _create_volume_material(name: str):
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new(type="ShaderNodeOutputMaterial")
    volume = nodes.new(type="ShaderNodeVolumePrincipled")
    volume.inputs["Density"].default_value = 1.5
    volume.inputs["Color"].default_value = (0.04, 0.15, 0.8, 1.0)
    emission_color = volume.inputs.get("Emission Color") or volume.inputs.get("Emission")
    if emission_color is not None:
        emission_color.default_value = (0.03, 0.1, 0.6, 1.0)
    emission_strength = volume.inputs.get("Emission Strength")
    if emission_strength is not None:
        emission_strength.default_value = 0.7
    material.node_tree.links.new(volume.outputs["Volume"], output.inputs["Volume"])
    return material


def _create_quad(name: str):
    return _create_mesh_object(
        name,
        ((-1.5, -1.0, 0.0), (1.5, -1.0, 0.0), (1.5, 1.0, 0.0), (-1.5, 1.0, 0.0)),
        ((0, 1, 2, 3),),
    )


def _create_cube(name: str):
    return _create_mesh_object(
        name,
        (
            (-0.9, -0.9, -0.9), (0.9, -0.9, -0.9), (0.9, 0.9, -0.9),
            (-0.9, 0.9, -0.9), (-0.9, -0.9, 0.9), (0.9, -0.9, 0.9),
            (0.9, 0.9, 0.9), (-0.9, 0.9, 0.9),
        ),
        (
            (0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
            (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7),
        ),
    )


def _settings(
    output_directory: Path,
    stem: str,
    *,
    sequence_start: int = 0,
    sequence_count: int = 0,
) -> A1SingleObjectExportSettings:
    return A1SingleObjectExportSettings(
        export=ExportSettings(
            texture_width=64,
            texture_height=64,
            output_directory=output_directory,
            images_relative_path="images",
            bake_margin=1,
            sequence_start_frame=sequence_start,
            sequence_frame_count=sequence_count,
        ),
        prefix=stem,
        output_stem=stem,
        json_output_stem=stem,
        source_geometry_mode=A1SourceGeometryMode.EVALUATED,
        uv=UvUnwrapSettings(layer_name="SpineBakeUV"),
        bake_execution=BakeExecutionSettings(
            samples=2,
            texture_export_mode=A1TextureExportMode.CAMERA_PROJECTION,
        ),
    )


def _read_image(path: Path) -> tuple[tuple[int, int], tuple[float, ...]]:
    image = bpy.data.images.load(str(path), check_existing=False)
    try:
        size = tuple(int(value) for value in image.size[:2])
        return size, tuple(float(value) for value in image.pixels[:])
    finally:
        bpy.data.images.remove(image)


def _read_pixels(path: Path) -> tuple[float, ...]:
    return _read_image(path)[1]


def _visible_and_transparent_counts(pixels: tuple[float, ...]) -> tuple[int, int]:
    visible = transparent = 0
    for offset in range(0, len(pixels), 4):
        alpha = pixels[offset + 3]
        if alpha > 0.08:
            visible += 1
        elif alpha < 0.01:
            transparent += 1
    return visible, transparent


def _scene_render_fingerprint() -> tuple[object, ...]:
    scene = bpy.context.scene
    return (
        scene.render.engine,
        int(scene.render.resolution_x),
        int(scene.render.resolution_y),
        int(scene.render.resolution_percentage),
        str(scene.render.filepath),
        bool(scene.render.film_transparent),
        bool(scene.render.use_file_extension),
        str(scene.render.image_settings.file_format),
        str(scene.render.image_settings.color_mode),
        str(scene.render.image_settings.color_depth),
        int(scene.cycles.samples),
        int(scene.frame_current),
        tuple(
            sorted(
                (obj.name, bool(obj.hide_render), bool(getattr(obj, "visible_camera", True)))
                for obj in scene.objects
            )
        ),
    )


def _prepare_scene_with_sentinel() -> object:
    _clear_scene()
    _purge_orphan_scene_data()
    _configure_scene()
    _create_camera()
    _create_area_light()
    sentinel = _create_sentinel()
    sentinel.location = (8.0, 0.0, 0.0)
    sentinel.hide_render = False
    if hasattr(sentinel, "visible_camera"):
        sentinel.visible_camera = True
    _activate_only(sentinel)
    return sentinel


def _projection_attachment(document: dict) -> dict:
    _assert(len(document["slots"]) == 1, "projection should produce one mesh slot")
    slot_name = document["slots"][0]["name"]
    return document["skins"][0]["attachments"][slot_name][slot_name]


def _assert_cropped_attachment(attachment: dict, image_size: tuple[int, int]) -> None:
    hull = int(attachment["hull"])
    _assert(attachment["type"] == "mesh", "projection attachment is not mesh")
    _assert(hull >= 3, f"projection hull is degenerate: {hull}")
    _assert(len(attachment["uvs"]) == hull * 2, "UV count does not match hull")
    _assert(
        len(attachment["triangles"]) == (hull - 2) * 3,
        "triangle count does not match convex fan",
    )
    _assert(
        float(attachment["width"]) == float(image_size[0]),
        "attachment width does not match cropped image",
    )
    _assert(
        float(attachment["height"]) == float(image_size[1]),
        "attachment height does not match cropped image",
    )
    _assert(all(0.0 <= float(value) <= 1.0 for value in attachment["uvs"]), "UV outside 0..1")


def test_production_fresnel_projection_exports_union_crop_and_screen_hull() -> None:
    sentinel = _prepare_scene_with_sentinel()
    with tempfile.TemporaryDirectory(prefix="spine2d-b4-fresnel-") as directory:
        output_directory = Path(directory)
        source = _create_quad("FresnelSource")
        material = _create_layer_weight_material("FresnelMaterial")
        source.data.materials.append(material)
        _activate_only(sentinel)
        source.select_set(False)

        context_before = _capture_context()
        render_before = _scene_render_fingerprint()
        material_before = _material_fingerprint(material)
        prepared = prepare_a1_object(source, _settings(output_directory, "FresnelProjection"))
        _assert(isinstance(prepared.bake_plan, CameraProjectionPlan), "camera graph missed B4")

        result = export_a1_single_object(source, _settings(output_directory, "FresnelProjection"))
        _assert(result.success, f"B4 Fresnel export failed: {result.issues}")
        json_path = output_directory / "FresnelProjection.json"
        png_path = output_directory / "images" / "FresnelProjection_Baked.png"
        _assert(png_path.read_bytes()[:8] == PNG_SIGNATURE, "projection PNG is invalid")
        image_size, pixels = _read_image(png_path)
        _assert(0 < image_size[0] <= 64 and 0 < image_size[1] <= 64, "invalid crop size")
        _assert(image_size != (64, 64), f"projection was not cropped: {image_size}")
        visible, transparent = _visible_and_transparent_counts(pixels)
        _assert(visible > 100, f"projection contains too few visible pixels: {visible}")
        _assert(transparent > 0, "crop padding did not preserve transparent border")

        document = json.loads(json_path.read_text(encoding="utf-8"))
        attachment = _projection_attachment(document)
        _assert_cropped_attachment(attachment, image_size)
        _assert(
            result.statistics["projection_crop_width"] == image_size[0]
            and result.statistics["projection_crop_height"] == image_size[1],
            "projection statistics do not match cropped image",
        )
        _assert(_capture_context() == context_before, "B4 export changed Blender context")
        _assert(_scene_render_fingerprint() == render_before, "B4 export changed render state")
        _assert(_material_fingerprint(material) == material_before, "B4 mutated material")
        _assert(not _temporary_datablock_names(), "B4 leaked temporary datablocks")


def test_glass_and_volume_are_rendered_by_camera_projection() -> None:
    _prepare_scene_with_sentinel()
    with tempfile.TemporaryDirectory(prefix="spine2d-b4-glass-volume-") as directory:
        output_directory = Path(directory)
        glass_obj = _create_cube("GlassSource")
        glass_obj.data.materials.append(_create_glass_material("GlassMaterial"))
        glass_result = export_a1_single_object(glass_obj, _settings(output_directory, "GlassProjection"))
        _assert(glass_result.success, f"Glass projection failed: {glass_result.issues}")
        glass_size, glass_pixels = _read_image(output_directory / "images" / "GlassProjection_Baked.png")
        glass_visible, glass_transparent = _visible_and_transparent_counts(glass_pixels)
        _assert(glass_visible > 20 and glass_transparent > 0, "invalid Glass crop")
        _assert(glass_size != (64, 64), "Glass projection was not cropped")

        glass_obj.hide_render = True
        volume_obj = _create_cube("VolumeSource")
        volume_obj.data.materials.append(_create_volume_material("VolumeMaterial"))
        volume_result = export_a1_single_object(volume_obj, _settings(output_directory, "VolumeProjection"))
        _assert(volume_result.success, f"Volume projection failed: {volume_result.issues}")
        volume_size, volume_pixels = _read_image(output_directory / "images" / "VolumeProjection_Baked.png")
        volume_visible, volume_transparent = _visible_and_transparent_counts(volume_pixels)
        _assert(volume_visible > 20 and volume_transparent > 0, "invalid Volume crop")
        _assert(volume_size != (64, 64), "Volume projection was not cropped")
        _assert(not _temporary_datablock_names(), "Glass/Volume tests leaked temporary data")


def test_camera_projection_sequence_uses_one_union_crop_and_hull() -> None:
    _prepare_scene_with_sentinel()
    with tempfile.TemporaryDirectory(prefix="spine2d-b4-sequence-") as directory:
        output_directory = Path(directory)
        source = _create_quad("ProjectionSequenceSource")
        source.data.materials.append(_create_layer_weight_material("ProjectionSequenceMaterial"))
        source.rotation_euler = (0.0, 0.0, 0.0)
        source.keyframe_insert(data_path="rotation_euler", frame=1)
        source.rotation_euler = (0.0, 1.1, 0.0)
        source.keyframe_insert(data_path="rotation_euler", frame=2)
        bpy.context.scene.frame_set(7)
        frame_before = int(bpy.context.scene.frame_current)

        result = export_a1_single_object(
            source,
            _settings(output_directory, "ProjectionSequence", sequence_start=1, sequence_count=2),
        )
        _assert(result.success, f"B4 sequence failed: {result.issues}")
        first_path = output_directory / "images" / "ProjectionSequence_Baked_0001.png"
        second_path = output_directory / "images" / "ProjectionSequence_Baked_0002.png"
        first_size, first = _read_image(first_path)
        second_size, second = _read_image(second_path)
        _assert(first_size == second_size, "sequence frames use different crop dimensions")
        _assert(first_size != (64, 64), "sequence union was not cropped")
        difference = sum(abs(left - right) for left, right in zip(first, second)) / len(first)
        _assert(difference > 0.005, f"B4 sequence frames are indistinguishable: {difference}")
        document = json.loads((output_directory / "ProjectionSequence.json").read_text("utf-8"))
        _assert_cropped_attachment(_projection_attachment(document), first_size)
        _assert(int(bpy.context.scene.frame_current) == frame_before, "timeline frame not restored")


def test_forced_render_failure_rolls_back_json_texture_and_visibility() -> None:
    sentinel = _prepare_scene_with_sentinel()
    with tempfile.TemporaryDirectory(prefix="spine2d-b4-rollback-") as directory:
        output_directory = Path(directory)
        source = _create_quad("ProjectionRollbackSource")
        material = _create_layer_weight_material("ProjectionRollbackMaterial")
        source.data.materials.append(material)
        _activate_only(sentinel)
        source.select_set(False)

        final_json = output_directory / "ProjectionRollback.json"
        final_png = output_directory / "images" / "ProjectionRollback_Baked.png"
        final_png.parent.mkdir(parents=True, exist_ok=True)
        old_json, old_png = b"previous-json", b"previous-png"
        final_json.write_bytes(old_json)
        final_png.write_bytes(old_png)
        context_before = _capture_context()
        render_before = _scene_render_fingerprint()
        material_before = _material_fingerprint(material)

        with mock.patch.object(
            render_module,
            "_call_render_operator",
            side_effect=CameraProjectionExecutionError("forced B4 render failure"),
        ):
            result = export_a1_single_object(
                source,
                _settings(output_directory, "ProjectionRollback"),
            )

        _assert(not result.success, "forced B4 failure returned success")
        _assert(final_json.read_bytes() == old_json, "B4 rollback corrupted JSON")
        _assert(final_png.read_bytes() == old_png, "B4 rollback corrupted texture")
        leftovers = tuple(
            sorted(
                str(path.relative_to(output_directory))
                for path in output_directory.rglob("*")
                if path.is_file()
            )
        )
        _assert(
            leftovers == ("ProjectionRollback.json", "images/ProjectionRollback_Baked.png"),
            f"B4 rollback left staged/backup files: {leftovers}",
        )
        _assert(_capture_context() == context_before, "failed B4 changed context")
        _assert(_scene_render_fingerprint() == render_before, "failed B4 changed render state")
        _assert(_material_fingerprint(material) == material_before, "failed B4 mutated material")
        _assert(not _temporary_datablock_names(), "failed B4 leaked temporary data")


def main() -> None:
    print(f"Blender version: {bpy.app.version_string}")
    tests = (
        test_production_fresnel_projection_exports_union_crop_and_screen_hull,
        test_glass_and_volume_are_rendered_by_camera_projection,
        test_camera_projection_sequence_uses_one_union_crop_and_hull,
        test_forced_render_failure_rolls_back_json_texture_and_visibility,
    )
    for test in tests:
        print(f"[CAMERA-PROJECTION] RUN {test.__name__}")
        test()
        print(f"[CAMERA-PROJECTION] PASS {test.__name__}")
    print(f"[CAMERA-PROJECTION] PASS {len(tests)} integration tests")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
