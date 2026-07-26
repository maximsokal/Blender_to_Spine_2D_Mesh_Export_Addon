"""Real Blender 5.2 directional regression for Spine UV and PNG file-space."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import traceback

import bpy


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIRECTORY.parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import Blender_to_Spine2D_Mesh_Exporter as addon  # noqa: E402
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.a1_ui_bridge import (  # noqa: E402
    export_active_object_a1,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.source_uv_integrity import (  # noqa: E402
    capture_source_uv_fingerprint,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import (  # noqa: E402
    A1TextureExportMode,
)


SOURCE_NAME = "DirectionalTriangle"
SOURCE_UV_LAYER = "UVMap"
SOURCE_IMAGE_SIZE = 64


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _clear_scene() -> None:
    if bpy.context.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for collection in (
        bpy.data.meshes,
        bpy.data.materials,
        bpy.data.images,
    ):
        for item in tuple(collection):
            if item.users == 0:
                collection.remove(item)


def _quadrant_color(x: int, y: int, size: int) -> tuple[float, float, float, float]:
    lower = y < size // 2
    left = x < size // 2
    if lower and left:
        return 1.0, 0.0, 0.0, 1.0
    if lower and not left:
        return 0.0, 1.0, 0.0, 1.0
    if not lower and left:
        return 0.0, 0.0, 1.0, 1.0
    return 1.0, 1.0, 0.0, 1.0


def _create_directional_source_image() -> bpy.types.Image:
    image = bpy.data.images.new(
        "DirectionalSourceImage",
        width=SOURCE_IMAGE_SIZE,
        height=SOURCE_IMAGE_SIZE,
        alpha=True,
        float_buffer=False,
    )
    try:
        image.colorspace_settings.name = "Non-Color"
    except Exception:
        # Color-space naming differs across Blender configurations. Channel dominance,
        # rather than exact numeric values, is the invariant asserted below.
        pass

    pixels: list[float] = []
    for y in range(SOURCE_IMAGE_SIZE):
        for x in range(SOURCE_IMAGE_SIZE):
            pixels.extend(_quadrant_color(x, y, SOURCE_IMAGE_SIZE))
    image.pixels.foreach_set(pixels)
    image.update()
    return image


def _create_directional_material(image: bpy.types.Image) -> bpy.types.Material:
    material = bpy.data.materials.new("DirectionalMaterial")
    material.use_nodes = True
    node_tree = material.node_tree
    _assert(node_tree is not None, "Directional material has no node tree")
    nodes = node_tree.nodes
    links = node_tree.links
    nodes.clear()

    output = nodes.new(type="ShaderNodeOutputMaterial")
    emission = nodes.new(type="ShaderNodeEmission")
    uv_map = nodes.new(type="ShaderNodeUVMap")
    texture = nodes.new(type="ShaderNodeTexImage")

    output.name = "Material Output"
    output.target = "ALL"
    emission.name = "Directional Emission"
    uv_map.name = "Directional UVMap"
    uv_map.uv_map = SOURCE_UV_LAYER
    texture.name = "Directional Texture"
    texture.image = image
    texture.interpolation = "Closest"
    texture.extension = "CLIP"

    links.new(uv_map.outputs["UV"], texture.inputs["Vector"])
    links.new(texture.outputs["Color"], emission.inputs["Color"])
    links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return material


def _create_directional_triangle() -> bpy.types.Object:
    mesh = bpy.data.meshes.new(f"{SOURCE_NAME}Mesh")
    mesh.from_pydata(
        (
            (-1.0, -1.0, 0.0),
            (1.0, -1.0, 0.0),
            (-1.0, 1.0, 0.0),
        ),
        (),
        ((0, 1, 2),),
    )
    mesh.update(calc_edges=True)

    layer = mesh.uv_layers.new(name=SOURCE_UV_LAYER)
    source_uvs = (
        (0.0, 0.0),
        (1.0, 0.0),
        (0.0, 1.0),
    )
    polygon = mesh.polygons[0]
    for corner_index, coordinate in enumerate(source_uvs):
        loop_index = polygon.loop_start + corner_index
        layer.uv[loop_index].vector = coordinate
    mesh.uv_layers.active = layer
    layer.active_render = True

    source = bpy.data.objects.new(SOURCE_NAME, mesh)
    bpy.context.scene.collection.objects.link(source)
    image = _create_directional_source_image()
    material = _create_directional_material(image)
    mesh.materials.append(material)
    polygon.material_index = 0

    bpy.ops.object.select_all(action="DESELECT")
    source.select_set(True)
    bpy.context.view_layer.objects.active = source
    return source


def _configure_scene(output_directory: Path) -> None:
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.spine2d_texture_export_mode = A1TextureExportMode.NORMAL_UV_SEGMENTS.value
    scene.spine2d_texture_size = 256
    scene.spine2d_json_path = str(output_directory)
    scene.spine2d_images_path = "images"
    scene.spine2d_angle_limit = 30
    scene.spine2d_angular_mode = "SEED_CONE"
    scene.spine2d_local_angle_limit = 30.0
    scene.spine2d_seam_maker_mode = "AUTO"
    scene.spine2d_frames_for_render = 0
    scene.spine2d_bake_frame_start = 0
    scene.spine2d_control_icons = False
    scene.spine2d_export_preview_animation = False
    scene.spine2d_material_source_policy = "REQUIRE_SOURCE"


def _source_state(source: bpy.types.Object):
    return (
        source.data.as_pointer(),
        len(source.data.vertices),
        len(source.data.polygons),
        capture_source_uv_fingerprint(source),
    )


def _assert_source_state(source: bpy.types.Object, expected) -> None:
    mesh_pointer, vertex_count, face_count, uv_fingerprint = expected
    _assert(source.data.as_pointer() == mesh_pointer, "Source Mesh was replaced")
    _assert(len(source.data.vertices) == vertex_count, "Source vertex count changed")
    _assert(len(source.data.polygons) == face_count, "Source face count changed")
    _assert(
        capture_source_uv_fingerprint(source) == uv_fingerprint,
        "Source UV fingerprint changed",
    )


def _load_outputs(output_directory: Path) -> tuple[dict, Path]:
    json_path = output_directory / f"{SOURCE_NAME}_merged.json"
    texture_path = output_directory / "images" / f"{SOURCE_NAME}_Baked.png"
    _assert(json_path.is_file(), f"JSON was not created: {json_path}")
    _assert(texture_path.is_file(), f"Texture was not created: {texture_path}")
    return json.loads(json_path.read_text(encoding="utf-8")), texture_path


def _load_rgba(texture_path: Path):
    image = bpy.data.images.load(str(texture_path), check_existing=False)
    try:
        width, height = int(image.size[0]), int(image.size[1])
        pixels = [0.0] * (width * height * 4)
        image.pixels.foreach_get(pixels)
        return width, height, tuple(pixels)
    finally:
        bpy.data.images.remove(image)


def _sample_spine_uv(image_data, uv: tuple[float, float]):
    width, height, pixels = image_data
    u = min(1.0, max(0.0, float(uv[0])))
    v = min(1.0, max(0.0, float(uv[1])))
    loaded_v = 1.0 - v
    x = min(width - 1, max(0, int(round(u * (width - 1)))))
    y = min(height - 1, max(0, int(round(loaded_v * (height - 1)))))
    offset = (y * width + x) * 4
    return tuple(float(pixels[offset + index]) for index in range(4))


def _inset_uv(
    vertex_index: int,
    triangle: tuple[int, int, int],
    uvs: tuple[tuple[float, float], ...],
) -> tuple[float, float]:
    _assert(vertex_index in triangle, f"Vertex {vertex_index} is not in {triangle}")
    others = tuple(index for index in triangle if index != vertex_index)
    _assert(len(others) == 2, f"Triangle contains duplicate indices: {triangle}")
    primary = uvs[vertex_index]
    first = uvs[others[0]]
    second = uvs[others[1]]
    return (
        primary[0] * 0.75 + first[0] * 0.125 + second[0] * 0.125,
        primary[1] * 0.75 + first[1] * 0.125 + second[1] * 0.125,
    )


def _expected_channel_from_bone(bone: dict) -> int:
    x = float(bone.get("x", 0.0))
    y = float(bone.get("y", 0.0))
    if x < 0.0 and y > 0.0:
        return 0  # source lower-left → red
    if x > 0.0 and y > 0.0:
        return 1  # source lower-right → green
    if x < 0.0 and y < 0.0:
        return 2  # source upper-left → blue
    raise AssertionError(f"Unexpected directional vertex bone position: {(x, y)}")


def _assert_dominant_channel(rgba, channel: int, *, label: str) -> None:
    red, green, blue, alpha = rgba
    _assert(alpha > 0.5, f"{label} sampled transparent pixels: {rgba}")
    channels = (red, green, blue)
    dominant = channels[channel]
    others = tuple(value for index, value in enumerate(channels) if index != channel)
    _assert(
        dominant > 0.45 and all(dominant > value + 0.2 for value in others),
        f"{label} expected channel {channel} to dominate, got {rgba}",
    )


def test_directional_png_matches_spine_attachment_vertices() -> None:
    _clear_scene()
    with tempfile.TemporaryDirectory(prefix="spine2d-directional-uv-") as directory:
        output_directory = Path(directory)
        source = _create_directional_triangle()
        _configure_scene(output_directory)
        expected_source_state = _source_state(source)

        result = export_active_object_a1(bpy.context)

        _assert(result.success, f"Directional export failed: {result.issues}")
        _assert_source_state(source, expected_source_state)
        document, texture_path = _load_outputs(output_directory)
        image_data = _load_rgba(texture_path)

        attachments = document["skins"][0]["attachments"]
        _assert(len(attachments) == 1, f"Unexpected attachments: {tuple(attachments)}")
        slot_name = next(iter(attachments))
        attachment = attachments[slot_name][slot_name]
        flat_uvs = tuple(float(value) for value in attachment["uvs"])
        uvs = tuple(
            (flat_uvs[index], flat_uvs[index + 1])
            for index in range(0, len(flat_uvs), 2)
        )
        triangles = tuple(int(value) for value in attachment["triangles"])
        _assert(len(triangles) == 3, f"Expected one triangle, got {triangles}")
        triangle = triangles[0], triangles[1], triangles[2]

        bones = {bone["name"]: bone for bone in document["bones"]}
        for vertex_index in triangle:
            bone_name = f"{slot_name}_vertex_{vertex_index}"
            _assert(bone_name in bones, f"Missing attachment vertex bone {bone_name}")
            expected_channel = _expected_channel_from_bone(bones[bone_name])
            sample_uv = _inset_uv(vertex_index, triangle, uvs)
            rgba = _sample_spine_uv(image_data, sample_uv)
            _assert_dominant_channel(
                rgba,
                expected_channel,
                label=f"{bone_name} at Spine UV {sample_uv}",
            )

        temporary = tuple(
            item.name
            for collection in (
                bpy.data.objects,
                bpy.data.meshes,
                bpy.data.materials,
                bpy.data.images,
            )
            for item in collection
            if str(item.name).startswith("__Spine2D_")
        )
        _assert(not temporary, f"Directional export leaked datablocks: {temporary}")


def main() -> None:
    print(f"Blender version: {bpy.app.version_string}")
    _assert(bpy.app.version >= (5, 2, 0), "Blender 5.2+ is required")
    addon.register()
    try:
        test_directional_png_matches_spine_attachment_vertices()
        print("[SPINE_UV_FILE_SPACE_DIRECTIONAL] PASS")
    finally:
        addon.unregister()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
