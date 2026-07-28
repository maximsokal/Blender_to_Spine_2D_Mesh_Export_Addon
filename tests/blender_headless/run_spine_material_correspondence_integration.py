"""Blender 5.2 regression for source-material to Spine-corner correspondence."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import traceback

import bpy


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIRECTORY.parents[1]
for path in (SCRIPT_DIRECTORY, REPOSITORY_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

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
from run_spine_uv_file_space_integration import (  # noqa: E402
    _assert,
    _clear_scene,
    _create_directional_source_image,
    _inset_uv,
    _load_rgba,
    _sample_spine_uv,
)


SOURCE_NAME = "MaterialCorrespondence"
SOURCE_UV_LAYER = "UVMap"


def _create_sword_style_material(image: bpy.types.Image) -> bpy.types.Material:
    """Create the representative sword's implicit render-UV material graph.

    Texture Coordinate ``UV`` reads the mesh's ``active_render`` UV layer. The
    exporter must therefore keep ``UVMap`` as the shader-sampling role while Blender
    writes the semantic bake into the independent ``SpineBakeUV`` destination.
    """

    material = bpy.data.materials.new("MaterialCorrespondenceSwordStyle")
    material.use_nodes = True
    node_tree = material.node_tree
    _assert(node_tree is not None, "Sword-style material has no node tree")
    nodes = node_tree.nodes
    links = node_tree.links
    nodes.clear()

    output = nodes.new(type="ShaderNodeOutputMaterial")
    principled = nodes.new(type="ShaderNodeBsdfPrincipled")
    texture_coordinate = nodes.new(type="ShaderNodeTexCoord")
    mapping = nodes.new(type="ShaderNodeMapping")
    texture = nodes.new(type="ShaderNodeTexImage")

    output.name = "Material Output"
    output.target = "ALL"
    principled.name = "Principled BSDF"
    principled.inputs["Roughness"].default_value = 1.0
    texture_coordinate.name = "Texture Coordinate"
    mapping.name = "Mapping"
    texture.name = "Image Texture"
    texture.image = image
    texture.interpolation = "Closest"
    texture.extension = "CLIP"

    links.new(texture_coordinate.outputs["UV"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], texture.inputs["Vector"])
    links.new(texture.outputs["Color"], principled.inputs["Base Color"])
    links.new(principled.outputs["BSDF"], output.inputs["Surface"])
    return material


def _create_permuted_source_triangle() -> bpy.types.Object:
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

    # Geometry and source-material UV intentionally have different corner order.
    # This prevents the test from passing through a circular geometry==UV oracle.
    source_uvs = (
        (1.0, 1.0),  # geometry lower-left samples source top-right: yellow
        (0.0, 0.0),  # geometry lower-right samples source lower-left: red
        (1.0, 0.0),  # geometry upper-left samples source lower-right: green
    )
    layer = mesh.uv_layers.new(name=SOURCE_UV_LAYER)
    polygon = mesh.polygons[0]
    for corner_index, coordinate in enumerate(source_uvs):
        loop_index = polygon.loop_start + corner_index
        layer.uv[loop_index].vector = coordinate
    mesh.uv_layers.active = layer
    layer.active_render = True

    source = bpy.data.objects.new(SOURCE_NAME, mesh)
    bpy.context.scene.collection.objects.link(source)
    source_image = _create_directional_source_image()
    material = _create_sword_style_material(source_image)
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


def _expected_color_from_geometry_bone(bone: dict) -> tuple[float, float, float]:
    x = float(bone.get("x", 0.0))
    y = float(bone.get("y", 0.0))
    if x < 0.0 and y > 0.0:
        return 1.0, 1.0, 0.0  # lower-left geometry -> source UV (1, 1)
    if x > 0.0 and y > 0.0:
        return 1.0, 0.0, 0.0  # lower-right geometry -> source UV (0, 0)
    if x < 0.0 and y < 0.0:
        return 0.0, 1.0, 0.0  # upper-left geometry -> source UV (1, 0)
    raise AssertionError(f"Unexpected geometry vertex bone position: {(x, y)}")


def _assert_color_matches(
    rgba: tuple[float, float, float, float],
    expected: tuple[float, float, float],
    *,
    label: str,
) -> None:
    red, green, blue, alpha = rgba
    _assert(alpha > 0.5, f"{label} sampled transparent pixels: {rgba}")
    actual = (red, green, blue)
    for channel_index, expected_value in enumerate(expected):
        if expected_value > 0.5:
            _assert(
                actual[channel_index] > 0.4,
                f"{label} expected channel {channel_index}, got {rgba}",
            )
        else:
            _assert(
                actual[channel_index] < 0.35,
                f"{label} expected channel {channel_index} to stay low, got {rgba}",
            )


def _temporary_datablock_names() -> tuple[str, ...]:
    return tuple(
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


def test_sword_style_source_material_uv_matches_final_spine_vertices() -> None:
    _clear_scene()
    with tempfile.TemporaryDirectory(prefix="spine2d-material-correspondence-") as path:
        output_directory = Path(path)
        source = _create_permuted_source_triangle()
        _configure_scene(output_directory)
        source_state = (
            source.data.as_pointer(),
            len(source.data.vertices),
            len(source.data.polygons),
            capture_source_uv_fingerprint(source),
        )

        result = export_active_object_a1(bpy.context)

        _assert(result.success, f"Material correspondence export failed: {result.issues}")
        _assert(source.data.as_pointer() == source_state[0], "Source Mesh was replaced")
        _assert(len(source.data.vertices) == source_state[1], "Source vertices changed")
        _assert(len(source.data.polygons) == source_state[2], "Source polygons changed")
        _assert(
            capture_source_uv_fingerprint(source) == source_state[3],
            "Source UV state changed",
        )

        json_path = output_directory / f"{SOURCE_NAME}_merged.json"
        texture_path = output_directory / "images" / f"{SOURCE_NAME}_Baked.png"
        _assert(json_path.is_file(), f"JSON was not created: {json_path}")
        _assert(texture_path.is_file(), f"Texture was not created: {texture_path}")
        document = json.loads(json_path.read_text(encoding="utf-8"))
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
            expected = _expected_color_from_geometry_bone(bones[bone_name])
            sample_uv = _inset_uv(vertex_index, triangle, uvs)
            rgba = _sample_spine_uv(image_data, sample_uv)
            _assert_color_matches(
                rgba,
                expected,
                label=f"{bone_name} at Spine UV {sample_uv}",
            )

        _assert(not _temporary_datablock_names(), "Material test leaked temporary data")


def main() -> None:
    print(f"Blender version: {bpy.app.version_string}")
    _assert(bpy.app.version >= (5, 2, 0), "Blender 5.2 or newer is required")
    addon.register()
    try:
        test_sword_style_source_material_uv_matches_final_spine_vertices()
        print("[SPINE_MATERIAL_CORRESPONDENCE] PASS")
    finally:
        addon.unregister()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
