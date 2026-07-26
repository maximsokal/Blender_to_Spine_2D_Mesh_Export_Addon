"""Real Blender 5.2 regression for explicit Normal — UV Segments mode."""

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
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import (  # noqa: E402
    A1TextureExportMode,
)


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _clear_scene() -> None:
    if bpy.context.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for mesh in tuple(bpy.data.meshes):
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)
    for material in tuple(bpy.data.materials):
        if material.users == 0:
            bpy.data.materials.remove(material)


def _create_pyramid() -> bpy.types.Object:
    mesh = bpy.data.meshes.new("PyramidMesh")
    mesh.from_pydata(
        (
            (0.0, 0.0, 1.0),
            (-1.0, -1.0, 0.0),
            (1.0, -1.0, 0.0),
            (0.0, 1.0, 0.0),
        ),
        (),
        (
            (1, 2, 0),
            (2, 3, 0),
            (3, 1, 0),
            (1, 3, 2),
        ),
    )
    mesh.update(calc_edges=True)

    source = bpy.data.objects.new("Pyramid", mesh)
    bpy.context.scene.collection.objects.link(source)

    material = bpy.data.materials.new("PyramidMaterial")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    principled = nodes.get("Principled BSDF")
    output = nodes.get("Material Output")
    _assert(principled is not None, "Principled BSDF node is missing")
    _assert(output is not None, "Material Output node is missing")
    principled.inputs["Base Color"].default_value = (0.35, 0.12, 0.65, 1.0)
    principled.inputs["Roughness"].default_value = 0.5
    output.target = "ALL"
    mesh.materials.append(material)

    bpy.ops.object.select_all(action="DESELECT")
    source.select_set(True)
    bpy.context.view_layer.objects.active = source
    return source


def _configure_scene(output_directory: Path) -> None:
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.spine2d_texture_export_mode = (
        A1TextureExportMode.NORMAL_UV_SEGMENTS.value
    )
    scene.spine2d_texture_size = 64
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


def _temporary_datablocks() -> tuple[str, ...]:
    names: list[str] = []
    for collection in (
        bpy.data.objects,
        bpy.data.meshes,
        bpy.data.materials,
        bpy.data.images,
    ):
        names.extend(
            str(item.name)
            for item in collection
            if str(item.name).startswith("__Spine2D_")
            or ".spine2d-stage-v2~" in str(item.name)
        )
    return tuple(sorted(names))


def test_eevee_normal_mode_exports_four_uv_segments() -> None:
    _clear_scene()
    with tempfile.TemporaryDirectory(prefix="spine2d-normal-pyramid-") as directory:
        output_directory = Path(directory)
        source = _create_pyramid()
        _configure_scene(output_directory)

        source_mesh_pointer = source.data.as_pointer()
        source_vertex_count = len(source.data.vertices)
        source_face_count = len(source.data.polygons)
        engine_before = bpy.context.scene.render.engine

        result = export_active_object_a1(bpy.context)

        _assert(result.success, f"Normal pyramid export failed: {result.issues}")
        _assert(
            bpy.context.scene.render.engine == engine_before == "BLENDER_EEVEE",
            "Normal object bake did not restore the EEVEE Scene engine",
        )
        _assert(source.data.as_pointer() == source_mesh_pointer, "Source Mesh was replaced")
        _assert(
            len(source.data.vertices) == source_vertex_count == 4,
            "Source vertex count changed",
        )
        _assert(
            len(source.data.polygons) == source_face_count == 4,
            "Source face count changed",
        )

        json_path = output_directory / "Pyramid_merged.json"
        texture_path = output_directory / "images" / "Pyramid_Baked.png"
        _assert(json_path.is_file(), f"JSON was not created: {json_path}")
        _assert(texture_path.is_file(), f"Texture was not created: {texture_path}")

        document = json.loads(json_path.read_text(encoding="utf-8"))
        slots = tuple(slot["name"] for slot in document["slots"])
        expected_slots = tuple(f"Pyramid_Segment_{index}" for index in range(4))
        _assert(slots == expected_slots, f"Unexpected Normal slots: {slots}")

        attachments = document["skins"][0]["attachments"]
        _assert(
            tuple(attachments) == expected_slots,
            f"Normal export did not preserve four attachments: {tuple(attachments)}",
        )
        for slot_name in expected_slots:
            _assert(
                tuple(attachments[slot_name]) == (slot_name,),
                f"Unexpected attachment mapping for {slot_name}",
            )
        _assert(
            not any("CameraProjection" in name for name in slots),
            "Normal export produced camera-projection topology",
        )
        _assert(
            not _temporary_datablocks(),
            f"Normal export leaked temporary datablocks: {_temporary_datablocks()}",
        )


def main() -> None:
    print(f"Blender version: {bpy.app.version_string}")
    _assert(bpy.app.version >= (5, 2, 0), "Blender 5.2+ is required")
    addon.register()
    try:
        test_eevee_normal_mode_exports_four_uv_segments()
        print("[NORMAL_UV_PYRAMID] PASS")
    finally:
        addon.unregister()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
