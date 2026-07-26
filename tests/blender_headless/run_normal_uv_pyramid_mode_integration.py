"""Real Blender 5.2 regressions for Normal UV integrity and pixel round trips."""

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


FACE_COLORS = (
    (0.9, 0.03, 0.03, 1.0),
    (0.03, 0.9, 0.03, 1.0),
    (0.03, 0.03, 0.9, 1.0),
    (0.9, 0.9, 0.03, 1.0),
)


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


def _create_color_material(name: str, color) -> bpy.types.Material:
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    principled = nodes.get("Principled BSDF")
    output = nodes.get("Material Output")
    _assert(principled is not None, "Principled BSDF node is missing")
    _assert(output is not None, "Material Output node is missing")
    principled.inputs["Base Color"].default_value = color
    principled.inputs["Roughness"].default_value = 0.5
    output.target = "ALL"
    return material


def _create_valid_source_uv(mesh: bpy.types.Mesh) -> None:
    layer = mesh.uv_layers.new(name="UVMap")
    coordinates = (
        (0.0, 0.0),
        (1.0, 0.0),
        (0.5, 1.0),
    )
    for polygon in mesh.polygons:
        for corner_index in range(polygon.loop_total):
            loop_index = polygon.loop_start + corner_index
            layer.uv[loop_index].vector = coordinates[corner_index % 3]
    mesh.uv_layers.active = layer
    layer.active_render = True


def _create_pyramid(*, malformed_unused_uv: bool = False) -> bpy.types.Object:
    mesh = bpy.data.meshes.new("PyramidMesh")
    if malformed_unused_uv:
        # Historical files can retain a zero-length UV attribute created before topology.
        # Rewrite must ignore it when no material/source-boundary setting uses it.
        mesh.uv_layers.new(name="BrokenUnused")
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
    if malformed_unused_uv:
        broken = mesh.uv_layers.get("BrokenUnused")
        _assert(broken is not None, "BrokenUnused UV fixture was not retained")
        _assert(
            len(broken.uv) == 0,
            "Blender repaired the malformed UV fixture; regression setup is invalid",
        )
    _create_valid_source_uv(mesh)

    source = bpy.data.objects.new("Pyramid", mesh)
    bpy.context.scene.collection.objects.link(source)

    for index, color in enumerate(FACE_COLORS):
        mesh.materials.append(_create_color_material(f"FaceMaterial{index}", color))
    for polygon in mesh.polygons:
        polygon.material_index = polygon.index

    bpy.ops.object.select_all(action="DESELECT")
    source.select_set(True)
    bpy.context.view_layer.objects.active = source
    return source


def _configure_scene(output_directory: Path, *, seam_mode: str) -> None:
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.spine2d_texture_export_mode = (
        A1TextureExportMode.NORMAL_UV_SEGMENTS.value
    )
    scene.spine2d_texture_size = 128
    scene.spine2d_json_path = str(output_directory)
    scene.spine2d_images_path = "images"
    scene.spine2d_angle_limit = 30
    scene.spine2d_angular_mode = "SEED_CONE"
    scene.spine2d_local_angle_limit = 30.0
    scene.spine2d_seam_maker_mode = seam_mode
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


def _capture_source_state(source: bpy.types.Object):
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


def _load_exported_outputs(output_directory: Path) -> tuple[dict, Path]:
    json_path = output_directory / "Pyramid_merged.json"
    texture_path = output_directory / "images" / "Pyramid_Baked.png"
    _assert(json_path.is_file(), f"JSON was not created: {json_path}")
    _assert(texture_path.is_file(), f"Texture was not created: {texture_path}")
    return json.loads(json_path.read_text(encoding="utf-8")), texture_path


def _assert_no_camera_projection(document: dict) -> tuple[str, ...]:
    slots = tuple(slot["name"] for slot in document["slots"])
    _assert(slots, "Normal export produced no Spine slots")
    _assert(
        not any("CameraProjection" in name for name in slots),
        "Normal export produced camera-projection topology",
    )
    return slots


def _load_rgba(texture_path: Path):
    image = bpy.data.images.load(str(texture_path), check_existing=False)
    try:
        width, height = int(image.size[0]), int(image.size[1])
        pixels = [0.0] * (width * height * 4)
        image.pixels.foreach_get(pixels)
        return width, height, tuple(pixels)
    finally:
        bpy.data.images.remove(image)


def _sample_attachment_centroid(attachment: dict, image_data):
    width, height, pixels = image_data
    uvs = tuple(float(value) for value in attachment["uvs"])
    triangles = tuple(int(value) for value in attachment["triangles"])
    _assert(len(triangles) >= 3, "Attachment has no triangle")
    indices = triangles[:3]
    points = tuple((uvs[index * 2], uvs[index * 2 + 1]) for index in indices)
    u = sum(point[0] for point in points) / 3.0
    v = sum(point[1] for point in points) / 3.0
    x = min(width - 1, max(0, int(round(u * (width - 1)))))
    y = min(height - 1, max(0, int(round(v * (height - 1)))))
    offset = (y * width + x) * 4
    return tuple(pixels[offset + index] for index in range(4))


def _assert_expected_face_color(index: int, rgba) -> None:
    red, green, blue, alpha = rgba
    _assert(alpha > 0.5, f"Segment {index} samples transparent pixels: {rgba}")
    if index == 0:
        _assert(red > 0.5 and red > green + 0.25 and red > blue + 0.25, str(rgba))
    elif index == 1:
        _assert(green > 0.5 and green > red + 0.25 and green > blue + 0.25, str(rgba))
    elif index == 2:
        _assert(blue > 0.5 and blue > red + 0.25 and blue > green + 0.25, str(rgba))
    else:
        _assert(red > 0.5 and green > 0.5 and blue < 0.35, str(rgba))


def _assert_json_uvs_sample_face_colors(document: dict, texture_path: Path) -> None:
    image_data = _load_rgba(texture_path)
    attachments = document["skins"][0]["attachments"]
    for index in range(4):
        slot_name = f"Pyramid_Segment_{index}"
        attachment = attachments[slot_name][slot_name]
        rgba = _sample_attachment_centroid(attachment, image_data)
        _assert_expected_face_color(index, rgba)


def test_eevee_normal_auto_mode_exports_four_uv_segments() -> None:
    _clear_scene()
    with tempfile.TemporaryDirectory(prefix="spine2d-normal-pyramid-auto-") as directory:
        output_directory = Path(directory)
        source = _create_pyramid(malformed_unused_uv=True)
        _configure_scene(output_directory, seam_mode="AUTO")

        source_state = _capture_source_state(source)
        engine_before = bpy.context.scene.render.engine

        for attempt in range(3):
            result = export_active_object_a1(bpy.context)
            _assert(
                result.success,
                f"Normal AUTO pyramid export {attempt + 1} failed: {result.issues}",
            )
            _assert_source_state(source, source_state)

        _assert(
            bpy.context.scene.render.engine == engine_before == "BLENDER_EEVEE",
            "Normal AUTO object bake did not restore the EEVEE Scene engine",
        )

        document, texture_path = _load_exported_outputs(output_directory)
        slots = _assert_no_camera_projection(document)
        expected_slots = tuple(f"Pyramid_Segment_{index}" for index in range(4))
        _assert(slots == expected_slots, f"Unexpected Normal AUTO slots: {slots}")

        attachments = document["skins"][0]["attachments"]
        _assert(
            tuple(attachments) == expected_slots,
            f"Normal AUTO export did not preserve four attachments: {tuple(attachments)}",
        )
        for slot_name in expected_slots:
            _assert(
                tuple(attachments[slot_name]) == (slot_name,),
                f"Unexpected attachment mapping for {slot_name}",
            )
        _assert_json_uvs_sample_face_colors(document, texture_path)
        _assert(
            not _temporary_datablocks(),
            f"Normal AUTO export leaked temporary datablocks: {_temporary_datablocks()}",
        )


def test_eevee_normal_custom_mode_accepts_promoted_physical_hull_points() -> None:
    """Custom topology may place a topological interior vertex on the XY hull."""

    _clear_scene()
    with tempfile.TemporaryDirectory(prefix="spine2d-normal-pyramid-custom-") as directory:
        output_directory = Path(directory)
        source = _create_pyramid()
        _configure_scene(output_directory, seam_mode="CUSTOM")

        source_state = _capture_source_state(source)
        engine_before = bpy.context.scene.render.engine

        result = export_active_object_a1(bpy.context)

        _assert(result.success, f"Normal CUSTOM pyramid export failed: {result.issues}")
        _assert(
            bpy.context.scene.render.engine == engine_before == "BLENDER_EEVEE",
            "Normal CUSTOM object bake did not restore the EEVEE Scene engine",
        )
        _assert_source_state(source, source_state)

        document, _texture_path = _load_exported_outputs(output_directory)
        slots = _assert_no_camera_projection(document)
        _assert(
            all(name.startswith("Pyramid_Segment_") for name in slots),
            f"Unexpected Normal CUSTOM slots: {slots}",
        )
        attachments = document["skins"][0]["attachments"]
        _assert(
            tuple(attachments) == slots,
            "Normal CUSTOM attachment groups do not match their slots",
        )
        _assert(
            not _temporary_datablocks(),
            f"Normal CUSTOM export leaked temporary datablocks: {_temporary_datablocks()}",
        )


def test_missing_material_image_fails_before_cycles_bake() -> None:
    _clear_scene()
    with tempfile.TemporaryDirectory(prefix="spine2d-missing-image-") as directory:
        output_directory = Path(directory)
        source = _create_pyramid()
        _configure_scene(output_directory, seam_mode="AUTO")
        source_state = _capture_source_state(source)

        material = source.data.materials[0]
        nodes = material.node_tree.nodes
        links = material.node_tree.links
        principled = nodes.get("Principled BSDF")
        image_node = nodes.new("ShaderNodeTexImage")
        image = bpy.data.images.new("MissingSourceImage", width=8, height=8)
        image.source = "FILE"
        image.filepath_raw = str(output_directory / "does-not-exist.png")
        image_node.image = image
        links.new(image_node.outputs["Color"], principled.inputs["Base Color"])

        result = export_active_object_a1(bpy.context)

        _assert(not result.success, "Missing material image unexpectedly exported")
        messages = tuple(issue.message for issue in result.issues)
        _assert(
            any("Relink or pack" in message for message in messages),
            f"Missing-image diagnostic was not returned: {messages}",
        )
        _assert_source_state(source, source_state)
        _assert(
            not (output_directory / "Pyramid_merged.json").exists(),
            "Failed missing-image export committed JSON",
        )
        _assert(
            not _temporary_datablocks(),
            f"Missing-image failure leaked temporary datablocks: {_temporary_datablocks()}",
        )


def test_edit_mode_is_rejected_without_mutating_source_or_mode() -> None:
    _clear_scene()
    with tempfile.TemporaryDirectory(prefix="spine2d-edit-mode-") as directory:
        output_directory = Path(directory)
        source = _create_pyramid()
        _configure_scene(output_directory, seam_mode="AUTO")
        source_state = _capture_source_state(source)

        bpy.ops.object.mode_set(mode="EDIT")
        try:
            result = export_active_object_a1(bpy.context)
            _assert(not result.success, "Edit Mode unexpectedly exported")
            _assert(
                any("Finish or cancel Edit Mode" in issue.message for issue in result.issues),
                f"Edit Mode diagnostic was not returned: {result.issues}",
            )
            _assert(bpy.context.mode == "EDIT_MESH", "Exporter changed Edit Mode")
        finally:
            bpy.ops.object.mode_set(mode="OBJECT")

        _assert_source_state(source, source_state)
        _assert(
            not (output_directory / "Pyramid_merged.json").exists(),
            "Edit Mode failure committed JSON",
        )


def main() -> None:
    print(f"Blender version: {bpy.app.version_string}")
    _assert(bpy.app.version >= (5, 2, 0), "Blender 5.2+ is required")
    addon.register()
    try:
        test_eevee_normal_auto_mode_exports_four_uv_segments()
        print("[NORMAL_UV_PYRAMID_AUTO_ROUNDTRIP] PASS")
        test_eevee_normal_custom_mode_accepts_promoted_physical_hull_points()
        print("[NORMAL_UV_PYRAMID_CUSTOM] PASS")
        test_missing_material_image_fails_before_cycles_bake()
        print("[MISSING_IMAGE_PREFLIGHT] PASS")
        test_edit_mode_is_rejected_without_mutating_source_or_mode()
        print("[EDIT_MODE_CONTRACT] PASS")
        print("[NORMAL_UV_PYRAMID] PASS")
    finally:
        addon.unregister()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
