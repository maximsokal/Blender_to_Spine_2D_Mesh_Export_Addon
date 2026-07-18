"""Real Blender integration for depth-correct grouped connected B4 rendering."""

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

from Blender_to_Spine2D_Mesh_Exporter.application import (  # noqa: E402
    A1MultiObjectExportSettings,
    A1MultiObjectMode,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter import (  # noqa: E402
    A1MultiObjectSource,
    export_a1_multi_object,
)
from run_bake_integration import (  # noqa: E402
    _activate_only,
    _assert,
    _capture_context,
    _clear_scene,
    _create_mesh_object,
    _create_sentinel,
    _temporary_datablock_names,
)
from run_camera_projection_integration import (  # noqa: E402
    PNG_SIGNATURE,
    _aim_at,
    _configure_scene,
    _create_area_light,
    _create_camera,
    _purge_orphan_scene_data,
    _read_image,
    _scene_render_fingerprint,
    _settings,
)


def _create_colored_layer_weight_material(name: str, color):
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new(type="ShaderNodeOutputMaterial")
    layer_weight = nodes.new(type="ShaderNodeLayerWeight")
    ramp = nodes.new(type="ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (*color, 1.0)
    ramp.color_ramp.elements[1].color = (
        min(1.0, color[0] * 0.45 + 0.1),
        min(1.0, color[1] * 0.45 + 0.1),
        min(1.0, color[2] * 0.45 + 0.1),
        1.0,
    )
    emission = nodes.new(type="ShaderNodeEmission")
    emission.inputs["Strength"].default_value = 1.4
    material.node_tree.links.new(layer_weight.outputs["Facing"], ramp.inputs["Fac"])
    material.node_tree.links.new(ramp.outputs["Color"], emission.inputs["Color"])
    material.node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return material


def _create_offset_quad(name: str, *, center_x: float, z: float):
    obj = _create_mesh_object(
        name,
        (
            (-1.35, -1.0, 0.0),
            (1.35, -1.0, 0.0),
            (1.35, 1.0, 0.0),
            (-1.35, 1.0, 0.0),
        ),
        ((0, 1, 2, 3),),
    )
    obj.location = (center_x, 0.0, z)
    return obj


def _prepare_scene():
    _clear_scene()
    _purge_orphan_scene_data()
    _configure_scene()
    camera = _create_camera("GroupedProjectionCamera")
    camera.data.lens = 55.0
    _aim_at(camera, bpy.mathutils.Vector((0.0, 0.0, 0.0))) if False else None
    _create_area_light("GroupedProjectionLight", energy=350.0)
    sentinel = _create_sentinel()
    sentinel.location = (8.0, 0.0, 0.0)
    sentinel.hide_render = False
    if hasattr(sentinel, "visible_camera"):
        sentinel.visible_camera = True
    _activate_only(sentinel)
    return sentinel


def _dominant_color_counts(pixels):
    red = blue = visible = 0
    for offset in range(0, len(pixels), 4):
        r, g, b, a = pixels[offset : offset + 4]
        if a <= 0.08:
            continue
        visible += 1
        if r > b * 1.35 and r > g * 1.2:
            red += 1
        if b > r * 1.35 and b > g * 1.2:
            blue += 1
    return red, blue, visible


def test_grouped_connected_projection_exports_one_visible_depth_layer() -> None:
    sentinel = _prepare_scene()
    with tempfile.TemporaryDirectory(prefix="spine2d-b4-grouped-") as directory:
        output_directory = Path(directory)
        back = _create_offset_quad("GroupedBack", center_x=-0.55, z=-0.12)
        front = _create_offset_quad("GroupedFront", center_x=0.55, z=0.12)
        back.data.materials.append(
            _create_colored_layer_weight_material(
                "GroupedBackMaterial",
                (0.95, 0.04, 0.03),
            )
        )
        front.data.materials.append(
            _create_colored_layer_weight_material(
                "GroupedFrontMaterial",
                (0.03, 0.08, 0.95),
            )
        )
        _activate_only(sentinel)
        back.select_set(False)
        front.select_set(False)

        context_before = _capture_context()
        render_before = _scene_render_fingerprint()
        sources = (
            A1MultiObjectSource(
                source_object=back,
                component_id="back",
                animation_namespace="back",
                settings=_settings(output_directory, "GroupedBack"),
            ),
            A1MultiObjectSource(
                source_object=front,
                component_id="front",
                animation_namespace="front",
                settings=_settings(output_directory, "GroupedFront"),
            ),
        )
        result = export_a1_multi_object(
            sources,
            A1MultiObjectExportSettings(
                output_directory=output_directory,
                output_stem="GroupedDepth",
                mode=A1MultiObjectMode.CONNECTED,
                anchor_component_id="back",
            ),
        )

        _assert(result.success, f"grouped connected B4 failed: {result.issues}")
        _assert(result.statistics["grouped_b4_enabled"] == 1, "AUTO did not group B4")
        _assert(result.statistics["grouped_b4_source_count"] == 2, "wrong grouped count")

        json_path = output_directory / "GroupedDepth.json"
        grouped_png = (
            output_directory
            / "images"
            / "GroupedDepth_grouped_camera_Baked.png"
        )
        _assert(json_path.is_file(), "grouped JSON is missing")
        _assert(grouped_png.read_bytes()[:8] == PNG_SIGNATURE, "grouped PNG is invalid")
        size, pixels = _read_image(grouped_png)
        _assert(size != (64, 64), f"grouped image was not cropped: {size}")
        red, blue, visible = _dominant_color_counts(pixels)
        _assert(visible > 100, f"grouped layer has too little coverage: {visible}")
        _assert(red > 20, f"back source color is missing: {red}")
        _assert(blue > 20, f"front source color is missing: {blue}")

        document = json.loads(json_path.read_text(encoding="utf-8"))
        grouped_slots = [
            slot
            for slot in document["slots"]
            if slot.get("spine2dGroupedCamera") is True
            or slot["name"].endswith("grouped_camera_slot")
        ]
        _assert(len(grouped_slots) == 1, f"wrong grouped slots: {grouped_slots}")
        grouped_slot = grouped_slots[0]
        _assert(grouped_slot["bone"] == "root", "grouped slot is not root-bound")

        hidden_source_slots = [
            slot
            for slot in document["slots"]
            if slot["name"] != grouped_slot["name"] and slot.get("attachment")
        ]
        _assert(len(hidden_source_slots) >= 2, "source visual slots are missing")
        _assert(
            all(slot.get("color") == "ffffff00" for slot in hidden_source_slots),
            f"source visual slots are not transparent: {hidden_source_slots}",
        )

        skin = next(skin for skin in document["skins"] if skin["name"] == "default")
        attachment = skin["attachments"][grouped_slot["name"]][
            grouped_slot["attachment"]
        ]
        _assert(attachment["type"] == "mesh", "grouped attachment is not mesh")
        _assert(float(attachment["width"]) == size[0], "grouped width mismatch")
        _assert(float(attachment["height"]) == size[1], "grouped height mismatch")
        _assert(
            attachment.get("spine2dGroupedCamera") is True,
            "grouped attachment metadata is missing",
        )

        _assert(_capture_context() == context_before, "grouped B4 changed context")
        _assert(
            _scene_render_fingerprint() == render_before,
            "grouped B4 changed render/visibility state",
        )
        _assert(not _temporary_datablock_names(), "grouped B4 leaked temporary data")


def main() -> None:
    print(f"Blender version: {bpy.app.version_string}")
    tests = (test_grouped_connected_projection_exports_one_visible_depth_layer,)
    for test in tests:
        print(f"[GROUPED-CAMERA] RUN {test.__name__}")
        test()
        print(f"[GROUPED-CAMERA] PASS {test.__name__}")
    print(f"[GROUPED-CAMERA] PASS {len(tests)} integration tests")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
