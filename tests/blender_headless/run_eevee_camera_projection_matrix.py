"""Real Blender 4.4 Eevee production B4 render and export fixture."""

from __future__ import annotations

from dataclasses import replace
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

from Blender_to_Spine2D_Mesh_Exporter.blender_adapter import (  # noqa: E402
    export_a1_single_object,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import (  # noqa: E402
    BakeExecutionSettings,
)
from run_bake_integration import (  # noqa: E402
    _assert,
    _capture_context,
    _temporary_datablock_names,
)
from run_camera_projection_integration import (  # noqa: E402
    PNG_SIGNATURE,
    _create_quad,
    _prepare_scene_with_sentinel,
    _read_image,
    _scene_render_fingerprint,
    _settings,
)


def _create_eevee_camera_material(name: str):
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()

    cycles_output = nodes.new(type="ShaderNodeOutputMaterial")
    cycles_output.name = "Cycles Material Output"
    cycles_output.target = "CYCLES"
    cycles_emission = nodes.new(type="ShaderNodeEmission")
    cycles_emission.inputs["Color"].default_value = (0.02, 0.02, 0.02, 1.0)
    material.node_tree.links.new(
        cycles_emission.outputs["Emission"],
        cycles_output.inputs["Surface"],
    )

    eevee_output = nodes.new(type="ShaderNodeOutputMaterial")
    eevee_output.name = "Eevee Material Output"
    eevee_output.target = "EEVEE"
    layer_weight = nodes.new(type="ShaderNodeLayerWeight")
    ramp = nodes.new(type="ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (0.02, 0.18, 0.95, 1.0)
    ramp.color_ramp.elements[1].color = (0.95, 0.08, 0.02, 1.0)
    emission = nodes.new(type="ShaderNodeEmission")
    emission.inputs["Strength"].default_value = 1.3
    material.node_tree.links.new(layer_weight.outputs["Facing"], ramp.inputs["Fac"])
    material.node_tree.links.new(ramp.outputs["Color"], emission.inputs["Color"])
    material.node_tree.links.new(emission.outputs["Emission"], eevee_output.inputs["Surface"])
    return material


def _visible_transparent_and_colored(pixels):
    visible = transparent = colored = 0
    for offset in range(0, len(pixels), 4):
        red, green, blue, alpha = pixels[offset : offset + 4]
        if alpha > 0.08:
            visible += 1
            if max(red, green, blue) - min(red, green, blue) > 0.12:
                colored += 1
        else:
            transparent += 1
    return visible, transparent, colored


def test_eevee_b4_executes_real_render_and_finalizes_cropped_attachment() -> None:
    _prepare_scene_with_sentinel()
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    with tempfile.TemporaryDirectory(prefix="spine2d-eevee-b4-") as directory:
        output_directory = Path(directory)
        source = _create_quad("EeveeProductionSource")
        source.scale = (0.72, 0.46, 1.0)
        source.data.materials.append(
            _create_eevee_camera_material("EeveeProductionMaterial")
        )
        settings = replace(
            _settings(output_directory, "EeveeProduction"),
            bake_execution=BakeExecutionSettings(
                render_engine="BLENDER_EEVEE_NEXT",
                samples=4,
            ),
        )

        context_before = _capture_context()
        render_before = _scene_render_fingerprint()
        result = export_a1_single_object(source, settings)

        _assert(result.success, f"Eevee B4 export failed: {result.issues}")
        _assert(result.statistics["render_engine"] == "BLENDER_EEVEE_NEXT", "wrong engine")
        _assert(result.statistics["shader_render_target"] == "EEVEE", "wrong target")
        _assert(
            result.statistics["projection_crop_width"] < 64
            or result.statistics["projection_crop_height"] < 64,
            f"Eevee B4 was not cropped: {result.statistics}",
        )

        image_path = output_directory / "images" / "EeveeProduction_Baked.png"
        json_path = output_directory / "EeveeProduction.json"
        _assert(image_path.read_bytes()[:8] == PNG_SIGNATURE, "Eevee PNG is invalid")
        size, pixels = _read_image(image_path)
        visible, transparent, colored = _visible_transparent_and_colored(pixels)
        _assert(visible > 50, f"Eevee image has too little visible coverage: {visible}")
        _assert(transparent > 20, "Eevee image lost transparent background")
        _assert(colored > 30, "Eevee renderer-specific material did not produce color")

        document = json.loads(json_path.read_text(encoding="utf-8"))
        attachments = [
            attachment
            for skin in document["skins"]
            for slot_attachments in skin["attachments"].values()
            for attachment in slot_attachments.values()
            if attachment.get("type") == "mesh"
            and str(attachment.get("path", "")).endswith("EeveeProduction_Baked")
        ]
        _assert(len(attachments) == 1, f"wrong Eevee attachments: {attachments}")
        attachment = attachments[0]
        _assert(float(attachment["width"]) == size[0], "Eevee attachment width mismatch")
        _assert(float(attachment["height"]) == size[1], "Eevee attachment height mismatch")
        _assert(len(attachment["triangles"]) >= 6, "Eevee attachment is degenerate")

        _assert(_capture_context() == context_before, "Eevee B4 changed context")
        _assert(
            _scene_render_fingerprint() == render_before,
            "Eevee B4 changed render or visibility state",
        )
        _assert(not _temporary_datablock_names(), "Eevee B4 leaked temporary datablocks")


def main() -> None:
    print(f"Blender version: {bpy.app.version_string}")
    test_eevee_b4_executes_real_render_and_finalizes_cropped_attachment()
    print("[EEVEE-B4] PASS real production camera projection")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
