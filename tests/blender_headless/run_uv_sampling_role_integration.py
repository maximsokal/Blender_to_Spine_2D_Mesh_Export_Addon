"""Blender 5.2 regression for independent source-sampling and bake-target UV roles."""

from __future__ import annotations

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
    prepare_a1_object,
)
from run_bake_integration import (  # noqa: E402
    _assert,
    _clear_scene,
    _create_quad,
    _temporary_datablock_names,
)
from run_camera_projection_integration import _read_pixels, _settings  # noqa: E402


def _source_uv_image_material(name: str):
    """Build the exact UV graph used by the representative sword asset."""

    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()

    output = nodes.new(type="ShaderNodeOutputMaterial")
    principled = nodes.new(type="ShaderNodeBsdfPrincipled")
    principled.inputs["Roughness"].default_value = 1.0
    texture_coordinate = nodes.new(type="ShaderNodeTexCoord")
    mapping = nodes.new(type="ShaderNodeMapping")
    image_node = nodes.new(type="ShaderNodeTexImage")
    image_node.interpolation = "Closest"

    image = bpy.data.images.new(
        name=f"{name}_Source",
        width=2,
        height=1,
        alpha=True,
        float_buffer=True,
    )
    # Left texel is red; right texel is blue. SourceUV samples only the left texel.
    image.pixels[:] = (
        1.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        1.0,
        1.0,
    )
    image.update()
    try:
        image.colorspace_settings.name = "Non-Color"
    except Exception:
        pass
    image_node.image = image

    # The representative sword uses this exact semantic path. Texture Coordinate UV
    # must read SourceUV through active_render while Blender writes into SpineBakeUV.
    material.node_tree.links.new(
        texture_coordinate.outputs["UV"],
        mapping.inputs["Vector"],
    )
    material.node_tree.links.new(
        mapping.outputs["Vector"],
        image_node.inputs["Vector"],
    )
    material.node_tree.links.new(
        image_node.outputs["Color"],
        principled.inputs["Base Color"],
    )
    material.node_tree.links.new(principled.outputs["BSDF"], output.inputs["Surface"])
    return material, image


def _assign_constant_source_uv(obj) -> None:
    mesh = obj.data
    layers = mesh.uv_layers
    source = layers.get("SourceUV") or layers.new(name="SourceUV")
    for item in source.data:
        item.uv = (0.25, 0.5)
    layers.active = source
    for layer in layers:
        layer.active_render = layer is source


def test_source_render_uv_is_not_replaced_by_spine_bake_uv() -> None:
    _clear_scene()
    bpy.context.scene.render.engine = "CYCLES"
    with tempfile.TemporaryDirectory(prefix="spine2d-uv-sampling-role-") as directory:
        source = _create_quad("UvSamplingRole")
        _assign_constant_source_uv(source)
        material, source_image = _source_uv_image_material("UvSamplingRoleMaterial")
        source.data.materials.append(material)
        settings = _settings(Path(directory), "UvSamplingRole")

        prepared = prepare_a1_object(source, settings)
        target = prepared.bake_target_snapshot
        _assert(
            target.active_uv_layer == "SpineBakeUV",
            f"wrong bake target UV: {target.active_uv_layer}",
        )
        _assert(
            target.render_uv_layer == "SourceUV",
            f"source render UV was lost: {target.render_uv_layer}",
        )
        _assert(
            {"SourceUV", "SpineBakeUV"}.issubset(set(target.uv_layer_names)),
            f"UV layers were not preserved: {target.uv_layer_names}",
        )

        result = export_a1_single_object(source, settings)
        pixels = _read_pixels(result.image_paths[0])
        covered = [
            (
                float(pixels[offset]),
                float(pixels[offset + 1]),
                float(pixels[offset + 2]),
            )
            for offset in range(0, len(pixels), 4)
            if float(pixels[offset + 3]) > 0.5
        ]
        _assert(len(covered) > 20, "UV-role bake produced too few covered pixels")
        mean_red = sum(value[0] for value in covered) / len(covered)
        mean_blue = sum(value[2] for value in covered) / len(covered)
        _assert(
            mean_red > 0.75,
            f"source render UV did not sample the red texel: red={mean_red}",
        )
        _assert(
            mean_blue < 0.2,
            f"SpineBakeUV leaked into source texture sampling: blue={mean_blue}",
        )
        _assert(not _temporary_datablock_names(), "UV-role bake leaked temporary data")
        _assert(source_image.name in bpy.data.images, "source image was removed")


def main() -> None:
    test_source_render_uv_is_not_replaced_by_spine_bake_uv()
    print("[PASS] test_source_render_uv_is_not_replaced_by_spine_bake_uv")
    print("UV sampling role integration passed: 1 test")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
