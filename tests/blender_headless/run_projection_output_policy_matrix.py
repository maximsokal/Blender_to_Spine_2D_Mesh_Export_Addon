"""Real Blender 5.2 matrix for B4 SDR/straight and HDR/premultiplied output."""

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
    ProjectionAlphaRepresentation,
    ProjectionDynamicRange,
    ProjectionOutputPolicy,
    ProjectionToneMapping,
    TextureFormat,
)
from run_bake_integration import (  # noqa: E402
    _assert,
    _capture_context,
    _temporary_datablock_names,
)
from run_camera_projection_integration import (  # noqa: E402
    _create_quad,
    _prepare_scene_with_sentinel,
    _scene_render_fingerprint,
    _settings,
)


def _create_hdr_camera_material(name: str):
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new(type="ShaderNodeOutputMaterial")
    layer_weight = nodes.new(type="ShaderNodeLayerWeight")
    multiply = nodes.new(type="ShaderNodeMath")
    multiply.operation = "MULTIPLY"
    multiply.inputs[1].default_value = 8.0
    emission = nodes.new(type="ShaderNodeEmission")
    emission.inputs["Color"].default_value = (1.0, 0.32, 0.08, 1.0)
    material.node_tree.links.new(layer_weight.outputs["Facing"], multiply.inputs[0])
    material.node_tree.links.new(multiply.outputs[0], emission.inputs["Strength"])
    material.node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return material


def _read_blender_image(path: Path):
    image = None
    try:
        image = bpy.data.images.load(str(path), check_existing=False)
        width, height = (int(value) for value in image.size[:2])
        pixels = [0.0] * (width * height * 4)
        image.pixels.foreach_get(pixels)
        return (
            (width, height),
            tuple(float(value) for value in pixels),
            bool(getattr(image, "is_float", False)),
            str(getattr(image, "alpha_mode", "")),
        )
    finally:
        if image is not None:
            bpy.data.images.remove(image)


def _pixel_statistics(pixels):
    maximum_rgb = 0.0
    visible = transparent = partial = 0
    for offset in range(0, len(pixels), 4):
        red, green, blue, alpha = pixels[offset : offset + 4]
        maximum_rgb = max(maximum_rgb, red, green, blue)
        if alpha > 0.08:
            visible += 1
        else:
            transparent += 1
        if 0.02 < alpha < 0.98:
            partial += 1
    return maximum_rgb, visible, transparent, partial


def _coverage_diagnostic(
    *,
    size: tuple[int, int],
    maximum_rgb: float,
    visible: int,
    transparent: int,
    partial: int,
    alpha_mode: str,
) -> str:
    return (
        f"size={size}, visible={visible}, transparent={transparent}, "
        f"partial={partial}, max_rgb={maximum_rgb:.6f}, alpha_mode={alpha_mode!r}"
    )


def _attachment(document, stem):
    matches = [
        attachment
        for skin in document["skins"]
        for slot_attachments in skin["attachments"].values()
        for attachment in slot_attachments.values()
        if attachment.get("type") == "mesh"
        and str(attachment.get("path", "")).endswith(f"{stem}_Baked")
    ]
    _assert(len(matches) == 1, f"wrong attachment count for {stem}: {matches}")
    return matches[0]


def test_sdr_png_auto_policy_is_display_referred_and_straight() -> None:
    _prepare_scene_with_sentinel()
    with tempfile.TemporaryDirectory(prefix="spine2d-b4-sdr-") as directory:
        output_directory = Path(directory)
        source = _create_quad("SdrOutputSource")
        source.scale = (0.68, 0.5, 1.0)
        source.data.materials.append(_create_hdr_camera_material("SdrOutputMaterial"))
        base_settings = _settings(output_directory, "SdrOutput")
        settings = replace(
            base_settings,
            texture_format=TextureFormat.PNG,
            bake_execution=replace(
                base_settings.bake_execution,
                samples=4,
            ),
        )
        context_before = _capture_context()
        render_before = _scene_render_fingerprint()

        result = export_a1_single_object(source, settings)

        _assert(result.success, f"SDR B4 export failed: {result.issues}")
        image_path = output_directory / "images" / "SdrOutput_Baked.png"
        size, pixels, is_float, alpha_mode = _read_blender_image(image_path)
        maximum_rgb, visible, transparent, partial = _pixel_statistics(pixels)
        diagnostic = _coverage_diagnostic(
            size=size,
            maximum_rgb=maximum_rgb,
            visible=visible,
            transparent=transparent,
            partial=partial,
            alpha_mode=alpha_mode,
        )
        _assert(not is_float, f"PNG unexpectedly loaded as float image: {diagnostic}")
        _assert(
            maximum_rgb <= 1.0001,
            f"SDR PNG retained HDR values: {diagnostic}",
        )
        _assert(
            visible > 30 and transparent > 20,
            f"SDR PNG coverage is invalid: {diagnostic}",
        )
        _assert(
            alpha_mode in {"STRAIGHT", "CHANNEL_PACKED"},
            f"wrong PNG alpha: {diagnostic}",
        )

        document = json.loads((output_directory / "SdrOutput.json").read_text("utf-8"))
        attachment = _attachment(document, "SdrOutput")
        _assert(float(attachment["width"]) == size[0], "SDR attachment width mismatch")
        _assert(float(attachment["height"]) == size[1], "SDR attachment height mismatch")
        _assert(_capture_context() == context_before, "SDR export changed context")
        _assert(_scene_render_fingerprint() == render_before, "SDR export changed scene")
        _assert(not _temporary_datablock_names(), "SDR export leaked datablocks")


def test_openexr_auto_policy_preserves_scene_linear_hdr_and_premultiplied_alpha() -> None:
    _prepare_scene_with_sentinel()
    with tempfile.TemporaryDirectory(prefix="spine2d-b4-hdr-") as directory:
        output_directory = Path(directory)
        source = _create_quad("HdrOutputSource")
        source.scale = (0.68, 0.5, 1.0)
        source.data.materials.append(_create_hdr_camera_material("HdrOutputMaterial"))
        base_settings = _settings(output_directory, "HdrOutput")
        settings = replace(
            base_settings,
            texture_format=TextureFormat.OPEN_EXR,
            bake_execution=replace(
                base_settings.bake_execution,
                samples=4,
                projection_output_policy=ProjectionOutputPolicy(
                    dynamic_range=ProjectionDynamicRange.SCENE_LINEAR_HDR,
                    tone_mapping=ProjectionToneMapping.NONE,
                    alpha_representation=(
                        ProjectionAlphaRepresentation.PREMULTIPLIED
                    ),
                ),
            ),
        )
        context_before = _capture_context()
        render_before = _scene_render_fingerprint()

        result = export_a1_single_object(source, settings)

        _assert(result.success, f"HDR B4 export failed: {result.issues}")
        image_path = output_directory / "images" / "HdrOutput_Baked.exr"
        size, pixels, is_float, alpha_mode = _read_blender_image(image_path)
        maximum_rgb, visible, transparent, partial = _pixel_statistics(pixels)
        diagnostic = _coverage_diagnostic(
            size=size,
            maximum_rgb=maximum_rgb,
            visible=visible,
            transparent=transparent,
            partial=partial,
            alpha_mode=alpha_mode,
        )
        _assert(is_float, f"OPEN_EXR did not load as a float image: {diagnostic}")
        _assert(
            maximum_rgb > 1.2,
            f"HDR values were tone-mapped or clipped: {diagnostic}",
        )
        _assert(
            visible > 30 and transparent > 20,
            f"HDR EXR coverage is invalid: {diagnostic}",
        )
        _assert(
            alpha_mode in {"PREMUL", "PREMULTIPLIED"},
            f"wrong EXR alpha: {diagnostic}",
        )

        document = json.loads((output_directory / "HdrOutput.json").read_text("utf-8"))
        attachment = _attachment(document, "HdrOutput")
        _assert(float(attachment["width"]) == size[0], "HDR attachment width mismatch")
        _assert(float(attachment["height"]) == size[1], "HDR attachment height mismatch")
        _assert(_capture_context() == context_before, "HDR export changed context")
        _assert(_scene_render_fingerprint() == render_before, "HDR export changed scene")
        _assert(not _temporary_datablock_names(), "HDR export leaked datablocks")


def main() -> None:
    print(f"Blender version: {bpy.app.version_string}")
    tests = (
        test_sdr_png_auto_policy_is_display_referred_and_straight,
        test_openexr_auto_policy_preserves_scene_linear_hdr_and_premultiplied_alpha,
    )
    for test in tests:
        print(f"[OUTPUT-POLICY] RUN {test.__name__}")
        test()
        print(f"[OUTPUT-POLICY] PASS {test.__name__}")
    print(f"[OUTPUT-POLICY] PASS {len(tests)} scenarios")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
