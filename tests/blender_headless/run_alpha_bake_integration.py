"""Real Blender 4.4 tests for automatic alpha and transparency baking."""

from __future__ import annotations

from pathlib import Path
import statistics
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
    analyse_object_materials,
    execute_bake_plan,
    read_source_mesh_snapshot,
    unwrap_snapshot_uv,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import (  # noqa: E402
    BakeCompositeMode,
    BakeExecutionSettings,
    BakeMode,
    BakeSettings,
    BakeStrategyId,
    MaterialPreparationMode,
    MaterialSemanticChannel,
    build_bake_plan,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.uv import UvUnwrapSettings  # noqa: E402
from run_bake_integration import (  # noqa: E402
    _activate_only,
    _assert,
    _capture_context,
    _capture_scene_bake_state,
    _clear_scene,
    _create_quad,
    _create_sentinel,
    _material_fingerprint,
    _temporary_datablock_names,
)
from run_multipass_bake_integration import _create_two_quad_object  # noqa: E402


def _new_principled_material(name: str, color, *, alpha: float = 1.0):
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new(type="ShaderNodeOutputMaterial")
    principled = nodes.new(type="ShaderNodeBsdfPrincipled")
    principled.inputs["Base Color"].default_value = (*color, 1.0)
    principled.inputs["Roughness"].default_value = 1.0
    principled.inputs["Alpha"].default_value = alpha
    material.node_tree.links.new(principled.outputs["BSDF"], output.inputs["Surface"])
    return material


def _new_image_alpha_material(name: str, *, alpha: float):
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new(type="ShaderNodeOutputMaterial")
    principled = nodes.new(type="ShaderNodeBsdfPrincipled")
    principled.inputs["Roughness"].default_value = 1.0
    image_node = nodes.new(type="ShaderNodeTexImage")
    image = bpy.data.images.new(
        name=f"{name}_Image",
        width=2,
        height=2,
        alpha=True,
        float_buffer=True,
    )
    image.generated_color = (0.05, 0.75, 0.15, alpha)
    image.pixels[:] = [0.05, 0.75, 0.15, alpha] * 4
    image.update()
    image_node.image = image
    material.node_tree.links.new(image_node.outputs["Color"], principled.inputs["Base Color"])
    material.node_tree.links.new(image_node.outputs["Alpha"], principled.inputs["Alpha"])
    material.node_tree.links.new(principled.outputs["BSDF"], output.inputs["Surface"])
    return material, image


def _new_transparent_mix_material(name: str, *, factor: float, transparent_first: bool):
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new(type="ShaderNodeOutputMaterial")
    mix = nodes.new(type="ShaderNodeMixShader")
    mix.inputs[0].default_value = factor
    transparent = nodes.new(type="ShaderNodeBsdfTransparent")
    principled = nodes.new(type="ShaderNodeBsdfPrincipled")
    principled.inputs["Base Color"].default_value = (0.75, 0.08, 0.04, 1.0)
    principled.inputs["Roughness"].default_value = 1.0

    first = transparent.outputs["BSDF"] if transparent_first else principled.outputs["BSDF"]
    second = principled.outputs["BSDF"] if transparent_first else transparent.outputs["BSDF"]
    material.node_tree.links.new(first, mix.inputs[1])
    material.node_tree.links.new(second, mix.inputs[2])
    material.node_tree.links.new(mix.outputs["Shader"], output.inputs["Surface"])
    return material


def _new_transparent_material(name: str):
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new(type="ShaderNodeOutputMaterial")
    transparent = nodes.new(type="ShaderNodeBsdfTransparent")
    material.node_tree.links.new(transparent.outputs["BSDF"], output.inputs["Surface"])
    return material


def _prepare_plan(obj, output_directory: Path, stem: str):
    source_snapshot = read_source_mesh_snapshot(obj)
    target_snapshot = unwrap_snapshot_uv(
        source_snapshot,
        UvUnwrapSettings(layer_name="SpineBakeUV"),
    ).snapshot
    analysis = analyse_object_materials(
        obj,
        source_object_id=source_snapshot.source_object_id,
    )
    plan = build_bake_plan(
        analysis,
        BakeSettings(
            width=64,
            height=64,
            output_directory=output_directory,
            output_stem=stem,
            uv_layer_name="SpineBakeUV",
            margin_pixels=1,
            diffuse_mode=BakeMode.DIFFUSE,
            procedural_mode=BakeMode.DIFFUSE,
        ),
    )
    return target_snapshot, analysis, plan


def _read_pixels(path: Path) -> tuple[float, ...]:
    image = bpy.data.images.load(str(path), check_existing=False)
    try:
        image.alpha_mode = "STRAIGHT"
        return tuple(float(value) for value in image.pixels[:])
    finally:
        bpy.data.images.remove(image)


def _alpha_values(pixels, *, minimum: float = 0.02) -> list[float]:
    return [
        float(pixels[offset + 3])
        for offset in range(0, len(pixels), 4)
        if float(pixels[offset + 3]) >= minimum
    ]


def _assert_alpha_band(pixels, expected: float, *, tolerance: float = 0.08) -> None:
    values = [
        alpha
        for alpha in _alpha_values(pixels)
        if abs(alpha - expected) <= tolerance
    ]
    _assert(
        len(values) > 20,
        f"expected alpha band {expected:.3f}, found only {len(values)} pixels",
    )


def test_principled_constant_alpha_is_composed_into_png() -> None:
    _clear_scene()
    with tempfile.TemporaryDirectory(prefix="spine2d-alpha-principled-") as directory:
        obj = _create_quad("PrincipledAlpha")
        material = _new_principled_material(
            "PrincipledAlphaMaterial",
            (0.8, 0.05, 0.02),
            alpha=0.35,
        )
        obj.data.materials.append(material)
        sentinel = _create_sentinel()
        _activate_only(sentinel)
        obj.select_set(False)

        target, analysis, plan = _prepare_plan(obj, Path(directory), "PrincipledAlpha")
        _assert(
            set(analysis.slots[0].semantic_channels)
            == {MaterialSemanticChannel.SURFACE_COLOR, MaterialSemanticChannel.ALPHA},
            f"unexpected channels: {analysis.slots[0].semantic_channels}",
        )
        _assert(
            tuple(item.strategy_id for item in plan.passes)
            == (BakeStrategyId.SURFACE_COLOR, BakeStrategyId.ALPHA),
            f"unexpected alpha plan: {plan.passes}",
        )
        _assert(
            plan.composite.mode is BakeCompositeMode.ADD_RGB_REPLACE_ALPHA,
            "alpha plan did not select replace-alpha composition",
        )

        context_before = _capture_context()
        scene_before = _capture_scene_bake_state()
        material_before = _material_fingerprint(material)
        result = execute_bake_plan(
            obj,
            target,
            plan,
            BakeExecutionSettings(samples=1),
        )
        pixels = _read_pixels(result.representative_artifact.output_path)
        _assert_alpha_band(pixels, 0.35)
        covered_red = [
            pixels[offset]
            for offset in range(0, len(pixels), 4)
            if pixels[offset + 3] > 0.2
        ]
        _assert(covered_red and max(covered_red) > 0.35, "surface RGB was lost")
        _assert(_capture_context() == context_before, "alpha bake changed context")
        _assert(_capture_scene_bake_state() == scene_before, "alpha bake changed scene")
        _assert(_material_fingerprint(material) == material_before, "source graph mutated")
        _assert(not _temporary_datablock_names(), "alpha bake leaked temporary data")


def test_image_alpha_link_is_evaluated_by_alpha_pass() -> None:
    _clear_scene()
    with tempfile.TemporaryDirectory(prefix="spine2d-alpha-image-") as directory:
        obj = _create_quad("ImageAlpha")
        material, source_image = _new_image_alpha_material("ImageAlphaMaterial", alpha=0.62)
        obj.data.materials.append(material)
        sentinel = _create_sentinel()
        _activate_only(sentinel)
        obj.select_set(False)

        target, _, plan = _prepare_plan(obj, Path(directory), "ImageAlpha")
        alpha_pass = plan.passes[-1]
        _assert(alpha_pass.strategy_id is BakeStrategyId.ALPHA, "alpha pass missing")
        _assert(
            alpha_pass.material_preparations[0].mode
            is MaterialPreparationMode.EXTRACT_ALPHA_TO_EMISSION,
            "image alpha slot was not marked for extraction",
        )
        result = execute_bake_plan(
            obj,
            target,
            plan,
            BakeExecutionSettings(samples=1),
        )
        pixels = _read_pixels(result.representative_artifact.output_path)
        _assert_alpha_band(pixels, 0.62)
        green = [
            pixels[offset + 1]
            for offset in range(0, len(pixels), 4)
            if pixels[offset + 3] > 0.4
        ]
        _assert(green and max(green) > 0.25, "image color was not preserved")
        _assert(source_image.name in bpy.data.images, "source image was removed")
        _assert(not _temporary_datablock_names(), "image alpha bake leaked data")


def test_mix_shader_order_produces_correct_opacity_for_two_material_slots() -> None:
    _clear_scene()
    with tempfile.TemporaryDirectory(prefix="spine2d-alpha-mix-") as directory:
        obj = _create_two_quad_object("MixAlphaSlots")
        low = _new_transparent_mix_material(
            "TransparentFirst",
            factor=0.25,
            transparent_first=True,
        )
        high = _new_transparent_mix_material(
            "TransparentSecond",
            factor=0.25,
            transparent_first=False,
        )
        obj.data.materials.append(low)
        obj.data.materials.append(high)
        obj.data.polygons[0].material_index = 0
        obj.data.polygons[1].material_index = 1
        sentinel = _create_sentinel()
        _activate_only(sentinel)
        obj.select_set(False)

        target, analysis, plan = _prepare_plan(obj, Path(directory), "MixAlphaSlots")
        _assert(
            all(MaterialSemanticChannel.ALPHA in slot.semantic_channels for slot in analysis.slots),
            f"Mix Shader alpha was not detected: {analysis.slots}",
        )
        before = tuple(_material_fingerprint(material) for material in (low, high))
        result = execute_bake_plan(
            obj,
            target,
            plan,
            BakeExecutionSettings(samples=1),
        )
        pixels = _read_pixels(result.representative_artifact.output_path)
        _assert_alpha_band(pixels, 0.25, tolerance=0.1)
        _assert_alpha_band(pixels, 0.75, tolerance=0.1)
        after = tuple(_material_fingerprint(material) for material in (low, high))
        _assert(after == before, "Mix Shader source links were mutated")
        _assert(not _temporary_datablock_names(), "Mix Shader alpha bake leaked data")


def test_pure_transparent_material_produces_zero_alpha_texture() -> None:
    _clear_scene()
    with tempfile.TemporaryDirectory(prefix="spine2d-alpha-transparent-") as directory:
        obj = _create_quad("PureTransparent")
        material = _new_transparent_material("PureTransparentMaterial")
        obj.data.materials.append(material)
        target, analysis, plan = _prepare_plan(obj, Path(directory), "PureTransparent")
        _assert(
            analysis.slots[0].semantic_channels == (MaterialSemanticChannel.ALPHA,),
            f"pure transparent graph was misclassified: {analysis.slots[0].semantic_channels}",
        )
        _assert(len(plan.passes) == 1, "pure transparent material needs only alpha pass")
        _assert(plan.requires_composition, "alpha-only plan bypassed compositor")

        result = execute_bake_plan(
            obj,
            target,
            plan,
            BakeExecutionSettings(samples=1),
        )
        pixels = _read_pixels(result.representative_artifact.output_path)
        _assert(max(pixels[3::4]) <= 0.01, "pure transparent output contains opacity")
        _assert(max(pixels[0::4]) <= 0.01, "alpha-only output contains red RGB")
        _assert(max(pixels[1::4]) <= 0.01, "alpha-only output contains green RGB")
        _assert(max(pixels[2::4]) <= 0.01, "alpha-only output contains blue RGB")
        _assert(not _temporary_datablock_names(), "transparent bake leaked data")


def main() -> None:
    print(f"Blender version: {bpy.app.version_string}")
    tests = (
        test_principled_constant_alpha_is_composed_into_png,
        test_image_alpha_link_is_evaluated_by_alpha_pass,
        test_mix_shader_order_produces_correct_opacity_for_two_material_slots,
        test_pure_transparent_material_produces_zero_alpha_texture,
    )
    for test in tests:
        print(f"[ALPHA] RUN {test.__name__}")
        test()
        print(f"[ALPHA] PASS {test.__name__}")
    print(f"[ALPHA] PASS {len(tests)} integration tests")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
