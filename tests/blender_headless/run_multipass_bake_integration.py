"""Real Blender 5.2 tests for semantic surface/emission multi-pass baking."""

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
    _configure_cycles_scene,
    _create_sentinel,
    _material_fingerprint,
    _temporary_datablock_names,
)


def _create_two_quad_object(name: str):
    mesh = bpy.data.meshes.new(f"{name}_Mesh")
    mesh.from_pydata(
        (
            (-2.0, -1.0, 0.0),
            (-0.2, -1.0, 0.0),
            (-0.2, 1.0, 0.0),
            (-2.0, 1.0, 0.0),
            (0.2, -1.0, 0.0),
            (2.0, -1.0, 0.0),
            (2.0, 1.0, 0.0),
            (0.2, 1.0, 0.0),
        ),
        (),
        ((0, 1, 2, 3), (4, 5, 6, 7)),
    )
    mesh.update(calc_edges=True)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def _create_principled_material(name: str, color):
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new(type="ShaderNodeOutputMaterial")
    principled = nodes.new(type="ShaderNodeBsdfPrincipled")
    principled.inputs["Base Color"].default_value = (*color, 1.0)
    principled.inputs["Roughness"].default_value = 1.0
    material.node_tree.links.new(
        principled.outputs["BSDF"],
        output.inputs["Surface"],
    )
    return material


def _create_emission_material(name: str, color):
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new(type="ShaderNodeOutputMaterial")
    emission = nodes.new(type="ShaderNodeEmission")
    emission.inputs["Color"].default_value = (*color, 1.0)
    emission.inputs["Strength"].default_value = 1.0
    material.node_tree.links.new(
        emission.outputs["Emission"],
        output.inputs["Surface"],
    )
    return material


def _create_principled_surface_emission_material(name: str):
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new(type="ShaderNodeOutputMaterial")
    principled = nodes.new(type="ShaderNodeBsdfPrincipled")
    principled.inputs["Base Color"].default_value = (0.75, 0.05, 0.05, 1.0)
    principled.inputs["Roughness"].default_value = 1.0
    principled.inputs["Emission Color"].default_value = (0.0, 0.0, 0.65, 1.0)
    principled.inputs["Emission Strength"].default_value = 1.0
    material.node_tree.links.new(
        principled.outputs["BSDF"],
        output.inputs["Surface"],
    )
    return material


def _prepare_plan(obj, output_directory: Path, stem: str):
    source_snapshot = read_source_mesh_snapshot(obj)
    target_snapshot = unwrap_snapshot_uv(
        source_snapshot,
        UvUnwrapSettings(layer_name="SpineBakeUV"),
    ).snapshot
    analysis = analyse_object_materials(
        obj,
        render_target="CYCLES",
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


def _read_png_pixels(path: Path) -> tuple[float, ...]:
    image = bpy.data.images.load(str(path), check_existing=False)
    try:
        return tuple(float(value) for value in image.pixels[:])
    finally:
        bpy.data.images.remove(image)


def _dominant_pixel_count(pixels, channel: int) -> int:
    count = 0
    for offset in range(0, len(pixels), 4):
        red, green, blue, alpha = pixels[offset : offset + 4]
        if alpha <= 0.05:
            continue
        values = (red, green, blue)
        if values[channel] > 0.2 and values[channel] > max(
            values[(channel + 1) % 3],
            values[(channel + 2) % 3],
        ) * 1.35:
            count += 1
    return count


def test_surface_and_emission_material_slots_are_composed() -> None:
    _clear_scene()
    _configure_cycles_scene()
    with tempfile.TemporaryDirectory(prefix="spine2d-multipass-slots-") as directory:
        output_directory = Path(directory)
        obj = _create_two_quad_object("SurfaceEmissionSlots")
        body = _create_principled_material("BodySurface", (0.8, 0.03, 0.02))
        glow = _create_emission_material("GlowEmission", (0.01, 0.05, 0.9))
        obj.data.materials.append(body)
        obj.data.materials.append(glow)
        obj.data.polygons[0].material_index = 0
        obj.data.polygons[1].material_index = 1
        sentinel = _create_sentinel()
        _activate_only(sentinel)
        obj.select_set(False)

        target_snapshot, analysis, plan = _prepare_plan(
            obj,
            output_directory,
            "SurfaceEmissionSlots",
        )
        _assert(plan.multipass, "surface/emission slots did not create multi-pass plan")
        _assert(
            tuple(item.strategy_id for item in plan.passes)
            == (BakeStrategyId.SURFACE_COLOR, BakeStrategyId.EMISSION),
            f"unexpected strategy order: {plan.passes}",
        )
        _assert(
            plan.composite.mode is BakeCompositeMode.ADD_RGB_MAX_ALPHA,
            "multi-pass plan has incorrect compositor",
        )
        _assert(
            analysis.slots[0].semantic_channels
            == (MaterialSemanticChannel.SURFACE_COLOR,),
            f"surface graph was misclassified: {analysis.slots[0].semantic_channels}",
        )
        _assert(
            analysis.slots[1].semantic_channels
            == (MaterialSemanticChannel.SURFACE_EMISSION,),
            f"emission graph was misclassified: {analysis.slots[1].semantic_channels}",
        )

        context_before = _capture_context()
        scene_before = _capture_scene_bake_state()
        body_before = _material_fingerprint(body)
        glow_before = _material_fingerprint(glow)
        result = execute_bake_plan(
            obj,
            target_snapshot,
            plan,
            BakeExecutionSettings(samples=1),
        )

        output_path = result.representative_artifact.output_path
        pixels = _read_png_pixels(output_path)
        _assert(_dominant_pixel_count(pixels, 0) > 20, "surface red pixels missing")
        _assert(_dominant_pixel_count(pixels, 2) > 20, "emission blue pixels missing")
        _assert(_capture_context() == context_before, "multi-pass changed context")
        _assert(_capture_scene_bake_state() == scene_before, "multi-pass changed scene")
        _assert(_material_fingerprint(body) == body_before, "surface material mutated")
        _assert(_material_fingerprint(glow) == glow_before, "emission material mutated")
        _assert(not _temporary_datablock_names(), "multi-pass leaked temporary data")


def test_one_principled_material_combines_surface_and_emission_channels() -> None:
    _clear_scene()
    _configure_cycles_scene()
    with tempfile.TemporaryDirectory(prefix="spine2d-multipass-principled-") as directory:
        output_directory = Path(directory)
        obj = _create_two_quad_object("PrincipledSurfaceEmission")
        material = _create_principled_surface_emission_material("LitGlow")
        obj.data.materials.append(material)
        for polygon in obj.data.polygons:
            polygon.material_index = 0
        sentinel = _create_sentinel()
        _activate_only(sentinel)
        obj.select_set(False)

        target_snapshot, analysis, plan = _prepare_plan(
            obj,
            output_directory,
            "PrincipledSurfaceEmission",
        )
        _assert(
            set(analysis.slots[0].semantic_channels)
            == {
                MaterialSemanticChannel.SURFACE_COLOR,
                MaterialSemanticChannel.SURFACE_EMISSION,
            },
            f"Principled channels were not separated: {analysis.slots[0].semantic_channels}",
        )
        _assert(plan.multipass, "Principled surface+emission did not use multi-pass")

        result = execute_bake_plan(
            obj,
            target_snapshot,
            plan,
            BakeExecutionSettings(samples=1),
        )
        pixels = _read_png_pixels(result.representative_artifact.output_path)
        magenta_like = 0
        for offset in range(0, len(pixels), 4):
            red, green, blue, alpha = pixels[offset : offset + 4]
            if alpha > 0.05 and red > 0.2 and blue > 0.2 and green < max(red, blue) * 0.5:
                magenta_like += 1
        _assert(magenta_like > 20, "surface and emission channels were not composed")
        _assert(not _temporary_datablock_names(), "Principled multi-pass leaked data")


def main() -> None:
    print(f"Blender version: {bpy.app.version_string}")
    tests = (
        test_surface_and_emission_material_slots_are_composed,
        test_one_principled_material_combines_surface_and_emission_channels,
    )
    for test in tests:
        print(f"[MULTIPASS] RUN {test.__name__}")
        test()
        print(f"[MULTIPASS] PASS {test.__name__}")
    print(f"[MULTIPASS] PASS {len(tests)} integration tests")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
