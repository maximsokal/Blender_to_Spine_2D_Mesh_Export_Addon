"""Sequence, nested-shader, and rollback checks for semantic alpha baking."""

from __future__ import annotations

from pathlib import Path
import statistics
import sys
import tempfile
import traceback
from unittest import mock

import bpy

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIRECTORY.parents[1]
for path in (SCRIPT_DIRECTORY, REPOSITORY_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from Blender_to_Spine2D_Mesh_Exporter.blender_adapter import (  # noqa: E402
    BakeExecutionError,
    analyse_object_materials,
    execute_bake_plan,
    read_source_mesh_snapshot,
    unwrap_snapshot_uv,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter import (  # noqa: E402
    bake_executor as public_bake_executor,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import (  # noqa: E402
    BakeExecutionSettings,
    BakeMode,
    BakeSettings,
    MaterialSemanticChannel,
    build_bake_plan,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.uv import UvUnwrapSettings  # noqa: E402
from run_alpha_bake_integration import (  # noqa: E402
    _alpha_values,
    _new_principled_material,
    _prepare_plan,
    _read_pixels,
)
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


def _prepare_sequence_plan(
    obj,
    output_directory: Path,
    stem: str,
    *,
    start_frame: int,
    frame_count: int,
):
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
            sequence_start_frame=start_frame,
            sequence_frame_count=frame_count,
            sequence_frame_digits=4,
        ),
    )
    return target_snapshot, analysis, plan


def _median_covered_alpha(path: Path) -> float:
    values = _alpha_values(_read_pixels(path), minimum=0.02)
    _assert(len(values) > 20, f"texture '{path.name}' has no covered alpha pixels")
    return float(statistics.median(values))


def _new_nested_mix_material(name: str):
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()

    output = nodes.new(type="ShaderNodeOutputMaterial")
    outer_mix = nodes.new(type="ShaderNodeMixShader")
    outer_mix.inputs[0].default_value = 0.5
    inner_mix = nodes.new(type="ShaderNodeMixShader")
    inner_mix.inputs[0].default_value = 0.5
    transparent = nodes.new(type="ShaderNodeBsdfTransparent")
    red = nodes.new(type="ShaderNodeBsdfPrincipled")
    red.inputs["Base Color"].default_value = (0.8, 0.02, 0.01, 1.0)
    red.inputs["Roughness"].default_value = 1.0
    blue = nodes.new(type="ShaderNodeBsdfPrincipled")
    blue.inputs["Base Color"].default_value = (0.01, 0.02, 0.8, 1.0)
    blue.inputs["Roughness"].default_value = 1.0

    links = material.node_tree.links
    links.new(transparent.outputs["BSDF"], inner_mix.inputs[1])
    links.new(red.outputs["BSDF"], inner_mix.inputs[2])
    links.new(inner_mix.outputs["Shader"], outer_mix.inputs[1])
    links.new(blue.outputs["BSDF"], outer_mix.inputs[2])
    links.new(outer_mix.outputs["Shader"], output.inputs["Surface"])
    return material


def test_animated_principled_alpha_sequence_changes_output_and_restores_frame() -> None:
    _clear_scene()
    with tempfile.TemporaryDirectory(prefix="spine2d-alpha-sequence-") as directory:
        obj = _create_quad("AnimatedAlpha")
        material = _new_principled_material(
            "AnimatedAlphaMaterial",
            (0.7, 0.12, 0.03),
            alpha=0.2,
        )
        obj.data.materials.append(material)
        principled = next(
            node for node in material.node_tree.nodes if node.type == "BSDF_PRINCIPLED"
        )
        alpha_socket = principled.inputs["Alpha"]
        alpha_socket.default_value = 0.2
        alpha_socket.keyframe_insert(data_path="default_value", frame=1)
        alpha_socket.default_value = 0.8
        alpha_socket.keyframe_insert(data_path="default_value", frame=3)
        for fcurve in material.node_tree.animation_data.action.fcurves:
            for keyframe in fcurve.keyframe_points:
                keyframe.interpolation = "LINEAR"

        sentinel = _create_sentinel()
        _activate_only(sentinel)
        obj.select_set(False)
        bpy.context.scene.frame_set(9)

        target, analysis, plan = _prepare_sequence_plan(
            obj,
            Path(directory),
            "AnimatedAlpha",
            start_frame=1,
            frame_count=3,
        )
        _assert(analysis.has_animated_dependencies, "animated alpha was not detected")
        _assert(
            tuple(task.timeline_frame for task in plan.frame_tasks) == (1, 2, 3),
            f"unexpected alpha frame tasks: {plan.frame_tasks}",
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
        alphas = tuple(
            _median_covered_alpha(artifact.output_path)
            for artifact in result.artifacts
        )
        _assert(
            alphas[0] < alphas[1] < alphas[2],
            f"animated alpha frames are not ordered: {alphas}",
        )
        _assert(alphas[2] - alphas[0] > 0.4, f"alpha animation was flattened: {alphas}")
        _assert(_capture_context() == context_before, "alpha sequence changed context")
        _assert(_capture_scene_bake_state() == scene_before, "alpha sequence changed scene")
        _assert(_material_fingerprint(material) == material_before, "animated source graph mutated")
        _assert(not _temporary_datablock_names(), "alpha sequence leaked temporary data")


def test_nested_mix_shader_preserves_straight_color_and_computed_opacity() -> None:
    _clear_scene()
    with tempfile.TemporaryDirectory(prefix="spine2d-alpha-nested-") as directory:
        obj = _create_quad("NestedMixAlpha")
        material = _new_nested_mix_material("NestedMixAlphaMaterial")
        obj.data.materials.append(material)
        target, analysis, plan = _prepare_plan(obj, Path(directory), "NestedMixAlpha")
        _assert(
            set(analysis.slots[0].semantic_channels)
            == {MaterialSemanticChannel.SURFACE_COLOR, MaterialSemanticChannel.ALPHA},
            f"nested mix was misclassified: {analysis.slots[0].semantic_channels}",
        )

        result = execute_bake_plan(
            obj,
            target,
            plan,
            BakeExecutionSettings(samples=1),
        )
        pixels = _read_pixels(result.representative_artifact.output_path)
        covered = [
            (
                pixels[offset],
                pixels[offset + 2],
                pixels[offset + 3],
            )
            for offset in range(0, len(pixels), 4)
            if pixels[offset + 3] > 0.5
        ]
        _assert(len(covered) > 20, "nested mix produced no covered pixels")
        median_alpha = statistics.median(item[2] for item in covered)
        _assert(abs(median_alpha - 0.75) < 0.1, f"nested opacity is wrong: {median_alpha}")
        _assert(max(item[0] for item in covered) > 0.15, "nested red color was lost")
        _assert(max(item[1] for item in covered) > 0.15, "nested blue color was lost")
        _assert(not _temporary_datablock_names(), "nested mix leaked temporary data")


def test_failure_on_alpha_pass_rolls_back_existing_png_and_restores_state() -> None:
    _clear_scene()
    with tempfile.TemporaryDirectory(prefix="spine2d-alpha-rollback-") as directory:
        obj = _create_quad("AlphaRollback")
        material = _new_principled_material(
            "AlphaRollbackMaterial",
            (0.6, 0.08, 0.02),
            alpha=0.4,
        )
        obj.data.materials.append(material)
        sentinel = _create_sentinel()
        _activate_only(sentinel)
        obj.select_set(False)

        target, _, plan = _prepare_plan(obj, Path(directory), "AlphaRollback")
        final_path = plan.representative_task.output_path
        final_path.parent.mkdir(parents=True, exist_ok=True)
        previous_bytes = b"previous-alpha-production-output"
        final_path.write_bytes(previous_bytes)
        context_before = _capture_context()
        scene_before = _capture_scene_bake_state()
        material_before = _material_fingerprint(material)

        original_call = public_bake_executor._call_bake_operator
        call_count = 0

        def fail_on_alpha_pass(bpy_module, bake_type):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise BakeExecutionError("forced alpha pass failure")
            return original_call(bpy_module, bake_type)

        with mock.patch.object(
            public_bake_executor,
            "_call_bake_operator",
            side_effect=fail_on_alpha_pass,
        ):
            try:
                execute_bake_plan(
                    obj,
                    target,
                    plan,
                    BakeExecutionSettings(samples=1),
                )
            except BakeExecutionError as exc:
                _assert("forced alpha pass failure" in str(exc), "primary alpha error hidden")
            else:
                raise AssertionError("forced alpha pass failure did not propagate")

        _assert(call_count == 2, f"failure was not triggered on alpha pass: {call_count}")
        _assert(final_path.read_bytes() == previous_bytes, "existing alpha output was corrupted")
        _assert(
            tuple(path.name for path in Path(directory).iterdir()) == (final_path.name,),
            "alpha rollback left staged or backup files",
        )
        _assert(_capture_context() == context_before, "failed alpha bake changed context")
        _assert(_capture_scene_bake_state() == scene_before, "failed alpha bake changed scene")
        _assert(_material_fingerprint(material) == material_before, "failed alpha bake mutated source")
        _assert(not _temporary_datablock_names(), "failed alpha bake leaked temporary data")


def main() -> None:
    print(f"Blender version: {bpy.app.version_string}")
    tests = (
        test_animated_principled_alpha_sequence_changes_output_and_restores_frame,
        test_nested_mix_shader_preserves_straight_color_and_computed_opacity,
        test_failure_on_alpha_pass_rolls_back_existing_png_and_restores_state,
    )
    for test in tests:
        print(f"[ALPHA-EXTENDED] RUN {test.__name__}")
        test()
        print(f"[ALPHA-EXTENDED] PASS {test.__name__}")
    print(f"[ALPHA-EXTENDED] PASS {len(tests)} integration tests")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
