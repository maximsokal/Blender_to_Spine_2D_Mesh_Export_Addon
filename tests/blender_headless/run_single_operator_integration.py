"""Real Blender 5.2 checks for the registered single-object Rewrite operator."""

from __future__ import annotations

import json
from pathlib import Path
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

import Blender_to_Spine2D_Mesh_Exporter as addon  # noqa: E402
from Blender_to_Spine2D_Mesh_Exporter import single_object_operator  # noqa: E402
from Blender_to_Spine2D_Mesh_Exporter.application import (  # noqa: E402
    resolve_a1_output_paths,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.a1_ui_export_plan import (  # noqa: E402
    build_active_ui_export_plan,
)
from run_bake_integration import (  # noqa: E402
    PNG_SIGNATURE,
    _activate_only,
    _assert,
    _capture_context,
    _capture_scene_bake_state,
    _clear_scene,
    _create_emission_material,
    _create_quad,
    _material_fingerprint,
    _temporary_datablock_names,
)


def _configure_scene(output_directory: Path) -> int:
    if not isinstance(output_directory, Path):
        raise TypeError("output_directory must be pathlib.Path")

    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 1
    scene.spine2d_texture_size = 64
    scene.spine2d_json_path = str(output_directory)
    scene.spine2d_images_path = "images"
    scene.spine2d_angle_limit = 30
    scene.spine2d_seam_maker_mode = "AUTO"
    scene.spine2d_frames_for_render = 0
    scene.spine2d_bake_frame_start = 0
    scene.spine2d_control_icons = True
    # This RNA value is retained only for historical .blend compatibility. Rewrite
    # public export deliberately ignores it and never publishes preview animation.
    scene.spine2d_export_preview_animation = True
    return int(scene.spine2d_texture_size)


def _public_output_paths(source) -> tuple[Path, Path]:
    if source is None:
        raise ValueError("source cannot be None")

    plan = build_active_ui_export_plan(bpy.context)
    _assert(plan.source_object is source, "public single UI plan changed active source")
    paths = resolve_a1_output_paths(source.name, plan.settings)
    return (
        paths.json_path,
        paths.image_directory / f"{paths.output_stem}_Baked.png",
    )


def _invoke_expected_operator_failure(operator, *, expected_message: str) -> None:
    if not callable(operator):
        raise TypeError("operator must be callable")
    if not isinstance(expected_message, str) or not expected_message:
        raise ValueError("expected_message must be a non-empty string")

    try:
        result = operator()
    except RuntimeError as exc:
        _assert(
            expected_message in str(exc),
            f"operator raised an unrelated RuntimeError: {exc}",
        )
        return

    _assert("CANCELLED" in result, f"Rewrite failure was hidden: {result}")


def test_registered_operator_uses_rewrite_backend() -> None:
    _clear_scene()
    with tempfile.TemporaryDirectory(prefix="spine2d-single-operator-") as directory:
        output_directory = Path(directory)
        source = _create_quad("SingleOperator")
        material = _create_emission_material(source)
        _activate_only(source)
        _configure_scene(output_directory)
        json_path, texture_path = _public_output_paths(source)

        context_before = _capture_context()
        scene_before = _capture_scene_bake_state()
        material_before = _material_fingerprint(material)

        result = bpy.ops.object.save_uv_as_json()

        _assert("FINISHED" in result, f"Rewrite single operator failed: {result}")
        _assert(
            json_path.is_file(),
            f"single operator did not create versioned public JSON output: {json_path}",
        )
        _assert(
            texture_path.read_bytes()[:8] == PNG_SIGNATURE,
            "single operator texture is not a valid PNG",
        )
        document = json.loads(json_path.read_text(encoding="utf-8"))
        _assert(
            document["skins"][0]["attachments"]["SingleOperator_Segment_0"]
            ["SingleOperator_Segment_0"]["path"]
            == "images/SingleOperator_Baked",
            "attachment path does not preserve the texture stem",
        )

        # The public Scene default is the TWO_AXIS_ROTATION_SCALE profile. Its visual
        # controls intentionally replace Rotation Z with a dedicated uniform-scale
        # control, so the operator integration validates that public profile rather than
        # the retired three-axis v0.23 visual order.
        _assert(
            bpy.context.scene.spine2d_rig_profile == "TWO_AXIS_ROTATION_SCALE",
            "public Scene rig profile is no longer the approved two-axis profile",
        )
        _assert(
            tuple(slot["name"] for slot in document["slots"][:4])
            == (
                "SingleOperator_rotation_X",
                "SingleOperator_rotation_Y",
                "SingleOperator_scale",
                "SingleOperator_main",
            ),
            f"Rewrite public two-axis control slot order changed: {document['slots'][:4]}",
        )

        # Preview animation is intentionally not part of the Rewrite public export
        # surface. The historical RNA property remains loadable but must be inert even
        # when a legacy .blend stores True.
        _assert(
            "preview" not in document.get("animations", {}),
            "retired preview animation was re-enabled by compatibility RNA",
        )

        _assert(_capture_context() == context_before, "single operator changed context")
        _assert(
            _capture_scene_bake_state() == scene_before,
            "single operator changed scene bake state",
        )
        _assert(
            _material_fingerprint(material) == material_before,
            "single operator mutated the source material",
        )
        _assert(not _temporary_datablock_names(), "single operator leaked temporary data")


def test_visual_options_can_be_disabled_through_scene_properties() -> None:
    _clear_scene()
    with tempfile.TemporaryDirectory(prefix="spine2d-single-options-") as directory:
        output_directory = Path(directory)
        source = _create_quad("SingleOptionsOff")
        _create_emission_material(source)
        _activate_only(source)
        _configure_scene(output_directory)
        bpy.context.scene.spine2d_control_icons = False
        bpy.context.scene.spine2d_export_preview_animation = False
        json_path, _texture_path = _public_output_paths(source)

        result = bpy.ops.object.save_uv_as_json()

        _assert("FINISHED" in result, f"option-disabled export failed: {result}")
        document = json.loads(json_path.read_text(encoding="utf-8"))
        _assert(
            tuple(slot["name"] for slot in document["slots"])
            == ("SingleOptionsOff_Segment_0",),
            f"disabled control slots remain: {document['slots']}",
        )
        attachment_slots = tuple(document["skins"][0]["attachments"])
        _assert(
            attachment_slots == ("SingleOptionsOff_Segment_0",),
            f"disabled control attachments remain: {attachment_slots}",
        )
        _assert(
            "preview" not in document.get("animations", {}),
            "disabled preview animation remains in JSON",
        )
        _assert(not _temporary_datablock_names(), "option test leaked temporary data")


def test_rewrite_failure_cancels_without_fallback_or_output() -> None:
    _clear_scene()
    with tempfile.TemporaryDirectory(prefix="spine2d-single-no-fallback-") as directory:
        output_directory = Path(directory)
        source = _create_quad("SingleNoFallback")
        _create_emission_material(source)
        _activate_only(source)
        _configure_scene(output_directory)
        expected_json, expected_png = _public_output_paths(source)

        with mock.patch.object(
            single_object_operator,
            "export_active_object_a1",
            side_effect=RuntimeError("forced single rewrite failure"),
        ):
            _invoke_expected_operator_failure(
                bpy.ops.object.save_uv_as_json,
                expected_message="forced single rewrite failure",
            )

        _assert(not expected_json.exists(), "failed Rewrite export committed JSON")
        _assert(not expected_png.exists(), "failed Rewrite export committed texture")
        _assert(
            "legacy" not in single_object_operator.__dict__,
            "single-object runtime exposes a Legacy fallback symbol",
        )


def test_single_operator_registers_without_backend_switch_rna() -> None:
    _assert(
        single_object_operator.RNA_PROPERTIES == (),
        f"unexpected single-object RNA properties: {single_object_operator.RNA_PROPERTIES}",
    )
    _assert(
        not hasattr(single_object_operator, "SINGLE_BACKEND_PROPERTY"),
        "retired single backend selector remains in Rewrite runtime",
    )
    _assert(
        not hasattr(single_object_operator, "DEFAULT_SINGLE_BACKEND"),
        "retired single backend default remains in Rewrite runtime",
    )
    _assert(
        hasattr(bpy.ops.object, "save_uv_as_json"),
        "registered Rewrite single-object operator is unavailable",
    )
    _assert(
        single_object_operator.OBJECT_OT_SaveUVAsJSON.poll(bpy.context) in {True, False},
        "single-object operator poll returned a non-bool value",
    )


def main() -> None:
    print(f"Blender version: {bpy.app.version_string}")
    addon.register()
    try:
        tests = (
            test_single_operator_registers_without_backend_switch_rna,
            test_registered_operator_uses_rewrite_backend,
            test_visual_options_can_be_disabled_through_scene_properties,
            test_rewrite_failure_cancels_without_fallback_or_output,
        )
        for test in tests:
            print(f"[SINGLE_OPERATOR] RUN {test.__name__}")
            test()
            print(f"[SINGLE_OPERATOR] PASS {test.__name__}")
        print(f"[SINGLE_OPERATOR] PASS {len(tests)} integration tests")
    finally:
        addon.unregister()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise