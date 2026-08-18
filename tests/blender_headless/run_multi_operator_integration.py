"""Real Blender 5.2 checks for the registered multi-object Rewrite UI operator."""

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
from Blender_to_Spine2D_Mesh_Exporter import ui  # noqa: E402
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.a1_ui_export_plan import (  # noqa: E402
    build_selected_ui_export_plan,
)
from Blender_to_Spine2D_Mesh_Exporter.application import A1MultiObjectMode  # noqa: E402
from run_bake_integration import (  # noqa: E402
    PNG_SIGNATURE,
    _assert,
    _clear_scene,
    _create_emission_material,
    _create_quad,
    _temporary_datablock_names,
)


def _select_pair(first, second) -> None:
    for candidate in bpy.context.scene.objects:
        candidate.select_set(False)
    first.select_set(True)
    second.select_set(True)
    bpy.context.view_layer.objects.active = first


def _configure_scene(output_directory: Path) -> int:
    if not isinstance(output_directory, Path):
        raise TypeError("output_directory must be pathlib.Path")

    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 1
    scene.spine2d_texture_size = 32
    scene.spine2d_json_path = str(output_directory)
    scene.spine2d_images_path = "images"
    scene.spine2d_angle_limit = 30
    scene.spine2d_seam_maker_mode = "AUTO"
    scene.spine2d_frames_for_render = 0
    scene.spine2d_bake_frame_start = 0
    scene.spine2d_control_icons = False
    scene.spine2d_export_preview_animation = False
    return int(scene.spine2d_texture_size)


def _analyze_ready_export() -> None:
    result = bpy.ops.object.spine2d_refresh_info()
    _assert("FINISHED" in result, f"Analyze did not finish: {result}")


def test_registered_operator_uses_public_rewrite_standalone_route() -> None:
    _clear_scene()
    with tempfile.TemporaryDirectory(prefix="spine2d-operator-rewrite-") as directory:
        output_directory = Path(directory)
        first = _create_quad("OperatorA")
        second = _create_quad("OperatorB")
        _create_emission_material(first)
        _create_emission_material(second)
        _select_pair(first, second)
        _configure_scene(output_directory)

        plan = build_selected_ui_export_plan(bpy.context)
        _assert(
            plan.settings.mode is A1MultiObjectMode.STANDALONE,
            f"public selected export is not standalone: {plan.settings.mode}",
        )
        final_json = output_directory / f"{plan.settings.output_stem}.json"
        source_texture_names = tuple(
            f"{source.settings.output_stem}_Baked.png"
            for source in plan.standalone_sources
        )

        _analyze_ready_export()
        result = bpy.ops.object.spine2d_multi_export()

        _assert("FINISHED" in result, f"Rewrite multi operator failed: {result}")
        _assert(final_json.is_file(), f"operator did not create final JSON: {final_json}")
        for texture_name in source_texture_names:
            texture_path = output_directory / "images" / texture_name
            _assert(texture_path.is_file(), f"operator texture is missing: {texture_path}")
            _assert(
                texture_path.read_bytes()[:8] == PNG_SIGNATURE,
                f"operator texture is not PNG: {texture_path}",
            )
        document = json.loads(final_json.read_text(encoding="utf-8"))
        _assert(len(document.get("slots", ())) >= 2, "multi-object JSON lost attachments")
        _assert(
            bpy.context.view_layer.objects.active is first,
            "operator changed the active object",
        )
        _assert(
            {obj.name for obj in bpy.context.selected_objects}
            == {"OperatorA", "OperatorB"},
            "operator changed selection",
        )
        _assert(not _temporary_datablock_names(), "operator leaked temporary data")


def test_persisted_connect_flags_do_not_change_public_selected_route() -> None:
    _clear_scene()
    with tempfile.TemporaryDirectory(prefix="spine2d-operator-connect-inert-") as directory:
        output_directory = Path(directory)
        first = _create_quad("ConnectInertA")
        second = _create_quad("ConnectInertB")
        _create_emission_material(first)
        _create_emission_material(second)
        _select_pair(first, second)
        _configure_scene(output_directory)
        first.spine2d_connect_settings.enabled = True
        second.spine2d_connect_settings.enabled = True

        plan = build_selected_ui_export_plan(bpy.context)

        _assert(
            plan.settings.mode is A1MultiObjectMode.STANDALONE,
            f"persisted Connect changed public mode: {plan.settings.mode}",
        )
        _assert(not plan.connected_sources, "public route retained connected sources")
        _assert(
            len(plan.standalone_sources) == 2,
            f"public route lost selected meshes: {plan.standalone_sources}",
        )


def test_rewrite_failure_cancels_without_legacy_fallback() -> None:
    _clear_scene()
    with tempfile.TemporaryDirectory(prefix="spine2d-operator-no-fallback-") as directory:
        output_directory = Path(directory)
        first = _create_quad("NoFallbackA")
        second = _create_quad("NoFallbackB")
        _create_emission_material(first)
        _create_emission_material(second)
        _select_pair(first, second)
        _configure_scene(output_directory)
        _analyze_ready_export()

        with mock.patch.object(
            ui,
            "export_selected_objects_a1",
            side_effect=RuntimeError("forced rewrite operator failure"),
        ):
            result = bpy.ops.object.spine2d_multi_export()

        _assert("CANCELLED" in result, f"Rewrite failure was hidden: {result}")
        _assert(
            "export_selected_objects" not in ui.__dict__,
            "Rewrite UI exposes a Legacy selected-object exporter",
        )
        _assert(
            not tuple(output_directory.rglob("*.json")),
            "failed Rewrite multi export committed JSON",
        )
        _assert(
            not tuple(output_directory.rglob("*.png")),
            "failed Rewrite multi export committed textures",
        )


def test_multi_operator_has_no_backend_switch_rna() -> None:
    _assert(
        not hasattr(bpy.types.Scene, "spine2d_multi_export_backend"),
        "retired multi backend Scene property is registered",
    )
    _assert(
        "DEFAULT_MULTI_BACKEND" not in ui.__dict__,
        "retired multi backend default remains in Rewrite UI",
    )
    _assert(
        hasattr(bpy.ops.object, "spine2d_multi_export"),
        "registered Rewrite multi-object operator is unavailable",
    )
    _assert(
        ui.OBJECT_OT_Spine2DMultiExport.poll(bpy.context) in {True, False},
        "multi-object operator poll returned a non-bool value",
    )


def main() -> None:
    print(f"Blender version: {bpy.app.version_string}")
    addon.register()
    try:
        tests = (
            test_multi_operator_has_no_backend_switch_rna,
            test_registered_operator_uses_public_rewrite_standalone_route,
            test_persisted_connect_flags_do_not_change_public_selected_route,
            test_rewrite_failure_cancels_without_legacy_fallback,
        )
        for test in tests:
            print(f"[OPERATOR] RUN {test.__name__}")
            test()
            print(f"[OPERATOR] PASS {test.__name__}")
        print(f"[OPERATOR] PASS {len(tests)} integration tests")
    finally:
        addon.unregister()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
