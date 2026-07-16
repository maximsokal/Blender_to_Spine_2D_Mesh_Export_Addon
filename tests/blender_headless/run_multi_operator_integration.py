"""Real Blender 4.4 checks for the registered multi-export UI operator."""

from __future__ import annotations

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


def _configure_scene(output_directory: Path, backend: str) -> None:
    scene = bpy.context.scene
    scene.spine2d_texture_size = 32
    scene.spine2d_json_path = str(output_directory)
    scene.spine2d_images_path = "images"
    scene.spine2d_angle_limit = 30
    scene.spine2d_seam_maker_mode = "AUTO"
    scene.spine2d_multi_export_backend = backend


def test_registered_operator_uses_rewrite_backend() -> None:
    _clear_scene()
    with tempfile.TemporaryDirectory(prefix="spine2d-operator-rewrite-") as directory:
        output_directory = Path(directory)
        first = _create_quad("OperatorA")
        second = _create_quad("OperatorB")
        _create_emission_material(first)
        _create_emission_material(second)
        _select_pair(first, second)
        _configure_scene(output_directory, "REWRITE")

        result = bpy.ops.object.spine2d_multi_export()

        _assert("FINISHED" in result, f"rewrite operator failed: {result}")
        final_json = output_directory / "OperatorA_plus_1_objects.json"
        texture_a = output_directory / "images" / "OperatorA_Baked.png"
        texture_b = output_directory / "images" / "OperatorB_Baked.png"
        _assert(final_json.is_file(), "operator did not create final JSON")
        _assert(texture_a.read_bytes()[:8] == PNG_SIGNATURE, "OperatorA PNG invalid")
        _assert(texture_b.read_bytes()[:8] == PNG_SIGNATURE, "OperatorB PNG invalid")
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


def test_legacy_backend_is_explicit() -> None:
    _clear_scene()
    with tempfile.TemporaryDirectory(prefix="spine2d-operator-legacy-") as directory:
        output_directory = Path(directory)
        first = _create_quad("LegacyA")
        second = _create_quad("LegacyB")
        _create_emission_material(first)
        _create_emission_material(second)
        _select_pair(first, second)
        _configure_scene(output_directory, "LEGACY")
        expected = str(output_directory / "legacy-result.json")

        with mock.patch.object(
            ui,
            "export_selected_objects",
            return_value=expected,
        ) as legacy_export:
            result = bpy.ops.object.spine2d_multi_export()

        _assert("FINISHED" in result, f"legacy operator failed: {result}")
        _assert(legacy_export.call_count == 1, "legacy backend was not invoked once")
        args = legacy_export.call_args.args
        _assert(args[:2] == (32, 32), f"legacy texture size changed: {args}")
        _assert(args[2] == str(output_directory), f"legacy output path changed: {args}")


def test_rewrite_failure_does_not_fall_back_to_legacy() -> None:
    _clear_scene()
    with tempfile.TemporaryDirectory(prefix="spine2d-operator-no-fallback-") as directory:
        output_directory = Path(directory)
        first = _create_quad("NoFallbackA")
        second = _create_quad("NoFallbackB")
        _create_emission_material(first)
        _create_emission_material(second)
        _select_pair(first, second)
        _configure_scene(output_directory, "REWRITE")

        with mock.patch.object(
            ui,
            "export_selected_objects_a1",
            side_effect=RuntimeError("forced rewrite operator failure"),
        ), mock.patch.object(ui, "export_selected_objects") as legacy_export:
            result = bpy.ops.object.spine2d_multi_export()

        _assert("CANCELLED" in result, f"rewrite failure was hidden: {result}")
        _assert(
            legacy_export.call_count == 0,
            "legacy exporter was invoked as an automatic fallback",
        )


def main() -> None:
    print(f"Blender version: {bpy.app.version_string}")
    addon.register()
    try:
        tests = (
            test_registered_operator_uses_rewrite_backend,
            test_legacy_backend_is_explicit,
            test_rewrite_failure_does_not_fall_back_to_legacy,
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
