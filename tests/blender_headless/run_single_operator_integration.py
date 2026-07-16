"""Real Blender 4.4 checks for the registered single-object export operator."""

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
from Blender_to_Spine2D_Mesh_Exporter import single_object_operator  # noqa: E402
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


def _configure_scene(output_directory: Path, backend: str) -> int:
    scene = bpy.context.scene
    scene.spine2d_texture_size = 64
    scene.spine2d_json_path = str(output_directory)
    scene.spine2d_images_path = "images"
    scene.spine2d_angle_limit = 30
    scene.spine2d_seam_maker_mode = "AUTO"
    scene.spine2d_frames_for_render = 0
    scene.spine2d_bake_frame_start = 0
    scene.spine2d_single_export_backend = backend
    return int(scene.spine2d_texture_size)


def test_registered_operator_uses_rewrite_backend() -> None:
    _clear_scene()
    with tempfile.TemporaryDirectory(prefix="spine2d-single-operator-") as directory:
        output_directory = Path(directory)
        source = _create_quad("SingleOperator")
        material = _create_emission_material(source)
        _activate_only(source)
        _configure_scene(output_directory, "REWRITE")

        context_before = _capture_context()
        scene_before = _capture_scene_bake_state()
        material_before = _material_fingerprint(material)

        result = bpy.ops.object.save_uv_as_json()

        _assert("FINISHED" in result, f"rewrite single operator failed: {result}")
        json_path = output_directory / "SingleOperator_merged.json"
        texture_path = output_directory / "images" / "SingleOperator_Baked.png"
        _assert(json_path.is_file(), "single operator did not create legacy-named JSON")
        _assert(
            texture_path.read_bytes()[:8] == PNG_SIGNATURE,
            "single operator texture is not a valid PNG",
        )
        document_text = json_path.read_text(encoding="utf-8")
        _assert(
            "images/SingleOperator_Baked" in document_text,
            "attachment path does not preserve the texture stem",
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


def test_legacy_backend_is_explicit_and_preserves_size_sync() -> None:
    _clear_scene()
    with tempfile.TemporaryDirectory(prefix="spine2d-single-legacy-") as directory:
        output_directory = Path(directory)
        source = _create_quad("SingleLegacy")
        _create_emission_material(source)
        _activate_only(source)
        texture_size = _configure_scene(output_directory, "LEGACY")
        expected = str(output_directory / "SingleLegacy_merged.json")

        with mock.patch.object(
            single_object_operator.legacy_main,
            "save_uv_as_json",
            return_value=expected,
        ) as legacy_export:
            result = bpy.ops.object.save_uv_as_json()

        _assert("FINISHED" in result, f"legacy single operator failed: {result}")
        _assert(legacy_export.call_count == 1, "legacy single exporter not called once")
        args = legacy_export.call_args.args
        kwargs = legacy_export.call_args.kwargs
        _assert(args[0] is source, "legacy single exporter received wrong object")
        _assert(args[1:3] == (texture_size, texture_size), "legacy size changed")
        _assert(
            kwargs["output_dir"] == str(output_directory),
            "legacy output directory changed",
        )
        _assert(
            single_object_operator.legacy_main.TEXTURE_WIDTH == texture_size,
            "legacy main width was not synchronized",
        )
        _assert(
            single_object_operator.json_export.TEXTURE_HEIGHT == texture_size,
            "json_export height was not synchronized",
        )


def test_rewrite_failure_does_not_fall_back_to_legacy() -> None:
    _clear_scene()
    with tempfile.TemporaryDirectory(prefix="spine2d-single-no-fallback-") as directory:
        output_directory = Path(directory)
        source = _create_quad("SingleNoFallback")
        _create_emission_material(source)
        _activate_only(source)
        _configure_scene(output_directory, "REWRITE")

        with mock.patch.object(
            single_object_operator,
            "export_active_object_a1",
            side_effect=RuntimeError("forced single rewrite failure"),
        ), mock.patch.object(
            single_object_operator.legacy_main,
            "save_uv_as_json",
        ) as legacy_export:
            try:
                result = bpy.ops.object.save_uv_as_json()
            except RuntimeError as exc:
                _assert(
                    "forced single rewrite failure" in str(exc),
                    f"operator hid the primary rewrite error: {exc}",
                )
            else:
                _assert("CANCELLED" in result, f"rewrite failure was hidden: {result}")

        _assert(
            legacy_export.call_count == 0,
            "single Legacy exporter was invoked automatically",
        )


def test_single_backend_property_and_child_panel_register_cleanly() -> None:
    _assert(
        hasattr(bpy.types.Scene, single_object_operator.SINGLE_BACKEND_PROPERTY),
        "single backend Scene property is not registered",
    )
    _assert(
        bpy.context.scene.spine2d_single_export_backend == "REWRITE",
        "single backend default is not Rewrite",
    )
    _assert(
        single_object_operator.OBJECT_PT_Spine2DSingleExportBackend.poll(bpy.context)
        in {True, False},
        "single backend panel poll returned a non-bool value",
    )


def main() -> None:
    print(f"Blender version: {bpy.app.version_string}")
    addon.register()
    try:
        tests = (
            test_single_backend_property_and_child_panel_register_cleanly,
            test_registered_operator_uses_rewrite_backend,
            test_legacy_backend_is_explicit_and_preserves_size_sync,
            test_rewrite_failure_does_not_fall_back_to_legacy,
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
