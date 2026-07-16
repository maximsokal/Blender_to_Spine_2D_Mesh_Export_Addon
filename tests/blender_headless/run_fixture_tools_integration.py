"""Real Blender 4.4 smoke tests for the fixture worker and image comparator."""

from __future__ import annotations

from array import array
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

from run_bake_integration import (  # noqa: E402
    PNG_SIGNATURE,
    _activate_only,
    _assert,
    _clear_scene,
    _create_emission_material,
    _create_quad,
    _temporary_datablock_names,
)
from tools.blender_a1_fixture_worker import _run as run_fixture_worker  # noqa: E402
from tools.blender_a1_image_compare import compare_image_directories  # noqa: E402


def _save_png(path: Path, pixels: tuple[float, ...]) -> None:
    image = bpy.data.images.new(
        name=f"FixtureImage:{path.name}",
        width=2,
        height=2,
        alpha=True,
        float_buffer=False,
    )
    try:
        if len(pixels) != len(image.pixels):
            raise AssertionError(
                f"Expected {len(image.pixels)} pixel values, received {len(pixels)}"
            )
        image.pixels.foreach_set(array("f", pixels))
        image.file_format = "PNG"
        image.filepath_raw = str(path)
        image.save()
    finally:
        bpy.data.images.remove(image)


def test_rewrite_fixture_worker_exports_saved_blend() -> None:
    _clear_scene()
    with tempfile.TemporaryDirectory(prefix="spine2d-fixture-worker-") as directory:
        root = Path(directory)
        blend_path = root / "source.blend"
        output_directory = root / "rewrite-exports"
        source = _create_quad("FixtureHero")
        _create_emission_material(source)
        _activate_only(source)
        bpy.ops.wm.save_as_mainfile(filepath=str(blend_path), check_existing=False)

        payload = {
            "case_id": "worker-smoke",
            "mode": "single",
            "active_object": "FixtureHero",
            "selected_objects": ["FixtureHero"],
            "connected_objects": [],
            "expected_json_name": None,
            "output_directory": str(output_directory),
            "settings": {
                "texture_size": 64,
                "images_path": "images",
                "seam_mode": "AUTO",
                "angle_limit": 30.0,
                "sequence": {"start_frame": 0, "frame_count": 0},
                "per_object_sequence": {},
                "control_icons": True,
                "preview_animation": True,
            },
        }

        report = run_fixture_worker(payload, "REWRITE")

        _assert(report["success"], f"fixture worker failed: {report}")
        _assert(report["source_unchanged"], "fixture worker changed source .blend")
        _assert(report["context_restored"], "fixture worker changed Blender context")
        _assert(report["mesh_restored"], "fixture worker changed source mesh")
        _assert(
            report["temporary_datablocks_clean"],
            f"fixture worker leaked datablocks: {report['datablock_additions']}",
        )
        json_path = output_directory / "FixtureHero_merged.json"
        png_path = output_directory / "images" / "FixtureHero_Baked.png"
        _assert(json_path.is_file(), "fixture worker JSON missing")
        _assert(png_path.read_bytes()[:8] == PNG_SIGNATURE, "fixture worker PNG invalid")
        _assert(not _temporary_datablock_names(), "fixture worker leaked temp data")


def test_blender_pixel_comparator_accepts_equal_and_rejects_changed_png() -> None:
    _clear_scene()
    with tempfile.TemporaryDirectory(prefix="spine2d-image-parity-") as directory:
        root = Path(directory)
        expected_directory = root / "expected"
        actual_directory = root / "actual"
        expected_directory.mkdir()
        actual_directory.mkdir()
        opaque = (
            1.0,
            0.0,
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
            0.0,
            1.0,
            0.5,
            0.25,
            0.5,
            0.75,
            1.0,
        )
        _save_png(expected_directory / "frame.png", opaque)
        _save_png(actual_directory / "frame.png", opaque)

        equal = compare_image_directories(
            expected_directory,
            actual_directory,
            absolute_tolerance=1e-6,
            max_differing_pixel_ratio=0.0,
            max_mean_absolute_delta=0.0,
        )
        _assert(equal["compatible"], f"equal images rejected: {equal}")
        _assert(
            equal["comparisons"][0]["byte_identical"],
            "equal Blender PNG files are not byte-identical",
        )

        changed = list(opaque)
        changed[0] = 0.5
        _save_png(actual_directory / "frame.png", tuple(changed))
        different = compare_image_directories(
            expected_directory,
            actual_directory,
            absolute_tolerance=1e-6,
            max_differing_pixel_ratio=0.0,
            max_mean_absolute_delta=0.0,
        )
        _assert(not different["compatible"], "changed image was accepted")
        stats = different["comparisons"][0]["pixel_statistics"]
        _assert(stats["differing_pixel_count"] >= 1, "pixel delta was not detected")
        _assert(stats["maximum_absolute_delta"] > 0.0, "maximum delta is zero")
        _assert(not _temporary_datablock_names(), "image comparator leaked temp data")


def main() -> None:
    print(f"Blender version: {bpy.app.version_string}")
    tests = (
        test_rewrite_fixture_worker_exports_saved_blend,
        test_blender_pixel_comparator_accepts_equal_and_rejects_changed_png,
    )
    for test in tests:
        print(f"[FIXTURE_TOOLS] RUN {test.__name__}")
        test()
        print(f"[FIXTURE_TOOLS] PASS {test.__name__}")
    print(f"[FIXTURE_TOOLS] PASS {len(tests)} integration tests")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
