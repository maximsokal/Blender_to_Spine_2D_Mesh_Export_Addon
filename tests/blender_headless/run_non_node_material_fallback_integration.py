"""Blender 4.4 regression for legacy non-node diffuse-color materials."""

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
    BakeExecutionError,
    export_a1_single_object,
)
from run_bake_integration import (  # noqa: E402
    _assert,
    _clear_scene,
    _create_quad,
    _temporary_datablock_names,
)
from run_camera_projection_integration import _read_pixels, _settings  # noqa: E402


def test_opaque_non_node_diffuse_color_is_baked_from_copy() -> None:
    _clear_scene()
    bpy.context.scene.render.engine = "CYCLES"
    with tempfile.TemporaryDirectory(prefix="spine2d-non-node-material-") as directory:
        source = _create_quad("LegacyDiffuseSource")
        material = bpy.data.materials.new(name="LegacyDiffuseMaterial")
        material.use_nodes = False
        material.diffuse_color = (0.03, 0.82, 0.12, 1.0)
        original_color = tuple(float(value) for value in material.diffuse_color)
        source.data.materials.append(material)

        result = export_a1_single_object(
            source,
            _settings(Path(directory), "LegacyDiffuse"),
        )
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
        _assert(len(covered) > 20, "legacy diffuse bake has too few covered pixels")
        mean = tuple(sum(value[index] for value in covered) / len(covered) for index in range(3))
        _assert(mean[1] > 0.65, f"legacy green diffuse color was lost: {mean}")
        _assert(mean[1] > mean[0] * 4.0, f"legacy color became red/gray: {mean}")
        _assert(mean[1] > mean[2] * 3.0, f"legacy color became blue/gray: {mean}")
        _assert(not material.use_nodes, "source legacy material was converted to nodes")
        _assert(
            tuple(float(value) for value in material.diffuse_color) == original_color,
            "source legacy diffuse_color changed",
        )
        _assert(not _temporary_datablock_names(), "legacy fallback leaked temporary data")


def test_transparent_non_node_material_fails_explicitly() -> None:
    _clear_scene()
    bpy.context.scene.render.engine = "CYCLES"
    with tempfile.TemporaryDirectory(prefix="spine2d-non-node-alpha-") as directory:
        source = _create_quad("LegacyAlphaSource")
        material = bpy.data.materials.new(name="LegacyAlphaMaterial")
        material.use_nodes = False
        material.diffuse_color = (0.3, 0.7, 0.1, 0.35)
        source.data.materials.append(material)

        try:
            export_a1_single_object(
                source,
                _settings(Path(directory), "LegacyAlpha"),
            )
        except BakeExecutionError as exc:
            _assert(
                "enable material nodes so opacity can be analyzed" in str(exc),
                f"transparent legacy error is not actionable: {exc}",
            )
        else:
            raise AssertionError("transparent non-node material lost alpha silently")
        _assert(not material.use_nodes, "failed export mutated source legacy material")
        _assert(not _temporary_datablock_names(), "failed legacy export leaked temporary data")


def main() -> None:
    tests = (
        test_opaque_non_node_diffuse_color_is_baked_from_copy,
        test_transparent_non_node_material_fails_explicitly,
    )
    for test in tests:
        test()
        print(f"[PASS] {test.__name__}")
    print(f"Non-node material fallback integration passed: {len(tests)} tests")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
