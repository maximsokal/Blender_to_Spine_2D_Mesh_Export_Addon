"""Blender 4.4 regressions for Rewrite generated-material execution."""

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

from Blender_to_Spine2D_Mesh_Exporter.application import (  # noqa: E402
    A1SingleObjectExportSettings,
    A1SourceGeometryMode,
    ExportSettings,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter import (  # noqa: E402
    export_a1_single_object,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import (  # noqa: E402
    A1GeneratedMaterialPattern,
    A1MaterialSourcePolicy,
    BakeExecutionSettings,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.uv import UvUnwrapSettings  # noqa: E402
from run_bake_integration import (  # noqa: E402
    _assert,
    _clear_scene,
    _create_emission_material,
    _create_quad,
    _material_fingerprint,
    _temporary_datablock_names,
)
from run_camera_projection_integration import _read_pixels  # noqa: E402


GENERATED_ATTRIBUTE_NAME = "Spine2DGeneratedColor"


def _settings(
    output_directory: Path,
    stem: str,
    *,
    policy: A1MaterialSourcePolicy,
    color: tuple[float, float, float, float],
) -> A1SingleObjectExportSettings:
    return A1SingleObjectExportSettings(
        export=ExportSettings(
            texture_width=64,
            texture_height=64,
            output_directory=output_directory,
            images_relative_path="images",
            bake_margin=1,
        ),
        prefix=stem,
        output_stem=stem,
        json_output_stem=stem,
        source_geometry_mode=A1SourceGeometryMode.EVALUATED,
        uv=UvUnwrapSettings(layer_name="SpineBakeUV"),
        bake_execution=BakeExecutionSettings(samples=1),
        material_source_policy=policy,
        generated_material_pattern=A1GeneratedMaterialPattern.SOLID_GRAY,
        generated_gray_color=color,
    )


def _covered_rgb(path: Path) -> tuple[tuple[float, float, float], ...]:
    pixels = _read_pixels(path)
    return tuple(
        (
            float(pixels[offset]),
            float(pixels[offset + 1]),
            float(pixels[offset + 2]),
        )
        for offset in range(0, len(pixels), 4)
        if float(pixels[offset + 3]) > 0.5
    )


def _assert_green_generated_texture(path: Path) -> None:
    covered = _covered_rgb(path)
    _assert(len(covered) > 20, "generated bake has too few covered pixels")
    mean = tuple(
        sum(pixel[channel] for pixel in covered) / len(covered)
        for channel in range(3)
    )
    _assert(mean[1] > 0.55, f"generated green channel is too weak: {mean}")
    _assert(mean[1] > mean[0] * 3.0, f"source/red color leaked into bake: {mean}")
    _assert(mean[1] > mean[2] * 3.0, f"generated color became blue/gray: {mean}")


def _source_attribute_names(source: bpy.types.Object) -> tuple[str, ...]:
    return tuple(sorted(attribute.name for attribute in source.data.color_attributes))


def test_generate_if_missing_bakes_materialless_mesh_and_cleans_resources() -> None:
    _clear_scene()
    bpy.context.scene.render.engine = "CYCLES"
    with tempfile.TemporaryDirectory(prefix="spine2d-generated-missing-") as directory:
        output_directory = Path(directory)
        source = _create_quad("GeneratedMissingSource")
        attributes_before = _source_attribute_names(source)
        _assert(len(source.data.materials) == 0, "fixture unexpectedly has materials")

        result = export_a1_single_object(
            source,
            _settings(
                output_directory,
                "GeneratedMissing",
                policy=A1MaterialSourcePolicy.GENERATE_IF_MISSING,
                color=(0.0, 1.0, 0.0, 1.0),
            ),
        )

        _assert(result.success, f"generated fallback export failed: {result.issues}")
        _assert(
            result.statistics.get("generated_material_active") == 1,
            f"generated path was not recorded: {result.statistics}",
        )
        _assert_green_generated_texture(result.image_paths[0])
        _assert(len(source.data.materials) == 0, "fallback added source material slots")
        _assert(
            _source_attribute_names(source) == attributes_before,
            "fallback leaked generated color attribute onto source mesh",
        )
        _assert(
            GENERATED_ATTRIBUTE_NAME not in _source_attribute_names(source),
            "source mesh retained generated color attribute",
        )
        _assert(not _temporary_datablock_names(), "fallback leaked temporary datablocks")


def test_force_generated_ignores_source_material_without_mutating_it() -> None:
    _clear_scene()
    bpy.context.scene.render.engine = "CYCLES"
    with tempfile.TemporaryDirectory(prefix="spine2d-generated-force-") as directory:
        output_directory = Path(directory)
        source = _create_quad("GeneratedForceSource")
        material = _create_emission_material(source)
        material_before = _material_fingerprint(material)
        source_material_names = tuple(item.name for item in source.data.materials)
        attributes_before = _source_attribute_names(source)

        result = export_a1_single_object(
            source,
            _settings(
                output_directory,
                "GeneratedForce",
                policy=A1MaterialSourcePolicy.FORCE_GENERATED,
                color=(0.0, 1.0, 0.0, 1.0),
            ),
        )

        _assert(result.success, f"forced generated export failed: {result.issues}")
        _assert_green_generated_texture(result.image_paths[0])
        _assert(
            tuple(item.name for item in source.data.materials) == source_material_names,
            "force-generated path changed source material slots",
        )
        _assert(
            _material_fingerprint(material) == material_before,
            "force-generated path mutated source material graph",
        )
        _assert(
            _source_attribute_names(source) == attributes_before,
            "force-generated path changed source color attributes",
        )
        _assert(not _temporary_datablock_names(), "forced path leaked temporary datablocks")


def main() -> None:
    tests = (
        test_generate_if_missing_bakes_materialless_mesh_and_cleans_resources,
        test_force_generated_ignores_source_material_without_mutating_it,
    )
    for test in tests:
        test()
        print(f"[PASS] {test.__name__}")
    print(f"Generated material integration passed: {len(tests)} tests")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
