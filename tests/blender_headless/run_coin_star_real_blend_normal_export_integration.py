"""Export the real BlendKit coin through the public Normal UV Segments route.

Blender must open the caller-provided ``coin_star.blend`` before this script starts. The
regression proves that a material containing Fresnel, Generated coordinates, two Glossy
BSDF nodes, and a conservatively analysed muted Add Shader is baked into the generated
Spine UV layout instead of failing at PLAN_BAKE or switching exported geometry to Camera
Projection.

The public Blender UI currently owns ``ORIGINAL`` source geometry for Normal UV Segments.
This runner intentionally uses that exact contract. Topology-changing evaluated modifiers
are covered by a separate lineage/canonicalization contract and must not be smuggled into
this material-bake regression by forcing ``EVALUATED`` here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
from time import perf_counter
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
    A1TextureExportMode,
    BakeExecutionSettings,
    BakeMode,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.uv import (  # noqa: E402
    UvUnwrapSettings,
)
from run_bake_integration import (  # noqa: E402
    PNG_SIGNATURE,
    _assert,
    _capture_scene_bake_state,
    _temporary_datablock_names,
)
from run_coin_star_real_blend_shader_capability_integration import (  # noqa: E402
    _datablock_fingerprint,
    _object_fingerprint,
    _require_loaded_blend,
    _require_source_object,
    _scene_fingerprint,
)


_TEXTURE_SIZE = 256
_MAX_EXPORT_SECONDS = 240.0
_OUTPUT_STEM = "Game_Gold_Coin_Normal"


def _parse_arguments() -> argparse.Namespace:
    arguments = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(
        description="Export the real coin through Normal UV Segments."
    )
    parser.add_argument(
        "--expected-blend",
        required=True,
        help="Exact coin_star.blend path Blender must already have loaded.",
    )
    return parser.parse_args(arguments)


def _settings(output_directory: Path) -> A1SingleObjectExportSettings:
    if not isinstance(output_directory, Path):
        raise TypeError("output_directory must be pathlib.Path")

    return A1SingleObjectExportSettings(
        export=ExportSettings(
            texture_width=_TEXTURE_SIZE,
            texture_height=_TEXTURE_SIZE,
            output_directory=output_directory,
            images_relative_path="images",
            bake_margin=4,
        ),
        prefix="Game Gold Coin",
        output_stem=_OUTPUT_STEM,
        json_output_stem=_OUTPUT_STEM,
        source_geometry_mode=A1SourceGeometryMode.ORIGINAL,
        uv=UvUnwrapSettings(layer_name="SpineBakeUV"),
        diffuse_mode=BakeMode.DIFFUSE,
        procedural_mode=BakeMode.COMBINED,
        bake_execution=BakeExecutionSettings(
            render_engine="CYCLES",
            samples=1,
            texture_export_mode=A1TextureExportMode.NORMAL_UV_SEGMENTS,
        ),
    )


def _mesh_uv_stream_count(value: object) -> int:
    if isinstance(value, dict):
        count = 1 if isinstance(value.get("uvs"), list) and value["uvs"] else 0
        return count + sum(_mesh_uv_stream_count(child) for child in value.values())
    if isinstance(value, list):
        return sum(_mesh_uv_stream_count(child) for child in value)
    return 0


def _read_visible_image_signal(path: Path) -> tuple[int, int, int, float]:
    """Load one committed PNG, measure visible non-flat RGB, and remove the datablock."""

    if not isinstance(path, Path):
        raise TypeError("path must be pathlib.Path")
    image = None
    try:
        image = bpy.data.images.load(str(path), check_existing=False)
        width, height = (int(image.size[0]), int(image.size[1]))
        pixels = tuple(float(value) for value in image.pixels[:])
        _assert(len(pixels) == width * height * 4, "loaded coin PNG has invalid RGBA size")

        visible_rgb: list[float] = []
        visible_pixels = 0
        for index in range(0, len(pixels), 4):
            alpha = pixels[index + 3]
            if alpha <= 1.0 / 255.0:
                continue
            visible_pixels += 1
            visible_rgb.extend(pixels[index : index + 3])

        _assert(visible_pixels > 0, "Normal UV coin bake contains no visible pixels")
        _assert(visible_rgb, "Normal UV coin bake contains no visible RGB samples")
        signal_range = max(visible_rgb) - min(visible_rgb)
        _assert(
            signal_range > 1.0e-3,
            f"Normal UV coin bake is visually flat or black: range={signal_range}",
        )
        return width, height, visible_pixels, signal_range
    finally:
        if image is not None:
            bpy.data.images.remove(image)


def _run(expected_blend: str) -> None:
    loaded = _require_loaded_blend(expected_blend)
    source = _require_source_object()
    _assert(
        bpy.context.scene.camera is not None,
        "real coin Normal UV camera-context bake requires an active camera",
    )

    scene_before = _scene_fingerprint()
    bake_before = _capture_scene_bake_state()
    object_before = _object_fingerprint(source)
    datablocks_before = _datablock_fingerprint()
    temporary_before = _temporary_datablock_names()

    with tempfile.TemporaryDirectory(
        prefix="spine2d-coin-real-normal-export-"
    ) as directory:
        output_directory = Path(directory)
        started = perf_counter()
        result = export_a1_single_object(
            source,
            _settings(output_directory),
            context=bpy.context,
            scene=bpy.context.scene,
        )
        elapsed = perf_counter() - started

        _assert(result.success, f"real coin Normal UV export failed: {result.issues}")
        _assert(
            elapsed <= _MAX_EXPORT_SECONDS,
            f"real coin Normal UV export exceeded {_MAX_EXPORT_SECONDS}s: {elapsed:.3f}s",
        )

        outputs = tuple(Path(path).resolve(strict=False) for path in result.output_files)
        json_files = tuple(path for path in outputs if path.suffix.lower() == ".json")
        png_files = tuple(path for path in outputs if path.suffix.lower() == ".png")
        _assert(len(outputs) == 2, f"expected one JSON and one PNG, got {outputs}")
        _assert(len(json_files) == 1, f"expected one JSON output, got {json_files}")
        _assert(len(png_files) == 1, f"expected one PNG output, got {png_files}")
        _assert(
            all(path.is_file() and path.stat().st_size > 8 for path in outputs),
            f"real coin export contains missing or empty files: {outputs}",
        )
        _assert(
            png_files[0].read_bytes().startswith(PNG_SIGNATURE),
            f"real coin texture is not a PNG: {png_files[0]}",
        )

        document = json.loads(json_files[0].read_text(encoding="utf-8"))
        uv_streams = _mesh_uv_stream_count(document)
        _assert(uv_streams > 0, "real coin Spine JSON contains no mesh UV streams")

        width, height, visible_pixels, signal_range = _read_visible_image_signal(
            png_files[0]
        )
        _assert(
            (width, height) == (_TEXTURE_SIZE, _TEXTURE_SIZE),
            f"unexpected Normal UV coin texture size: {(width, height)}",
        )

        statistics = result.statistics
        _assert(
            statistics.get("texture_export_mode")
            == A1TextureExportMode.NORMAL_UV_SEGMENTS.value,
            f"coin export changed requested texture mode: {statistics}",
        )
        _assert(
            statistics.get("source_geometry_mode")
            == A1SourceGeometryMode.ORIGINAL.value,
            f"coin gate diverged from the public Normal source route: {statistics}",
        )
        _assert(
            statistics.get("modifier_count") == 0,
            f"public Normal route unexpectedly evaluated modifiers: {statistics}",
        )
        _assert(
            statistics.get("texture_pipeline") == "OBJECT_BAKE",
            f"coin export did not use object bake: {statistics}",
        )
        _assert(
            statistics.get("bake_mode") == BakeMode.COMBINED.value,
            f"coin export did not use COMBINED bake: {statistics}",
        )
        strategy_ids = str(statistics.get("bake_strategy_ids", ""))
        _assert(
            "CAMERA_COMBINED" in strategy_ids,
            f"coin export lost CAMERA_COMBINED strategy: {statistics}",
        )
        _assert(
            statistics.get("shader_capability") == "CAMERA_RENDER_REQUIRED",
            f"coin capability changed unexpectedly: {statistics}",
        )
        _assert(
            "projection_crop_width" not in statistics,
            f"Normal UV export silently became Camera Projection: {statistics}",
        )

        print(
            "[COIN-REAL-NORMAL-EXPORT] PASS "
            f"blend={loaded} object={source.name_full!r} outputs={len(outputs)} "
            f"texture={width}x{height} visible_pixels={visible_pixels} "
            f"signal_range={signal_range:.6f} uv_streams={uv_streams} "
            f"elapsed={elapsed:.3f}s mode=NORMAL_UV_SEGMENTS geometry=ORIGINAL "
            "pipeline=OBJECT_BAKE bake=COMBINED strategy=CAMERA_COMBINED",
            flush=True,
        )

    _assert(_scene_fingerprint() == scene_before, "coin export changed Blender context")
    _assert(_capture_scene_bake_state() == bake_before, "coin export changed bake state")
    _assert(_object_fingerprint(source) == object_before, "coin export changed source data")
    _assert(
        _datablock_fingerprint() == datablocks_before,
        "coin export created or removed persistent Blender datablocks",
    )
    _assert(
        _temporary_datablock_names() == temporary_before,
        "coin export leaked temporary Blender datablocks",
    )


def main() -> None:
    arguments = _parse_arguments()
    try:
        _run(arguments.expected_blend)
    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
