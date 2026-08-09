"""Export a focused grenade camera-context surface-color bake for visual review."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys
import traceback

import bpy


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIRECTORY.parents[1]
for path in (SCRIPT_DIRECTORY, REPOSITORY_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from Blender_to_Spine2D_Mesh_Exporter.application import (  # noqa: E402
    A1MultiObjectExportSettings,
    A1MultiObjectMode,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter import (  # noqa: E402
    export_a1_multi_object,
)
from run_bake_integration import PNG_SIGNATURE, _assert  # noqa: E402
from run_grenade_bump_displacement_normal_uv_integration import (  # noqa: E402
    _require_loaded_blend,
)
from run_grenade_plane_ui_coverage_integration import (  # noqa: E402
    _capture_ui_scene_profile,
    _datablock_fingerprint,
    _multi_source,
    _register_steps,
    _require_mesh_object,
    _settings_from_ui_profile,
    _source_fingerprint,
    _unregister_steps,
)


_TARGET_OBJECT = "Cylinder.019"
_COMPANION_OBJECT = "Cube"
_VISUAL_SMOKE_SAMPLES = 16


def _parse_arguments() -> argparse.Namespace:
    arguments = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(
        description="Bake Cylinder.019 + Cube for grenade camera-context color review."
    )
    parser.add_argument(
        "--expected-blend",
        required=True,
        help="Exact grenade.blend path Blender must already have loaded.",
    )
    parser.add_argument(
        "--output-directory",
        required=True,
        help="Existing empty directory that will receive the focused JSON/PNG outputs.",
    )
    return parser.parse_args(arguments)


def _require_empty_output_directory(raw_path: str) -> Path:
    output = Path(raw_path).expanduser().resolve(strict=False)
    output.mkdir(parents=True, exist_ok=True)
    existing = tuple(output.iterdir())
    _assert(
        not existing,
        "visual smoke output directory must be empty before the test starts: "
        f"path={output}, entries={tuple(item.name for item in existing)!r}",
    )
    return output


def _image_metrics(path: Path) -> dict[str, float | int]:
    _assert(path.is_file(), f"focused baked image is missing: {path}")
    image = bpy.data.images.load(str(path), check_existing=False)
    try:
        width = int(image.size[0])
        height = int(image.size[1])
        values = tuple(float(value) for value in image.pixels)
    finally:
        bpy.data.images.remove(image, do_unlink=True)

    _assert(width > 0 and height > 0, f"invalid baked image size: {(width, height)}")
    _assert(len(values) == width * height * 4, "unexpected baked RGBA buffer length")

    opaque = []
    for offset in range(0, len(values), 4):
        r, g, b, a = values[offset : offset + 4]
        if a > 1.0 / 255.0:
            opaque.append((r, g, b, a))
    _assert(opaque, "focused baked image contains no non-transparent pixels")

    count = len(opaque)
    mean_r = sum(pixel[0] for pixel in opaque) / count
    mean_g = sum(pixel[1] for pixel in opaque) / count
    mean_b = sum(pixel[2] for pixel in opaque) / count
    near_white = sum(
        1 for r, g, b, _a in opaque if min(r, g, b) >= 0.95
    )
    green_dominant = sum(
        1
        for r, g, b, _a in opaque
        if g >= r + 0.15 and g >= b + 0.15
    )
    return {
        "width": width,
        "height": height,
        "opaque_pixels": count,
        "mean_r": mean_r,
        "mean_g": mean_g,
        "mean_b": mean_b,
        "near_white_fraction": near_white / count,
        "green_dominant_fraction": green_dominant / count,
    }


def _run(expected_blend: str, raw_output_directory: str) -> None:
    loaded = _require_loaded_blend(expected_blend)
    output_directory = _require_empty_output_directory(raw_output_directory)
    completed = _register_steps()
    try:
        scene = bpy.context.scene
        target = _require_mesh_object(_TARGET_OBJECT)
        companion = _require_mesh_object(_COMPANION_OBJECT)

        target_before = _source_fingerprint(target, scene)
        companion_before = _source_fingerprint(companion, scene)
        datablocks_before = _datablock_fingerprint()

        scene_profile, texture_size = _capture_ui_scene_profile(
            scene,
            output_directory,
        )
        target_settings = _settings_from_ui_profile(
            target,
            scene_profile,
            texture_size,
        )
        companion_settings = _settings_from_ui_profile(
            companion,
            scene_profile,
            texture_size,
        )
        target_settings = replace(
            target_settings,
            bake_execution=replace(
                target_settings.bake_execution,
                samples=_VISUAL_SMOKE_SAMPLES,
            ),
        )
        companion_settings = replace(
            companion_settings,
            bake_execution=replace(
                companion_settings.bake_execution,
                samples=_VISUAL_SMOKE_SAMPLES,
            ),
        )

        result = export_a1_multi_object(
            (
                _multi_source(target, target_settings, 1),
                _multi_source(companion, companion_settings, 2),
            ),
            A1MultiObjectExportSettings(
                output_directory=output_directory,
                output_stem="Grenade_Camera_Context_Color_Smoke",
                mode=A1MultiObjectMode.STANDALONE,
            ),
            context=bpy.context,
            scene=scene,
        )
        _assert(
            bool(result.success),
            "focused grenade camera-context color export failed: "
            f"issues={result.issues!r}, statistics={dict(result.statistics)!r}",
        )

        outputs = tuple(Path(path).resolve(strict=False) for path in result.output_files)
        png_files = tuple(path for path in outputs if path.suffix.lower() == ".png")
        json_files = tuple(path for path in outputs if path.suffix.lower() == ".json")
        _assert(len(png_files) == 2, f"expected two focused PNG outputs: {png_files!r}")
        _assert(len(json_files) == 1, f"expected one focused JSON output: {json_files!r}")
        _assert(
            all(path.read_bytes().startswith(PNG_SIGNATURE) for path in png_files),
            f"focused outputs contain a non-PNG payload: {png_files!r}",
        )

        target_png = next(
            (
                path
                for path in png_files
                if path.name.casefold().startswith(_TARGET_OBJECT.casefold())
            ),
            None,
        )
        _assert(
            target_png is not None,
            f"unable to identify {_TARGET_OBJECT!r} PNG in {png_files!r}",
        )
        metrics = _image_metrics(target_png)

        target_prefix = "component.object_1:Cylinder.019."
        strategy = result.statistics.get(target_prefix + "bake_strategy_ids", "")
        bake_mode = result.statistics.get(target_prefix + "bake_mode", "")
        scene_aware = result.statistics.get(target_prefix + "bake_scene_aware", 0)
        _assert(
            "CAMERA_COMBINED" in str(strategy),
            f"focused target lost the camera-context strategy: {strategy!r}",
        )
        _assert(
            str(bake_mode) == "EMIT",
            "Normal/UV camera-context target must flatten surface color through EMIT, "
            f"got bake_mode={bake_mode!r}",
        )
        _assert(int(scene_aware) == 1, "focused target is no longer camera-context aware")

        _assert(
            _source_fingerprint(target, scene) == target_before,
            "focused color smoke changed target source/scene state",
        )
        _assert(
            _source_fingerprint(companion, scene) == companion_before,
            "focused color smoke changed companion source/scene state",
        )
        _assert(
            _datablock_fingerprint() == datablocks_before,
            "focused color smoke leaked or removed Blender datablocks",
        )

        print(
            "[GRENADE-CAMERA-SURFACE-COLOR] PASS "
            f"blend={loaded} target={target.name_full!r} companion={companion.name_full!r} "
            f"texture={texture_size} samples={_VISUAL_SMOKE_SAMPLES} "
            f"strategy={strategy!r} bake_mode={bake_mode!r} output={target_png} "
            f"metrics={metrics!r} source=unchanged",
            flush=True,
        )
    finally:
        _unregister_steps(completed)


def main() -> None:
    arguments = _parse_arguments()
    _run(arguments.expected_blend, arguments.output_directory)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
