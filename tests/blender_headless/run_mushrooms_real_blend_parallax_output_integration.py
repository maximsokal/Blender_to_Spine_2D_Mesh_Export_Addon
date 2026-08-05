"""Export the real mushrooms asset through render, crop, UV remap, and commit.

This runner owns the regression for a valid full-frame reserve UV becoming approximately
``-9.795`` after an alpha-only crop. Blender must open the caller-supplied mushrooms file
before this script starts. The public multi-object export must render FRONT plus all
reserve views, expand each crop to prepared attachment UVs, serialize unit-square mesh
UVs, and leave the loaded scene unchanged.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from math import radians
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
    A1MultiObjectExportSettings,
    A1MultiObjectMode,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter import (  # noqa: E402
    A1MultiObjectSource,
    export_a1_multi_object,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import (  # noqa: E402
    DepthParallaxSettings,
)
from run_bake_integration import (  # noqa: E402
    PNG_SIGNATURE,
    _assert,
    _capture_context,
    _capture_scene_bake_state,
    _temporary_datablock_names,
)
from run_camera_projection_integration import (  # noqa: E402
    _scene_render_fingerprint,
)
from run_mushrooms_real_blend_integration import (  # noqa: E402
    _camera_fingerprint,
    _loaded_scene_render_engine,
    _object_fingerprint,
    _require_loaded_blend,
    _require_source_objects,
    _settings,
)


_MAX_POINTS = 128
_HORIZON_DEGREES = 50.0
_MAX_EXPORT_SECONDS = 300.0
_MULTI_STEM = "MushroomsRealParallaxOutput"
_PLANE_COMPONENT = "object_1:Plane.008"
_EXPECTED_PLANE_VIEWS = (
    "FRONT",
    "RIGHT",
    "UP_RIGHT",
    "UP",
    "UP_LEFT",
    "LEFT",
    "DOWN_LEFT",
    "DOWN",
    "DOWN_RIGHT",
)


def _parse_arguments() -> argparse.Namespace:
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expected-blend",
        required=True,
        help="Exact mushrooms .blend path Blender must already have loaded.",
    )
    return parser.parse_args(values)


def _positive_settings(output_directory: Path, *, prefix: str):
    base = _settings(output_directory, prefix=prefix)
    return replace(
        base,
        bake_execution=replace(
            base.bake_execution,
            depth_projection=replace(
                base.bake_execution.depth_projection,
                max_points=_MAX_POINTS,
            ),
            depth_parallax=DepthParallaxSettings(
                horizon_angle_radians=radians(_HORIZON_DEGREES),
            ),
        ),
    )


def _mesh_uv_streams(value: object, *, path: str = "root"):
    if isinstance(value, dict):
        uvs = value.get("uvs")
        if isinstance(uvs, list):
            yield path, uvs
        for key, child in value.items():
            yield from _mesh_uv_streams(child, path=f"{path}/{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _mesh_uv_streams(child, path=f"{path}[{index}]")


def _assert_serialized_uvs(document: object) -> int:
    stream_count = 0
    for path, stream in _mesh_uv_streams(document):
        if not stream:
            continue
        _assert(len(stream) % 2 == 0, f"odd UV stream at {path}")
        for index, raw in enumerate(stream):
            _assert(
                isinstance(raw, (int, float)) and not isinstance(raw, bool),
                f"non-numeric UV at {path}[{index}]: {raw!r}",
            )
            value = float(raw)
            _assert(
                -1.0e-6 <= value <= 1.0 + 1.0e-6,
                f"serialized UV outside unit square at {path}[{index}]={value}",
            )
        stream_count += 1
    _assert(stream_count > 0, "serialized document contains no mesh UV streams")
    return stream_count


def _plane_view_from_filename(path: Path) -> str | None:
    name = path.stem
    if not name.startswith("Plane.008"):
        return None
    marker = "_Parallax_"
    if marker in name:
        suffix = name.split(marker, 1)[1]
        view_id = suffix.split("_Baked", 1)[0].strip().upper()
        return view_id or None
    if "_Baked" in name:
        return "FRONT"
    return None


def _run(expected_blend: str) -> None:
    loaded = _require_loaded_blend(expected_blend)
    sources = _require_source_objects()

    context_before = _capture_context()
    bake_before = _capture_scene_bake_state()
    render_before = _scene_render_fingerprint()
    frame_before = int(bpy.context.scene.frame_current)
    engine_before = _loaded_scene_render_engine()
    camera_before = _camera_fingerprint()
    objects_before = {
        source.name: _object_fingerprint(source)
        for source in sources
    }
    temporary_before = _temporary_datablock_names()

    with tempfile.TemporaryDirectory(
        prefix="spine2d-mushrooms-real-parallax-output-"
    ) as directory:
        output_directory = Path(directory)
        component_sources = tuple(
            A1MultiObjectSource(
                source_object=source,
                component_id=f"object_{index}:{source.name}",
                settings=_positive_settings(
                    output_directory,
                    prefix=source.name,
                ),
            )
            for index, source in enumerate(sources, start=1)
        )
        settings = A1MultiObjectExportSettings(
            output_directory=output_directory,
            output_stem=_MULTI_STEM,
            mode=A1MultiObjectMode.STANDALONE,
        )

        started = perf_counter()
        result = export_a1_multi_object(
            component_sources,
            settings,
            context=bpy.context,
            scene=bpy.context.scene,
        )
        elapsed = perf_counter() - started

        _assert(result.success, f"real mushrooms output failed: {result.issues}")
        _assert(
            elapsed <= _MAX_EXPORT_SECONDS,
            f"real mushrooms output exceeded {_MAX_EXPORT_SECONDS}s: {elapsed:.3f}s",
        )
        output_files = tuple(path.resolve(strict=False) for path in result.output_files)
        _assert(output_files, "real mushrooms output returned no files")
        _assert(
            all(path.is_file() and path.stat().st_size > 0 for path in output_files),
            f"real mushrooms output contains missing/empty files: {output_files}",
        )
        json_files = tuple(path for path in output_files if path.suffix.lower() == ".json")
        png_files = tuple(path for path in output_files if path.suffix.lower() == ".png")
        _assert(len(json_files) == 1, f"expected one JSON: {json_files}")
        _assert(len(png_files) >= 10, f"expected FRONT/reserve PNG outputs: {png_files}")
        for path in png_files:
            _assert(
                path.read_bytes().startswith(PNG_SIGNATURE),
                f"invalid PNG signature: {path}",
            )

        plane_views = tuple(
            sorted(
                view_id
                for path in png_files
                if (view_id := _plane_view_from_filename(path)) is not None
            )
        )
        _assert(
            set(plane_views) == set(_EXPECTED_PLANE_VIEWS),
            f"Plane.008 output view set changed: {plane_views}",
        )

        document = json.loads(json_files[0].read_text(encoding="utf-8"))
        uv_stream_count = _assert_serialized_uvs(document)
        remaining_stages = tuple(
            path
            for path in output_directory.rglob("*")
            if path.is_file() and "spine2d-stage-v3" in path.name
        )
        _assert(
            not remaining_stages,
            f"committed output left staging files: {remaining_stages}",
        )

        _assert(
            int(
                result.statistics.get(
                    f"component.{_PLANE_COMPONENT}.depth_parallax_cropped_view_count",
                    -1,
                )
            )
            == len(_EXPECTED_PLANE_VIEWS),
            "Plane.008 did not finalize FRONT plus eight reserve crops",
        )
        _assert(
            int(
                result.statistics.get(
                    f"component.{_PLANE_COMPONENT}.parallax_texture_output_count",
                    -1,
                )
            )
            == len(_EXPECTED_PLANE_VIEWS),
            "Plane.008 texture output count differs from view count",
        )

        print(
            "[MUSHROOMS-REAL-PARALLAX-OUTPUT] PASS "
            f"blend={loaded} render_engine={engine_before} "
            f"horizon={_HORIZON_DEGREES}deg max_points={_MAX_POINTS} "
            f"outputs={len(output_files)} png={len(png_files)} "
            f"plane_views={len(plane_views)} uv_streams={uv_stream_count} "
            f"elapsed={elapsed:.3f}s transaction=committed crop=alpha-union-geometry",
            flush=True,
        )

    _assert(_capture_context() == context_before, "export changed Blender context")
    _assert(_capture_scene_bake_state() == bake_before, "export changed bake state")
    _assert(_scene_render_fingerprint() == render_before, "export changed render state")
    _assert(
        int(bpy.context.scene.frame_current) == frame_before,
        "export changed Scene frame",
    )
    _assert(
        _loaded_scene_render_engine() == engine_before,
        "export changed render engine",
    )
    _assert(_camera_fingerprint() == camera_before, "export changed active camera")
    _assert(
        {
            source.name: _object_fingerprint(source)
            for source in sources
        }
        == objects_before,
        "export changed real source objects, meshes, materials, or modifiers",
    )
    _assert(
        _temporary_datablock_names() == temporary_before,
        "export leaked temporary Blender datablocks",
    )


def main() -> None:
    namespace = _parse_arguments()
    try:
        _run(namespace.expected_blend)
    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
