"""Run positive Depth parallax directly against the real mushrooms asset.

Blender must open ``E:\\test_BtSe\\mushrooms\\mushrooms.blend`` (or the exact caller
path) before executing this script. The regression reproduces the manual failure with a
50 degree horizon and ``Max Depth Points = 128``. Dense reserve ownership must survive a
budgeted proxy without mutating the loaded scene or taking unbounded time.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from math import radians
import os
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
    prepare_a1_multi_object,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.a1_object_preparation import (  # noqa: E402
    PreparedDepthA1Object,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import (  # noqa: E402
    DepthParallaxSettings,
)
from run_bake_integration import (  # noqa: E402
    _assert,
    _capture_context,
    _temporary_datablock_names,
)
from run_mushrooms_real_blend_integration import (  # noqa: E402
    _camera_fingerprint,
    _object_fingerprint,
    _require_loaded_blend,
    _require_source_objects,
    _settings,
)


_MAX_POINTS = 128
_HORIZON_DEGREES = 50.0
_MAX_PREPARATION_SECONDS = 120.0
_MULTI_STEM = "MushroomsRealParallaxBudget"


def _parse_arguments() -> argparse.Namespace:
    arguments = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(
        description="Run budgeted positive parallax against the real mushrooms asset."
    )
    parser.add_argument(
        "--expected-blend",
        required=True,
        help="Exact .blend path that Blender must already have loaded.",
    )
    return parser.parse_args(arguments)


def _canonical_path(value: str | Path) -> str:
    return os.path.normcase(
        os.path.normpath(str(Path(value).expanduser().resolve(strict=False)))
    )


def _positive_parallax_settings(output_directory: Path, *, prefix: str):
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


def _prepared_by_component(prepared_multi) -> dict[str, PreparedDepthA1Object]:
    result: dict[str, PreparedDepthA1Object] = {}
    for source, prepared in zip(
        prepared_multi.sources,
        prepared_multi.objects,
        strict=True,
    ):
        if not isinstance(prepared, PreparedDepthA1Object):
            raise TypeError(
                f"Component {source.component_id} returned {type(prepared)!r}"
            )
        result[source.component_id] = prepared
    return result


def _run(expected_blend: str) -> None:
    loaded = _require_loaded_blend(expected_blend)
    _assert(
        _canonical_path(loaded) == _canonical_path(expected_blend),
        f"wrong real blend loaded: {loaded}",
    )
    sources = _require_source_objects()

    context_before = _capture_context()
    frame_before = int(bpy.context.scene.frame_current)
    camera_before = _camera_fingerprint()
    object_before = {
        source.name: _object_fingerprint(source)
        for source in sources
    }
    temporary_before = _temporary_datablock_names()

    with tempfile.TemporaryDirectory(
        prefix="spine2d_mushrooms_parallax_budget_"
    ) as directory:
        output_directory = Path(directory)
        component_sources = tuple(
            A1MultiObjectSource(
                source_object=source,
                component_id=f"object_{index}:{source.name}",
                settings=_positive_parallax_settings(
                    output_directory,
                    prefix=f"MushroomsBudget_{source.name.replace('.', '_')}",
                ),
            )
            for index, source in enumerate(sources, start=1)
        )

        started = perf_counter()
        prepared_multi = prepare_a1_multi_object(
            component_sources,
            A1MultiObjectExportSettings(
                output_directory=output_directory,
                output_stem=_MULTI_STEM,
                mode=A1MultiObjectMode.STANDALONE,
            ),
            context=bpy.context,
            scene=bpy.context.scene,
        )
        elapsed = perf_counter() - started

        written_files = tuple(
            path
            for path in output_directory.rglob("*")
            if path.is_file()
        )
        _assert(
            not written_files,
            f"preparation unexpectedly wrote files: {written_files}",
        )

    _assert(
        elapsed <= _MAX_PREPARATION_SECONDS,
        (
            "real mushrooms positive-parallax preparation exceeded the bounded runtime; "
            f"elapsed={elapsed:.3f}s, limit={_MAX_PREPARATION_SECONDS}s"
        ),
    )

    prepared = _prepared_by_component(prepared_multi)
    plane = prepared["object_1:Plane.008"]
    cube = prepared["object_2:Cube.012"]

    plane_package = plane.depth_parallax_package
    cube_package = cube.depth_parallax_package
    for label, package in (
        ("Plane.008", plane_package),
        ("Cube.012", cube_package),
    ):
        _assert(
            len(package.union_snapshot.vertices) <= _MAX_POINTS,
            (
                f"{label} exceeded Max Depth Points: "
                f"points={len(package.union_snapshot.vertices)}"
            ),
        )
        _assert(
            package.reserve_surfaces,
            f"{label} positive horizon produced no reserve surfaces",
        )
        _assert(
            all(surface.source_face_indices for surface in package.reserve_surfaces),
            f"{label} reserve surface lost render ownership",
        )

    plane_proxy_surfaces = tuple(
        surface
        for surface in plane_package.reserve_surfaces
        if len(surface.source_face_indices) > len(surface.snapshot.faces)
    )
    _assert(
        plane_proxy_surfaces,
        (
            "Plane.008 did not exercise compact reserve ownership; "
            f"surfaces={plane_package.reserve_surfaces}"
        ),
    )
    _assert(
        "parallax-budget-proxy" in plane_package.union_snapshot.snapshot_id,
        (
            "Plane.008 did not retain the budget-proxy marker: "
            f"{plane_package.union_snapshot.snapshot_id}"
        ),
    )

    _assert(
        _capture_context() == context_before,
        "real blend selection or active object changed",
    )
    _assert(
        int(bpy.context.scene.frame_current) == frame_before,
        "real blend Scene frame changed",
    )
    _assert(
        _camera_fingerprint() == camera_before,
        "real blend active camera changed",
    )
    _assert(
        {
            source.name: _object_fingerprint(source)
            for source in sources
        }
        == object_before,
        "real blend source objects, meshes, materials, or modifiers changed",
    )
    _assert(
        _temporary_datablock_names() == temporary_before,
        "real blend preparation leaked temporary Blender datablocks",
    )

    plane_owner_count = sum(
        len(surface.source_face_indices)
        for surface in plane_package.reserve_surfaces
    )
    print(
        "[MUSHROOMS-REAL-PARALLAX-BUDGET] PASS "
        f"blend={loaded} horizon={_HORIZON_DEGREES}deg max_points={_MAX_POINTS} "
        f"plane_union={len(plane_package.union_snapshot.vertices)} "
        f"plane_reserve_owners={plane_owner_count} "
        f"plane_views={len(plane_package.reserve_surfaces)} "
        f"cube_union={len(cube_package.union_snapshot.vertices)} "
        f"elapsed={elapsed:.3f}s mode=PROXY_OR_EXACT pipeline=public-multi-object"
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
