"""Run Depth preparation directly against the real mushrooms ``.blend`` asset.

This runner never creates replacement geometry. Blender must open the caller-supplied
``mushrooms.blend`` before the script starts. The runner then:

1. verifies the exact loaded file path;
2. reads and normalizes the evaluated meshes of ``Plane.008`` and ``Cube.012`` through
   the same internal preparation stages used by production;
3. scans every evaluated n-gon and reports all planarity-policy violations together;
4. triangulates every normalized source snapshot;
5. runs both real objects through the public multi-object Depth preparation route;
6. verifies that source objects, meshes, modifiers, camera, selection, frame, and
   temporary datablocks remain unchanged.

Usage:

    blender --background E:\\test_BtSe\\mushrooms\\mushrooms.blend \
        --python tests/blender_headless/run_mushrooms_real_blend_integration.py -- \
        --expected-blend E:\\test_BtSe\\mushrooms\\mushrooms.blend
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from math import sqrt
import os
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
    A1MultiObjectExportSettings,
    A1MultiObjectMode,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter import (  # noqa: E402
    A1MultiObjectSource,
    prepare_a1_multi_object,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.a1_depth_source_geometry_preparation import (  # noqa: E402
    _canonicalize_depth_evaluated_identity,
    _normal_camera_request_settings,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.a1_object_preparation import (  # noqa: E402
    PreparedDepthA1Object,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.a1_source_geometry_preparation import (  # noqa: E402
    _normalize_source_geometry,
    _read_source_snapshot,
    _resolve_source_request,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.geometry import (  # noqa: E402
    MeshSnapshot,
    MeshSnapshotValidator,
    TriangulationSettings,
    triangulate_snapshot,
)
from run_bake_integration import (  # noqa: E402
    _assert,
    _capture_context,
    _temporary_datablock_names,
)
import run_depth_array_modifier_integration as depth_helpers  # noqa: E402


_OBJECT_NAMES = ("Plane.008", "Cube.012")
_MULTI_STEM = "MushroomsRealBlendRegression"


class RealBlendRegressionError(RuntimeError):
    """Raised when the loaded mushrooms asset violates a regression contract."""


def _parse_arguments() -> argparse.Namespace:
    """Parse arguments following Blender's ``--`` separator."""

    arguments = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(
        description="Run the Spine exporter against the real mushrooms blend asset."
    )
    parser.add_argument(
        "--expected-blend",
        required=True,
        help="Exact .blend path that Blender must already have loaded.",
    )
    return parser.parse_args(arguments)


def _canonical_path(value: str | Path) -> str:
    """Return a case-normalized absolute path suitable for Windows comparison."""

    resolved = Path(value).expanduser().resolve(strict=False)
    return os.path.normcase(os.path.normpath(str(resolved)))


def _require_loaded_blend(expected_blend: str) -> Path:
    """Require Blender to be running the exact requested asset, not a synthetic scene."""

    expected = Path(expected_blend).expanduser().resolve(strict=False)
    if not expected.is_file():
        raise RealBlendRegressionError(
            f"Expected mushrooms blend does not exist: {expected}"
        )

    loaded_value = bpy.data.filepath
    if not loaded_value:
        raise RealBlendRegressionError(
            "Blender has no loaded .blend; pass the real mushrooms file before --python"
        )

    loaded = Path(loaded_value).expanduser().resolve(strict=False)
    if _canonical_path(loaded) != _canonical_path(expected):
        raise RealBlendRegressionError(
            "Wrong blend loaded for real-asset regression: "
            f"loaded={loaded}, expected={expected}"
        )
    return loaded


def _require_source_objects() -> tuple[object, ...]:
    """Resolve the two actual Mesh objects by their names in the loaded asset."""

    resolved: list[object] = []
    for name in _OBJECT_NAMES:
        source = bpy.data.objects.get(name)
        if source is None:
            raise RealBlendRegressionError(
                f"Required object {name!r} is absent from {bpy.data.filepath}"
            )
        if source.type != "MESH" or source.data is None:
            raise RealBlendRegressionError(
                f"Required object {name!r} is not a Mesh object"
            )
        resolved.append(source)
    return tuple(resolved)


def _settings(output_directory: Path, *, prefix: str):
    """Return zero-horizon Depth settings for one real component."""

    if not isinstance(output_directory, Path):
        raise TypeError("output_directory must be pathlib.Path")
    if not isinstance(prefix, str) or not prefix.strip():
        raise ValueError("prefix must be a non-empty string")

    base = depth_helpers._settings(output_directory)
    return replace(
        base,
        prefix=prefix,
        output_stem=prefix,
        json_output_stem=prefix,
        export=replace(
            base.export,
            output_directory=output_directory,
        ),
    )


def _object_fingerprint(source) -> tuple[object, ...]:
    """Capture persistent source state without evaluating or modifying the object."""

    modifiers = tuple(
        (
            modifier.name,
            modifier.type,
            bool(modifier.show_viewport),
            bool(modifier.show_render),
        )
        for modifier in source.modifiers
    )
    return (
        source.name,
        source.data.name,
        tuple(tuple(float(value) for value in row) for row in source.matrix_world),
        tuple(tuple(float(value) for value in vertex.co) for vertex in source.data.vertices),
        tuple(
            tuple(int(value) for value in polygon.vertices)
            for polygon in source.data.polygons
        ),
        tuple(
            material.name if material is not None else None
            for material in source.data.materials
        ),
        modifiers,
    )


def _camera_fingerprint() -> tuple[object, ...]:
    """Capture the active Scene camera and its persistent projection state."""

    camera = bpy.context.scene.camera
    if camera is None or camera.type != "CAMERA" or camera.data is None:
        raise RealBlendRegressionError(
            "The loaded mushrooms scene has no active Camera object"
        )
    data = camera.data
    return (
        camera.name,
        data.name,
        tuple(tuple(float(value) for value in row) for row in camera.matrix_world),
        str(data.type),
        float(data.clip_start),
        float(data.clip_end),
        float(data.lens),
        float(data.ortho_scale),
    )


def _read_normalized_snapshot(source, settings) -> MeshSnapshot:
    """Execute the exact production read, identity, and world-normalization stages."""

    validated_settings = _normal_camera_request_settings(settings)
    request = _resolve_source_request(
        source,
        validated_settings,
        bpy.context.scene,
    )
    snapshot, _modifier_count, warnings, _uv_report = _read_source_snapshot(
        source,
        request.object_id,
        validated_settings,
        scene=request.scene,
        depsgraph=request.depsgraph,
    )
    snapshot, warnings, _statistics, _rebase = (
        _canonicalize_depth_evaluated_identity(
            snapshot,
            warnings,
            {},
            object_id=request.object_id,
        )
    )
    normalized = _normalize_source_geometry(
        snapshot,
        validated_settings,
        warnings,
        object_id=request.object_id,
    )
    MeshSnapshotValidator().validate_or_raise(normalized.snapshot)
    return normalized.snapshot


def _face_planarity_metrics(
    snapshot: MeshSnapshot,
    face,
) -> tuple[float, float, float]:
    """Measure one n-gon with the same Newell-plane definition as production."""

    loop_map = snapshot.loop_by_id()
    vertex_map = snapshot.vertex_by_id()
    points = tuple(
        vertex_map[loop_map[loop_id].vertex_id].position
        for loop_id in face.loop_ids
    )
    if len(points) < 4:
        raise ValueError("face_planarity_metrics requires an n-gon")

    newell = [0.0, 0.0, 0.0]
    for index, current in enumerate(points):
        following = points[(index + 1) % len(points)]
        newell[0] += (
            (current[1] - following[1])
            * (current[2] + following[2])
        )
        newell[1] += (
            (current[2] - following[2])
            * (current[0] + following[0])
        )
        newell[2] += (
            (current[0] - following[0])
            * (current[1] + following[1])
        )

    magnitude = sqrt(sum(component * component for component in newell))
    if magnitude <= 0.0:
        raise RealBlendRegressionError(
            f"{snapshot.object_name} face {face.id.index} has a collapsed Newell normal"
        )
    normal = tuple(component / magnitude for component in newell)
    centroid = tuple(
        sum(point[axis] for point in points) / float(len(points))
        for axis in range(3)
    )
    maximum_distance = max(
        abs(
            sum(
                (point[axis] - centroid[axis]) * normal[axis]
                for axis in range(3)
            )
        )
        for point in points
    )
    extents = tuple(
        max(point[axis] for point in points)
        - min(point[axis] for point in points)
        for axis in range(3)
    )
    polygon_scale = sqrt(sum(extent * extent for extent in extents))
    if polygon_scale <= 0.0:
        raise RealBlendRegressionError(
            f"{snapshot.object_name} face {face.id.index} has zero polygon scale"
        )
    return maximum_distance, polygon_scale, maximum_distance / polygon_scale


def _scan_snapshot_planarity(
    snapshot: MeshSnapshot,
) -> tuple[tuple[object, ...], ...]:
    """Scan every n-gon and fail once with the complete policy-violation set."""

    settings = TriangulationSettings()
    records: list[tuple[object, ...]] = []
    violations: list[tuple[object, ...]] = []

    for face in sorted(snapshot.faces, key=lambda item: item.id.index):
        if len(face.loop_ids) <= 3:
            continue
        maximum, scale, normalized = _face_planarity_metrics(snapshot, face)
        effective = max(
            settings.planarity_tolerance,
            settings.relative_planarity_tolerance * scale,
        )
        record = (
            int(face.id.index),
            int(face.source_id.face_index),
            len(face.loop_ids),
            maximum,
            scale,
            normalized,
            effective,
        )
        records.append(record)
        if (
            maximum > effective
            or normalized > settings.maximum_relative_planarity_warp
        ):
            violations.append(record)

    records.sort(key=lambda item: (float(item[3]), float(item[5])), reverse=True)
    violations.sort(
        key=lambda item: (float(item[3]), float(item[5])),
        reverse=True,
    )

    if violations:
        details = "\n".join(
            (
                f"  object={snapshot.object_name} local_face={record[0]} "
                f"source_face={record[1]} corners={record[2]} "
                f"maximum={record[3]} scale={record[4]} "
                f"normalized={record[5]} effective={record[6]}"
            )
            for record in violations
        )
        raise RealBlendRegressionError(
            "Real mushrooms n-gon planarity scan found all blockers:\n"
            f"{details}"
        )

    # This catches self-intersection, declared-normal drift, reversed generated triangles,
    # and other deterministic triangulation errors after planarity has been scanned fully.
    triangulate_snapshot(snapshot)
    return tuple(records)


def _prepared_by_component(prepared_multi) -> dict[str, PreparedDepthA1Object]:
    """Resolve public results by component identity instead of tuple position."""

    result: dict[str, PreparedDepthA1Object] = {}
    for source, prepared in zip(
        prepared_multi.sources,
        prepared_multi.objects,
        strict=True,
    ):
        if not isinstance(prepared, PreparedDepthA1Object):
            raise RealBlendRegressionError(
                f"Component {source.component_id} returned {type(prepared)!r}"
            )
        if source.component_id in result:
            raise RealBlendRegressionError(
                f"Duplicate prepared component {source.component_id}"
            )
        result[source.component_id] = prepared
    return result


def _run(expected_blend: str) -> None:
    loaded_blend = _require_loaded_blend(expected_blend)
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
        prefix="spine2d_mushrooms_real_blend_"
    ) as directory:
        output_directory = Path(directory)
        component_sources: list[A1MultiObjectSource] = []
        scan_records: dict[str, tuple[tuple[object, ...], ...]] = {}

        for index, source in enumerate(sources, start=1):
            prefix = f"MushroomsReal_{source.name.replace('.', '_')}"
            settings = _settings(output_directory, prefix=prefix)
            normalized = _read_normalized_snapshot(source, settings)
            scan_records[source.name] = _scan_snapshot_planarity(normalized)
            component_sources.append(
                A1MultiObjectSource(
                    source_object=source,
                    component_id=f"object_{index}:{source.name}",
                    settings=settings,
                )
            )

        if len(component_sources) != 2:
            raise RealBlendRegressionError(
                f"Expected two real sources, got {len(component_sources)}"
            )

        prepared_multi = prepare_a1_multi_object(
            tuple(component_sources),
            A1MultiObjectExportSettings(
                output_directory=output_directory,
                output_stem=_MULTI_STEM,
                mode=A1MultiObjectMode.STANDALONE,
            ),
            context=bpy.context,
            scene=bpy.context.scene,
        )

        written_files = tuple(
            path
            for path in output_directory.rglob("*")
            if path.is_file()
        )
        if written_files:
            raise RealBlendRegressionError(
                f"Preparation unexpectedly wrote files: {written_files}"
            )

    prepared_by_component = _prepared_by_component(prepared_multi)
    expected_components = {
        f"object_{index}:{source.name}"
        for index, source in enumerate(sources, start=1)
    }
    if set(prepared_by_component) != expected_components:
        raise RealBlendRegressionError(
            "Prepared component set changed: "
            f"actual={set(prepared_by_component)}, expected={expected_components}"
        )

    for component_id, prepared in prepared_by_component.items():
        triangle_count = int(
            prepared.statistics["depth_projection_source_triangle_count"]
        )
        if triangle_count <= 0:
            raise RealBlendRegressionError(
                f"{component_id} produced no projected source triangles"
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

    scan_summary = " ".join(
        (
            f"{name}_ngons={len(records)} "
            f"{name}_max_abs={max((record[3] for record in records), default=0.0)} "
            f"{name}_max_norm={max((record[5] for record in records), default=0.0)}"
        )
        for name, records in scan_records.items()
    )
    print(
        "[MUSHROOMS-REAL-BLEND] PASS "
        f"blend={loaded_blend} objects=Plane.008,Cube.012 "
        f"{scan_summary} pipeline=public-multi-object"
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
