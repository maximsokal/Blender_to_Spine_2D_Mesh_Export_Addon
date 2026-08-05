"""Run Depth preparation directly against the real mushrooms ``.blend`` asset.

Blender must open the caller-supplied asset before this script starts. No replacement
geometry is created. Every evaluated polygon from ``Plane.008`` and ``Cube.012`` is read
through the production adapter, normalized, triangulated, and checked for exact source-face
coverage before the same objects enter the public multi-object Depth pipeline.

Non-planar quads are valid curved-surface geometry. Their strict planarity metrics are
reported as diagnostics, never used to delete or skip faces in the default export path.
The request renderer is resolved from the loaded Scene so this direct asset regression
never imports a synthetic fixture's Cycles default into an Eevee-authored file.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, replace
from math import isfinite, sqrt
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
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.render_engine_contract import (  # noqa: E402
    render_engine_contract,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.geometry import (  # noqa: E402
    MeshSnapshot,
    MeshSnapshotValidator,
    NonPlanarPolygonPolicy,
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
_NORMAL_LENGTH_TOLERANCE = 1.0e-6


class RealBlendRegressionError(RuntimeError):
    """Raised when the loaded mushrooms asset violates a regression contract."""


@dataclass(frozen=True, slots=True)
class SnapshotTriangulationScan:
    object_name: str
    source_face_count: int
    ngon_count: int
    strict_planarity_violation_count: int
    expected_triangle_count: int
    actual_triangle_count: int
    maximum_absolute_warp: float
    maximum_normalized_warp: float

    def __post_init__(self) -> None:
        if not isinstance(self.object_name, str) or not self.object_name.strip():
            raise ValueError("object_name must be a non-empty string")
        for field_name in (
            "source_face_count",
            "ngon_count",
            "strict_planarity_violation_count",
            "expected_triangle_count",
            "actual_triangle_count",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")
        for field_name in (
            "maximum_absolute_warp",
            "maximum_normalized_warp",
        ):
            value = float(getattr(self, field_name))
            if not isfinite(value) or value < 0.0:
                raise ValueError(f"{field_name} must be finite and non-negative")


def _parse_arguments() -> argparse.Namespace:
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
    resolved = Path(value).expanduser().resolve(strict=False)
    return os.path.normcase(os.path.normpath(str(resolved)))


def _require_loaded_blend(expected_blend: str) -> Path:
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


def _loaded_scene_render_engine() -> str:
    """Return the canonical engine of the currently loaded real Scene."""

    scene = getattr(bpy.context, "scene", None)
    render = getattr(scene, "render", None)
    value = getattr(render, "engine", None)
    if not isinstance(value, str) or not value.strip():
        raise RealBlendRegressionError(
            "The loaded mushrooms Scene has no valid render engine"
        )
    try:
        return render_engine_contract(value).blender_engine
    except Exception as exc:
        raise RealBlendRegressionError(
            "The loaded mushrooms Scene uses an unsupported render engine: "
            f"{value!r}"
        ) from exc


def _settings(output_directory: Path, *, prefix: str):
    """Build real-asset settings without importing a synthetic renderer default."""

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
        bake_execution=replace(
            base.bake_execution,
            render_engine=_loaded_scene_render_engine(),
        ),
    )


def _object_fingerprint(source) -> tuple[object, ...]:
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
    if not isfinite(magnitude) or magnitude <= 0.0:
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
    if not isfinite(polygon_scale) or polygon_scale <= 0.0:
        raise RealBlendRegressionError(
            f"{snapshot.object_name} face {face.id.index} has zero polygon scale"
        )
    return maximum_distance, polygon_scale, maximum_distance / polygon_scale


def _expected_source_face_multiplicity(snapshot: MeshSnapshot) -> Counter:
    return Counter(
        {
            face.source_id: max(1, len(face.loop_ids) - 2)
            for face in snapshot.faces
        }
    )


def _scan_snapshot_triangulation(
    snapshot: MeshSnapshot,
) -> SnapshotTriangulationScan:
    """Triangulate every face and require exact N-2 lineage coverage."""

    default_settings = TriangulationSettings()
    if (
        default_settings.non_planar_policy
        is not NonPlanarPolygonPolicy.TRIANGULATE
    ):
        raise RealBlendRegressionError(
            "Default export triangulation policy is not TRIANGULATE"
        )

    strict_violation_count = 0
    maximum_absolute_warp = 0.0
    maximum_normalized_warp = 0.0
    ngon_count = 0

    for face in sorted(snapshot.faces, key=lambda item: item.id.index):
        if len(face.loop_ids) <= 3:
            continue
        ngon_count += 1
        maximum, scale, normalized = _face_planarity_metrics(snapshot, face)
        effective = max(
            default_settings.planarity_tolerance,
            default_settings.relative_planarity_tolerance * scale,
        )
        maximum_absolute_warp = max(maximum_absolute_warp, maximum)
        maximum_normalized_warp = max(maximum_normalized_warp, normalized)
        if (
            maximum > effective
            or normalized
            > default_settings.maximum_relative_planarity_warp
        ):
            strict_violation_count += 1

    expected_counts = _expected_source_face_multiplicity(snapshot)
    expected_triangle_count = sum(expected_counts.values())
    triangulated = triangulate_snapshot(snapshot, default_settings)
    actual_counts = Counter(
        face.source_id
        for face in triangulated.snapshot.faces
    )

    if actual_counts != expected_counts:
        missing = expected_counts - actual_counts
        excess = actual_counts - expected_counts
        raise RealBlendRegressionError(
            "Real mushrooms triangulation changed SourceFaceId multiplicity; "
            f"object={snapshot.object_name}, missing={tuple(missing.items())}, "
            f"excess={tuple(excess.items())}"
        )
    if len(triangulated.snapshot.faces) != expected_triangle_count:
        raise RealBlendRegressionError(
            "Real mushrooms triangulation did not produce exact N-2 coverage; "
            f"object={snapshot.object_name}, expected={expected_triangle_count}, "
            f"actual={len(triangulated.snapshot.faces)}"
        )
    if any(len(face.loop_ids) != 3 for face in triangulated.snapshot.faces):
        raise RealBlendRegressionError(
            f"{snapshot.object_name} triangulation retained a non-triangle face"
        )

    for face in triangulated.snapshot.faces:
        normal_length = sqrt(
            sum(float(component) ** 2 for component in face.normal)
        )
        if (
            not isfinite(normal_length)
            or abs(normal_length - 1.0) > _NORMAL_LENGTH_TOLERANCE
        ):
            raise RealBlendRegressionError(
                "Generated triangle normal is invalid; "
                f"object={snapshot.object_name}, face={face.id.index}, "
                f"normal={face.normal}, length={normal_length}"
            )

    return SnapshotTriangulationScan(
        object_name=snapshot.object_name,
        source_face_count=len(snapshot.faces),
        ngon_count=ngon_count,
        strict_planarity_violation_count=strict_violation_count,
        expected_triangle_count=expected_triangle_count,
        actual_triangle_count=len(triangulated.snapshot.faces),
        maximum_absolute_warp=maximum_absolute_warp,
        maximum_normalized_warp=maximum_normalized_warp,
    )


def _prepared_by_component(prepared_multi) -> dict[str, PreparedDepthA1Object]:
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
    render_engine_before = _loaded_scene_render_engine()
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
        scans: dict[str, SnapshotTriangulationScan] = {}

        for index, source in enumerate(sources, start=1):
            prefix = f"MushroomsReal_{source.name.replace('.', '_')}"
            settings = _settings(output_directory, prefix=prefix)
            if settings.bake_execution.render_engine != render_engine_before:
                raise RealBlendRegressionError(
                    "Real-asset request renderer differs from loaded Scene; "
                    f"requested={settings.bake_execution.render_engine}, "
                    f"scene={render_engine_before}"
                )
            normalized = _read_normalized_snapshot(source, settings)
            scans[source.name] = _scan_snapshot_triangulation(normalized)
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
            f"actual={set(prepared_by_component)}, "
            f"expected={expected_components}"
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
        _loaded_scene_render_engine() == render_engine_before,
        "real blend render engine changed",
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
            f"{name}_faces={scan.source_face_count} "
            f"{name}_ngons={scan.ngon_count} "
            f"{name}_triangles={scan.actual_triangle_count} "
            f"{name}_strict_warp={scan.strict_planarity_violation_count} "
            f"{name}_max_abs={scan.maximum_absolute_warp} "
            f"{name}_max_norm={scan.maximum_normalized_warp}"
        )
        for name, scan in scans.items()
    )
    print(
        "[MUSHROOMS-REAL-BLEND] PASS "
        f"blend={loaded_blend} objects=Plane.008,Cube.012 "
        f"render_engine={render_engine_before} {scan_summary} "
        "face_loss=0 policy=TRIANGULATE pipeline=public-multi-object"
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
