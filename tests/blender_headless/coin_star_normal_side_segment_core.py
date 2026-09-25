"""Prove that every real coin Normal region reaches a Spine mesh attachment.

Blender must open the caller-provided ``coin_star.blend`` before this script starts.
The fixture intentionally contains side surfaces that collapse to lines in a signed-axis
Setup Pose. They remain valid deformable weighted meshes and must not be filtered by
region assembly or physical-hull normalization.

The gate derives expected counts from the prepared geometry itself. No fixture-specific
segment count is hardcoded.
"""

from __future__ import annotations

import argparse
from math import isfinite
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
    prepare_a1_object,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.projection import (  # noqa: E402
    A1ProjectionDirection,
)
from run_bake_integration import (  # noqa: E402
    _assert,
    _capture_scene_bake_state,
    _temporary_datablock_names,
)
from run_coin_star_normal_projection_parity_integration import (  # noqa: E402
    _settings,
)
from run_coin_star_real_blend_shader_capability_integration import (  # noqa: E402
    _datablock_fingerprint,
    _object_fingerprint,
    _require_loaded_blend,
    _require_source_object,
    _scene_fingerprint,
)


_RELATIVE_AREA_EPSILON = 1.0e-10
_MINIMUM_AREA_EPSILON = 1.0e-12


def _parse_arguments() -> argparse.Namespace:
    arguments = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(
        description="Validate real coin Normal side-segment retention."
    )
    parser.add_argument(
        "--expected-blend",
        required=True,
        help="Exact coin_star.blend path Blender must already have loaded.",
    )
    return parser.parse_args(arguments)


def _cross(first: tuple[float, float], second: tuple[float, float], third: tuple[float, float]) -> float:
    return (
        (second[0] - first[0]) * (third[1] - first[1])
        - (second[1] - first[1]) * (third[0] - first[0])
    )


def _area_tolerance(points: tuple[tuple[float, float], ...]) -> float:
    if not isinstance(points, tuple) or not points:
        raise ValueError("points must be a non-empty tuple")
    if not all(
        isinstance(point, tuple)
        and len(point) == 2
        and all(isfinite(float(value)) for value in point)
        for point in points
    ):
        raise TypeError("points must contain finite coordinate pairs")

    x_values = tuple(float(point[0]) for point in points)
    y_values = tuple(float(point[1]) for point in points)
    extent = max(
        max(x_values) - min(x_values),
        max(y_values) - min(y_values),
        1.0,
    )
    return max(
        _MINIMUM_AREA_EPSILON,
        extent * extent * _RELATIVE_AREA_EPSILON,
    )


def _snapshot_collapsed_face_indices(snapshot) -> tuple[int, ...]:
    vertices = snapshot.vertex_by_id()
    loops = snapshot.loop_by_id()
    points = tuple(
        (float(vertex.position[0]), float(vertex.position[1]))
        for vertex in snapshot.vertices
    )
    tolerance = _area_tolerance(points)
    collapsed: list[int] = []

    for face in snapshot.faces:
        _assert(
            len(face.loop_ids) == 3,
            f"prepared coin region is not triangulated: face={face.id.index}",
        )
        triangle = tuple(
            (
                float(vertices[loops[loop_id].vertex_id].position[0]),
                float(vertices[loops[loop_id].vertex_id].position[1]),
            )
            for loop_id in face.loop_ids
        )
        if abs(_cross(triangle[0], triangle[1], triangle[2])) <= tolerance:
            collapsed.append(face.id.index)

    return tuple(collapsed)


def _projection_collapsed_triangle_indices(projection) -> tuple[int, ...]:
    positions = tuple(
        (
            float(vertex.bone_position_pixels[0]),
            float(vertex.bone_position_pixels[1]),
        )
        for vertex in projection.request.vertices
    )
    tolerance = _area_tolerance(positions)
    collapsed: list[int] = []

    for offset in range(0, len(projection.request.triangles), 3):
        indices = projection.request.triangles[offset : offset + 3]
        _assert(
            len(indices) == 3,
            f"attachment triangle stream is incomplete: offset={offset}",
        )
        triangle = tuple(positions[index] for index in indices)
        if abs(_cross(triangle[0], triangle[1], triangle[2])) <= tolerance:
            collapsed.append(offset // 3)

    return tuple(collapsed)


def _assert_region_projection_parity(prepared, *, label: str) -> tuple[int, int, int]:
    regions = tuple(prepared.uv_regions.snapshots)
    projections = tuple(prepared.document_assembly.projections)
    _assert(regions, f"{label} prepared no UV regions")
    _assert(
        len(projections) == len(regions),
        f"{label} lost prepared regions during Spine assembly: "
        f"regions={len(regions)}, projections={len(projections)}",
    )

    fully_edge_on = 0
    collapsed_faces = 0
    collapsed_triangles = 0
    for region_index, (snapshot, projection) in enumerate(
        zip(regions, projections, strict=True)
    ):
        face_count = len(snapshot.faces)
        triangle_count = len(projection.request.triangles) // 3
        _assert(
            triangle_count == face_count,
            f"{label} region {region_index} lost triangles: "
            f"faces={face_count}, triangles={triangle_count}",
        )
        _assert(
            len(projection.loop_to_attachment_index) == len(snapshot.loops),
            f"{label} region {region_index} lost loop-to-attachment ownership: "
            f"loops={len(snapshot.loops)}, "
            f"mapped={len(projection.loop_to_attachment_index)}",
        )

        region_collapsed = _snapshot_collapsed_face_indices(snapshot)
        projection_collapsed = _projection_collapsed_triangle_indices(projection)
        _assert(
            len(projection_collapsed) == len(region_collapsed),
            f"{label} region {region_index} changed setup-degenerate triangle count: "
            f"geometry={region_collapsed}, attachment={projection_collapsed}",
        )
        collapsed_faces += len(region_collapsed)
        collapsed_triangles += len(projection_collapsed)
        if face_count > 0 and len(region_collapsed) == face_count:
            fully_edge_on += 1

    return fully_edge_on, collapsed_faces, collapsed_triangles


def _run(expected_blend: str) -> None:
    loaded = _require_loaded_blend(expected_blend)
    source = _require_source_object()
    _assert(
        bpy.context.scene.camera is not None,
        "real coin side-segment gate requires an active scene camera",
    )

    scene_before = _scene_fingerprint()
    bake_before = _capture_scene_bake_state()
    object_before = _object_fingerprint(source)
    datablocks_before = _datablock_fingerprint()
    temporary_before = _temporary_datablock_names()

    with tempfile.TemporaryDirectory(
        prefix="spine2d-coin-normal-side-segments-"
    ) as directory:
        root = Path(directory)
        axis = prepare_a1_object(
            source,
            _settings(root / "axis", A1ProjectionDirection.POSITIVE_Z),
            context=bpy.context,
            scene=bpy.context.scene,
        )
        camera = prepare_a1_object(
            source,
            _settings(root / "camera", A1ProjectionDirection.ACTIVE_CAMERA),
            context=bpy.context,
            scene=bpy.context.scene,
        )

        axis_edge_on, axis_collapsed_faces, axis_collapsed_triangles = (
            _assert_region_projection_parity(axis, label="axis")
        )
        camera_edge_on, camera_collapsed_faces, camera_collapsed_triangles = (
            _assert_region_projection_parity(camera, label="active-camera")
        )

        _assert(
            axis_edge_on > 0,
            "real coin axis projection contains no fully edge-on side regions; "
            "the regression fixture no longer exercises the reported bug",
        )
        _assert(
            axis_collapsed_faces > 0,
            "real coin axis projection contains no setup-degenerate side faces",
        )
        _assert(
            axis_collapsed_triangles == axis_collapsed_faces,
            "axis attachment assembly changed setup-degenerate face ownership",
        )
        _assert(
            len(axis.uv_regions.snapshots) == len(camera.uv_regions.snapshots),
            "Normal projection directions produced different prepared region counts: "
            f"axis={len(axis.uv_regions.snapshots)}, "
            f"camera={len(camera.uv_regions.snapshots)}",
        )

        print(
            "[COIN-NORMAL-SIDE-SEGMENTS] PASS "
            f"blend={loaded} object={source.name_full!r} "
            f"regions={len(axis.uv_regions.snapshots)} "
            f"axis_edge_on_regions={axis_edge_on} "
            f"axis_collapsed_faces={axis_collapsed_faces} "
            f"axis_collapsed_triangles={axis_collapsed_triangles} "
            f"camera_edge_on_regions={camera_edge_on} "
            f"camera_collapsed_faces={camera_collapsed_faces} "
            f"camera_collapsed_triangles={camera_collapsed_triangles} "
            "ownership=all-regions-all-triangles",
            flush=True,
        )

    _assert(_scene_fingerprint() == scene_before, "side-segment gate changed Blender context")
    _assert(_capture_scene_bake_state() == bake_before, "side-segment gate changed bake state")
    _assert(_object_fingerprint(source) == object_before, "side-segment gate changed source data")
    _assert(
        _datablock_fingerprint() == datablocks_before,
        "side-segment gate created or removed persistent Blender datablocks",
    )
    _assert(
        _temporary_datablock_names() == temporary_before,
        "side-segment gate leaked temporary Blender datablocks",
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
