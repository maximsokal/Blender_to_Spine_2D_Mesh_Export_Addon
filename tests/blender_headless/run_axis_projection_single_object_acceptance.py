"""Validate six signed-axis projections through real Blender source preparation."""

from __future__ import annotations

import argparse
import json
from math import isclose
from pathlib import Path
import sys
import traceback

import bpy
from mathutils import Vector


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIRECTORY.parents[1]
for path in (SCRIPT_DIRECTORY, REPOSITORY_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from Blender_to_Spine2D_Mesh_Exporter.application import (  # noqa: E402
    A1SingleObjectExportSettings,
    A1SourceGeometryMode,
    ExportSettings,
    calculate_a1_main_position_pixels,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.a1_source_geometry_preparation import (  # noqa: E402
    prepare_a1_source_geometry,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.projection import (  # noqa: E402
    A1ProjectionDirection,
    resolve_a1_axis_projection_basis,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.legacy_rig_contracts import (  # noqa: E402
    UniformScaleMode,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.legacy_rig_scale import (  # noqa: E402
    calculate_uniform_scale,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.rig_profiles import (  # noqa: E402
    A1RigProfile,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.version_target import (  # noqa: E402
    SpineJsonTarget,
)


_TEXTURE_WIDTH = 64
_TEXTURE_HEIGHT = 32
_ABSOLUTE_TOLERANCE = 1.0e-7
_LOCAL_VERTICES = (
    (0.0, 0.0, 0.0),
    (1.0, 0.0, 1.0),
    (0.0, 1.0, 2.0),
)
_AXIS_DIRECTIONS = tuple(
    direction
    for direction in A1ProjectionDirection
    if direction is not A1ProjectionDirection.ACTIVE_CAMERA
)


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else ()
    return parser.parse_args(arguments)


def _prepare_output_directory(value: Path) -> Path:
    resolved = value.expanduser().resolve(strict=False)
    if resolved.exists() and not resolved.is_dir():
        raise ValueError(f"Output path is not a directory: {resolved}")
    if resolved.exists() and any(resolved.iterdir()):
        raise ValueError(f"Output directory must be empty: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _clear_scene() -> None:
    if bpy.context.object is not None and bpy.context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for mesh in tuple(bpy.data.meshes):
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)


def _activate_only(source_object: bpy.types.Object) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    source_object.select_set(True)
    bpy.context.view_layer.objects.active = source_object


def _matrix_tuple(matrix: object) -> tuple[float, ...]:
    return tuple(
        float(matrix[row][column])
        for row in range(4)
        for column in range(4)
    )


def _create_source_object() -> bpy.types.Object:
    mesh = bpy.data.meshes.new("AxisProjectionMesh")
    mesh.from_pydata(_LOCAL_VERTICES, (), ((0, 1, 2),))
    mesh.update(calc_edges=True)

    source_object = bpy.data.objects.new("AxisProjectionObject", mesh)
    bpy.context.scene.collection.objects.link(source_object)
    source_object.location = (3.25, -2.5, 4.75)
    source_object.rotation_euler = (0.31, -0.42, 0.27)
    source_object.scale = (1.5, 0.75, 2.0)
    _activate_only(source_object)
    bpy.context.view_layer.update()
    return source_object


def _settings(
    output_directory: Path,
    direction: A1ProjectionDirection,
) -> A1SingleObjectExportSettings:
    return A1SingleObjectExportSettings(
        export=ExportSettings(
            texture_width=_TEXTURE_WIDTH,
            texture_height=_TEXTURE_HEIGHT,
            output_directory=output_directory,
            images_relative_path="images",
            spine_version=SpineJsonTarget.SPINE_4_2.exact_version,
            rig_profile=A1RigProfile.TWO_AXIS_ROTATION_SCALE.value,
            bake_margin=1,
        ),
        prefix="AxisProjection",
        output_stem=f"axis_{direction.value.lower()}",
        source_geometry_mode=A1SourceGeometryMode.ORIGINAL,
        rig_scale_mode=UniformScaleMode.AVERAGE,
        projection_direction=direction,
    )


def _assert_close_tuple(
    actual: tuple[float, ...],
    expected: tuple[float, ...],
    *,
    label: str,
) -> None:
    if len(actual) != len(expected):
        raise AssertionError(
            f"{label} length mismatch: actual={len(actual)}, expected={len(expected)}"
        )
    mismatches = tuple(
        (index, actual_value, expected_value)
        for index, (actual_value, expected_value) in enumerate(
            zip(actual, expected, strict=True)
        )
        if not isclose(
            actual_value,
            expected_value,
            rel_tol=0.0,
            abs_tol=_ABSOLUTE_TOLERANCE,
        )
    )
    if mismatches:
        raise AssertionError(f"{label} mismatch: {mismatches}")


def _canonical_depth_values(
    expected_positions: tuple[tuple[float, float, float], ...],
) -> tuple[float, ...]:
    values = tuple(
        0.0 if round(position[2], 4) == 0.0 else float(round(position[2], 4))
        for position in expected_positions
    )
    return tuple(sorted(set(values)))


def _run_direction(
    source_object: bpy.types.Object,
    output_directory: Path,
    direction: A1ProjectionDirection,
) -> dict[str, object]:
    bpy.context.view_layer.update()
    matrix_world = source_object.matrix_world.copy()
    source_matrix_before = _matrix_tuple(matrix_world)
    source_vertices_before = tuple(
        tuple(float(component) for component in vertex.co)
        for vertex in source_object.data.vertices
    )

    basis = resolve_a1_axis_projection_basis(direction)
    world_origin = tuple(float(value) for value in matrix_world.translation)
    projected_origin = basis.project_point(world_origin)
    expected_positions = tuple(
        basis.project_point(tuple(float(value) for value in (matrix_world @ Vector(point))))
        .relative_to(projected_origin)
        .canonical_position
        for point in _LOCAL_VERTICES
    )

    settings = _settings(output_directory, direction)
    prepared = prepare_a1_source_geometry(
        source_object,
        settings,
        scene=bpy.context.scene,
    )

    actual_positions = tuple(
        tuple(float(value) for value in vertex.position)
        for vertex in prepared.source_snapshot.vertices
    )
    for index, (actual, expected) in enumerate(
        zip(actual_positions, expected_positions, strict=True)
    ):
        _assert_close_tuple(
            actual,
            expected,
            label=f"{direction.value} vertex[{index}]",
        )

    actual_origin = (
        float(prepared.source_snapshot.world_matrix[3]),
        float(prepared.source_snapshot.world_matrix[7]),
        float(prepared.source_snapshot.world_matrix[11]),
    )
    _assert_close_tuple(
        actual_origin,
        projected_origin.canonical_position,
        label=f"{direction.value} projected origin",
    )

    actual_depth_values = tuple(group.z_value for group in prepared.z_groups.groups)
    expected_depth_values = _canonical_depth_values(expected_positions)
    if actual_depth_values != expected_depth_values:
        raise AssertionError(
            f"{direction.value} depth groups mismatch: "
            f"actual={actual_depth_values}, expected={expected_depth_values}"
        )

    uniform_scale = calculate_uniform_scale(
        _TEXTURE_WIDTH,
        _TEXTURE_HEIGHT,
        UniformScaleMode.AVERAGE,
    )
    expected_main = (
        projected_origin.u * uniform_scale,
        projected_origin.v * uniform_scale,
    )
    actual_main = calculate_a1_main_position_pixels(
        prepared.source_snapshot,
        settings,
    )
    if actual_main is None:
        raise AssertionError(f"{direction.value} did not produce a main position")
    _assert_close_tuple(
        actual_main,
        expected_main,
        label=f"{direction.value} main position",
    )

    bpy.context.view_layer.update()
    source_matrix_after = _matrix_tuple(source_object.matrix_world)
    source_vertices_after = tuple(
        tuple(float(component) for component in vertex.co)
        for vertex in source_object.data.vertices
    )
    if source_matrix_after != source_matrix_before:
        raise AssertionError(f"{direction.value} mutated source matrix_world")
    if source_vertices_after != source_vertices_before:
        raise AssertionError(f"{direction.value} mutated source mesh vertices")

    return {
        "direction": direction.value,
        "projectedOrigin": list(actual_origin),
        "mainPositionPixels": list(actual_main),
        "depthGroups": list(actual_depth_values),
        "vertexPositions": [list(position) for position in actual_positions],
        "sourceUnchanged": True,
        "projectionApplied": bool(prepared.statistics["axis_projection_applied"]),
    }


def run(output_directory: Path) -> Path:
    output_root = _prepare_output_directory(output_directory)
    _clear_scene()
    source_object = _create_source_object()

    directions = tuple(
        _run_direction(source_object, output_root, direction)
        for direction in _AXIS_DIRECTIONS
    )
    positive_z = next(
        item
        for item in directions
        if item["direction"] == A1ProjectionDirection.POSITIVE_Z.value
    )
    if positive_z["projectionApplied"]:
        raise AssertionError("POSITIVE_Z must remain the exact compatibility path")
    if not all(
        item["projectionApplied"]
        for item in directions
        if item["direction"] != A1ProjectionDirection.POSITIVE_Z.value
    ):
        raise AssertionError("Every non-default signed axis must apply projection")

    report = {
        "status": "passed",
        "blenderVersion": bpy.app.version_string,
        "absoluteTolerance": _ABSOLUTE_TOLERANCE,
        "directionCount": len(directions),
        "directions": list(directions),
    }
    report_path = output_root / "axis_projection_single_object_acceptance.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report_path


def main() -> None:
    arguments = _parse_arguments()
    print(f"Blender version: {bpy.app.version_string}")
    print("[AXIS_PROJECTION_SINGLE] RUN six signed-axis projections")
    report_path = run(arguments.output)
    print(f"[AXIS_PROJECTION_SINGLE] REPORT {report_path}")
    print("[AXIS_PROJECTION_SINGLE] PASS")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
