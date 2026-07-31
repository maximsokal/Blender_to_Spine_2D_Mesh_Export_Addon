"""Validate Blender Object Origin placement through production preparation.

The worker creates real Blender Mesh objects with geometry above, below, and across
local Object Origin. It exercises single-object and public standalone multi-object
preparation without writing final export files.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import traceback
from typing import Iterable

import bpy


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIRECTORY.parents[1]
for path in (SCRIPT_DIRECTORY, REPOSITORY_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from Blender_to_Spine2D_Mesh_Exporter.application import (  # noqa: E402
    A1MultiObjectExportSettings,
    A1MultiObjectMode,
    A1SingleObjectExportSettings,
    A1SourceGeometryMode,
    ExportSettings,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter import (  # noqa: E402
    A1MultiObjectSource,
    prepare_a1_multi_object,
    prepare_a1_object,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import (  # noqa: E402
    BakeExecutionSettings,
    BakeMode,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.legacy_rig_contracts import (  # noqa: E402
    LegacyZGroupOriginMode,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.rig_profiles import (  # noqa: E402
    A1RigProfile,
    A1RigSetupPoseMode,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.version_target import (  # noqa: E402
    SpineJsonTarget,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.uv import UvUnwrapSettings  # noqa: E402
from run_bake_integration import (  # noqa: E402
    _activate_only,
    _assert,
    _clear_scene,
    _configure_cycles_scene,
    _create_emission_material,
)


_TEXTURE_SIZE = 32
_EXPECTED_SCALE = float(_TEXTURE_SIZE)
_MATRIX_ABSOLUTE_TOLERANCE = 1.0e-7


@dataclass(frozen=True)
class _SourceObjectState:
    """Authored source state captured after Blender dependency-graph evaluation."""

    vertex_coordinates: tuple[tuple[float, float, float], ...]
    matrix_world: tuple[float, ...]
    location: tuple[float, float, float]
    rotation_mode: str
    rotation_euler: tuple[float, float, float]
    rotation_quaternion: tuple[float, float, float, float]
    scale: tuple[float, float, float]
    parent_name: str | None


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else ()
    return parser.parse_args(arguments)


def _prepare_output_directory(value: Path) -> Path:
    if not isinstance(value, Path):
        raise TypeError("output must be pathlib.Path")
    resolved = value.expanduser().resolve(strict=False)
    if resolved.exists() and not resolved.is_dir():
        raise ValueError(f"Output path is not a directory: {resolved}")
    if resolved.exists() and any(resolved.iterdir()):
        raise ValueError(f"Output directory must be empty: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _synchronize_view_layer() -> None:
    """Force pending Object transform edits into matrix_world before assertions."""

    view_layer = getattr(bpy.context, "view_layer", None)
    update = getattr(view_layer, "update", None)
    if not callable(update):
        raise RuntimeError("Blender context has no callable view_layer.update()")
    update()


def _matrix_tuple(matrix: object) -> tuple[float, ...]:
    try:
        return tuple(
            float(matrix[row][column])
            for row in range(4)
            for column in range(4)
        )
    except Exception as exc:
        raise ValueError("Unable to read a finite 4x4 Blender matrix") from exc


def _capture_source_state(source_object: bpy.types.Object) -> _SourceObjectState:
    if source_object is None or source_object.type != "MESH":
        raise ValueError("source_object must be a Blender MESH object")
    if source_object.data is None:
        raise ValueError("source_object.data is missing")

    return _SourceObjectState(
        vertex_coordinates=tuple(
            tuple(float(component) for component in vertex.co)
            for vertex in source_object.data.vertices
        ),
        matrix_world=_matrix_tuple(source_object.matrix_world),
        location=tuple(float(value) for value in source_object.location),
        rotation_mode=str(source_object.rotation_mode),
        rotation_euler=tuple(float(value) for value in source_object.rotation_euler),
        rotation_quaternion=tuple(
            float(value) for value in source_object.rotation_quaternion
        ),
        scale=tuple(float(value) for value in source_object.scale),
        parent_name=(
            None
            if source_object.parent is None
            else str(source_object.parent.name_full or source_object.parent.name)
        ),
    )


def _assert_source_state_unchanged(
    source_object: bpy.types.Object,
    before: _SourceObjectState,
    *,
    label: str,
) -> float:
    """Reject source mutation while allowing only matrix float round-off."""

    if not isinstance(before, _SourceObjectState):
        raise TypeError("before must be _SourceObjectState")
    _synchronize_view_layer()
    after = _capture_source_state(source_object)

    _assert(
        after.vertex_coordinates == before.vertex_coordinates,
        f"{label} mutated source vertex coordinates",
    )
    _assert(after.location == before.location, f"{label} mutated Object.location")
    _assert(
        after.rotation_mode == before.rotation_mode,
        f"{label} mutated Object.rotation_mode",
    )
    _assert(
        after.rotation_euler == before.rotation_euler,
        f"{label} mutated Object.rotation_euler",
    )
    _assert(
        after.rotation_quaternion == before.rotation_quaternion,
        f"{label} mutated Object.rotation_quaternion",
    )
    _assert(after.scale == before.scale, f"{label} mutated Object.scale")
    _assert(after.parent_name == before.parent_name, f"{label} mutated Object.parent")

    deltas = tuple(
        abs(actual - expected)
        for expected, actual in zip(
            before.matrix_world,
            after.matrix_world,
            strict=True,
        )
    )
    maximum_delta = max(deltas, default=0.0)
    _assert(
        maximum_delta <= _MATRIX_ABSOLUTE_TOLERANCE,
        f"{label} mutated matrix_world: max_delta={maximum_delta:.12g}, "
        f"tolerance={_MATRIX_ABSOLUTE_TOLERANCE:.12g}, "
        f"before={before.matrix_world}, after={after.matrix_world}",
    )
    return maximum_delta


def _canonical_z_values(values: Iterable[float]) -> tuple[float, ...]:
    resolved = tuple(sorted({float(round(float(value), 4)) for value in values}))
    if not resolved:
        raise ValueError("At least one local Z value is required")
    return tuple(0.0 if value == 0.0 else value for value in resolved)


def _create_layered_mesh(
    name: str,
    local_z_values: tuple[float, ...],
    *,
    location: tuple[float, float, float],
) -> bpy.types.Object:
    """Create independent visible quads at exact object-local Z layers."""

    if not isinstance(name, str) or not name.strip():
        raise ValueError("name must be a non-empty string")
    z_values = _canonical_z_values(local_z_values)
    if not isinstance(location, tuple) or len(location) != 3:
        raise ValueError("location must contain three values")

    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int, int]] = []
    for layer_index, z_value in enumerate(z_values):
        center_x = float(layer_index) * 0.15
        first = len(vertices)
        vertices.extend(
            (
                (center_x - 0.5, -0.5, z_value),
                (center_x + 0.5, -0.5, z_value),
                (center_x + 0.5, 0.5, z_value),
                (center_x - 0.5, 0.5, z_value),
            )
        )
        faces.append((first, first + 1, first + 2, first + 3))

    mesh = bpy.data.meshes.new(f"{name}Mesh")
    mesh.from_pydata(vertices, (), faces)
    mesh.update(calc_edges=True)
    source_object = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(source_object)
    source_object.location = tuple(float(value) for value in location)
    _create_emission_material(source_object)
    return source_object


def _settings(
    output_directory: Path,
    *,
    prefix: str,
) -> A1SingleObjectExportSettings:
    return A1SingleObjectExportSettings(
        export=ExportSettings(
            texture_width=_TEXTURE_SIZE,
            texture_height=_TEXTURE_SIZE,
            output_directory=output_directory,
            images_relative_path="images",
            spine_version=SpineJsonTarget.SPINE_4_2.exact_version,
            rig_profile=A1RigProfile.TWO_AXIS_ROTATION_SCALE.value,
            bake_margin=1,
        ),
        prefix=prefix,
        output_stem=prefix,
        source_geometry_mode=A1SourceGeometryMode.ORIGINAL,
        uv=UvUnwrapSettings(layer_name="SpineBakeUV"),
        diffuse_mode=BakeMode.EMIT,
        procedural_mode=BakeMode.EMIT,
        bake_execution=BakeExecutionSettings(samples=1),
        rig_setup_pose_mode=A1RigSetupPoseMode.PRESERVE_COMPOSITION,
    )


def _bone_by_name(prepared: object) -> dict[str, object]:
    return {
        bone.name: bone
        for bone in prepared.document_assembly.document.bones
    }


def _assert_prepared_origin(
    prepared: object,
    *,
    expected_location_xy: tuple[float, float],
    expected_local_z: tuple[float, ...],
) -> dict[str, object]:
    rig = prepared.rig
    _assert(
        rig.request.z_group_origin_mode is LegacyZGroupOriginMode.OBJECT_ORIGIN,
        f"{prepared.object_id} did not select OBJECT_ORIGIN",
    )
    _assert(
        rig.request.setup_pose_mode is A1RigSetupPoseMode.PRESERVE_COMPOSITION,
        f"{prepared.object_id} did not preserve the per-object main bone",
    )

    bones = _bone_by_name(prepared)
    main = bones[rig.info.main_bone_name]
    expected_main = (
        round(float(expected_location_xy[0]) * _EXPECTED_SCALE, 2),
        round(float(expected_location_xy[1]) * _EXPECTED_SCALE, 2),
    )
    actual_main = (
        0.0 if main.x is None else float(main.x),
        0.0 if main.y is None else float(main.y),
    )
    _assert(
        actual_main == expected_main,
        f"{prepared.object_id} main pivot mismatch: "
        f"expected={expected_main}, actual={actual_main}",
    )

    canonical_z = _canonical_z_values(expected_local_z)
    _assert(
        tuple(group.z_value for group in rig.info.z_groups) == canonical_z,
        f"{prepared.object_id} Z-group identity changed",
    )
    expected_offsets = tuple(
        round(float(value) * _EXPECTED_SCALE, 2) for value in canonical_z
    )
    actual_offsets = tuple(
        float(bones[group.scale_bone_name].y or 0.0)
        for group in rig.info.z_groups
    )
    _assert(
        actual_offsets == expected_offsets,
        f"{prepared.object_id} signed depth mismatch: "
        f"expected={expected_offsets}, actual={actual_offsets}",
    )
    if 0.0 not in canonical_z:
        _assert(
            0.0 not in tuple(group.z_value for group in rig.info.z_groups),
            f"{prepared.object_id} created an artificial zero Z group",
        )

    return {
        "objectId": prepared.object_id,
        "mainBone": rig.info.main_bone_name,
        "mainPosition": list(actual_main),
        "zValues": list(canonical_z),
        "depthOffsets": list(actual_offsets),
        "originMode": rig.request.z_group_origin_mode.value,
        "setupPoseMode": rig.request.setup_pose_mode.value,
    }


def _single_case(output_root: Path) -> dict[str, object]:
    source_object = _create_layered_mesh(
        "PivotSingle",
        (-1.0, 0.0, 2.0),
        location=(1.25, -0.75, 9.0),
    )
    _activate_only(source_object)
    _synchronize_view_layer()
    state_before = _capture_source_state(source_object)

    prepared = prepare_a1_object(
        source_object,
        _settings(output_root, prefix="PivotSingle"),
        context=bpy.context,
        scene=bpy.context.scene,
    )
    maximum_matrix_delta = _assert_source_state_unchanged(
        source_object,
        state_before,
        label="Single preparation",
    )

    result = _assert_prepared_origin(
        prepared,
        expected_location_xy=(1.25, -0.75),
        expected_local_z=(-1.0, 0.0, 2.0),
    )
    result["sourceUnchanged"] = True
    result["maximumMatrixDelta"] = maximum_matrix_delta
    return result


def _standalone_case(output_root: Path) -> dict[str, object]:
    specifications = (
        ("PivotAcross", (-2.0, 0.0, 1.0), (2.0, 1.0, 3.0)),
        ("PivotBelow", (1.0, 2.0, 4.0), (-1.5, 2.25, -6.0)),
        ("PivotAbove", (-4.0, -2.0, -1.0), (0.5, -2.0, 12.0)),
    )
    sources: list[A1MultiObjectSource] = []
    by_prefix: dict[
        str,
        tuple[
            tuple[float, ...],
            tuple[float, float, float],
            bpy.types.Object,
            _SourceObjectState,
        ],
    ] = {}

    for index, (prefix, z_values, location) in enumerate(specifications, start=1):
        source_object = _create_layered_mesh(
            prefix,
            tuple(float(value) for value in z_values),
            location=tuple(float(value) for value in location),
        )
        _activate_only(source_object)
        _synchronize_view_layer()
        state_before = _capture_source_state(source_object)
        sources.append(
            A1MultiObjectSource(
                source_object=source_object,
                component_id=f"pivot_component_{index}",
                animation_namespace=f"pivot_object_{index}",
                settings=_settings(output_root, prefix=prefix),
            )
        )
        by_prefix[prefix] = (
            tuple(float(value) for value in z_values),
            tuple(float(value) for value in location),
            source_object,
            state_before,
        )

    prepared = prepare_a1_multi_object(
        tuple(sources),
        A1MultiObjectExportSettings(
            output_directory=output_root,
            output_stem="object_origin_standalone",
            mode=A1MultiObjectMode.STANDALONE,
            anchor_component_id=None,
        ),
        context=bpy.context,
        scene=bpy.context.scene,
    )
    _assert(len(prepared.objects) == 3, "Standalone preparation lost an object")

    objects: list[dict[str, object]] = []
    for item in prepared.objects:
        z_values, location, source_object, state_before = by_prefix[item.prefix]
        maximum_matrix_delta = _assert_source_state_unchanged(
            source_object,
            state_before,
            label=f"Standalone preparation for {item.prefix}",
        )
        result = _assert_prepared_origin(
            item,
            expected_location_xy=(location[0], location[1]),
            expected_local_z=z_values,
        )
        result["sourceUnchanged"] = True
        result["maximumMatrixDelta"] = maximum_matrix_delta
        objects.append(result)

    _assert(
        len({tuple(item["mainPosition"]) for item in objects}) == len(objects),
        "Standalone objects did not preserve independent pivots",
    )
    return {
        "mode": prepared.settings.mode.value,
        "objectCount": len(objects),
        "objects": objects,
    }


def run(output_directory: Path) -> Path:
    output_root = _prepare_output_directory(output_directory)
    _clear_scene()
    _configure_cycles_scene()
    report: dict[str, object] = {
        "status": "passed",
        "blenderVersion": bpy.app.version_string,
        "matrixAbsoluteTolerance": _MATRIX_ABSOLUTE_TOLERANCE,
        "single": _single_case(output_root),
    }

    _clear_scene()
    _configure_cycles_scene()
    report["standalone"] = _standalone_case(output_root)

    report_path = output_root / "object_origin_pivot_acceptance.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report_path


def main() -> None:
    arguments = _parse_arguments()
    print(f"Blender version: {bpy.app.version_string}")
    print("[OBJECT_ORIGIN_PIVOT] RUN single + standalone production preparation")
    report_path = run(arguments.output)
    print(f"[OBJECT_ORIGIN_PIVOT] REPORT {report_path}")
    print("[OBJECT_ORIGIN_PIVOT] PASS")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
