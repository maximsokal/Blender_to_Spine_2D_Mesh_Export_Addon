"""Real grenade.blend gate for the public selected-object Shared Pivot route.

The runner deliberately uses the artist-authored selection stored in grenade.blend. It
registers the Rewrite extension, routes through ``build_selected_ui_export_plan`` and
``export_selected_objects_a1``, and performs only one production bake/export.

Before export it independently resolves the aggregate exported-world-geometry AABB center.
After export it verifies that every selected object's Spine main bone uses that same
projected pivot, that X/Y controls still exist, that production JSON/PNG files are real,
and that Blender source geometry, transforms, selection, active object, cursor, frame,
and datablock namespaces were not changed by the export transaction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback

import bpy


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIRECTORY.parents[1]
for path in (SCRIPT_DIRECTORY, REPOSITORY_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import Blender_to_Spine2D_Mesh_Exporter as addon  # noqa: E402
from Blender_to_Spine2D_Mesh_Exporter.application import (  # noqa: E402
    resolve_a1_names,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.a1_shared_pivot import (  # noqa: E402
    resolve_a1_shared_pivot_world,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.a1_ui_export_plan import (  # noqa: E402
    build_selected_ui_export_plan,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.a1_ui_router import (  # noqa: E402
    export_selected_objects_a1,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import (  # noqa: E402
    A1TextureExportMode,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.projection import (  # noqa: E402
    A1ProjectionDirection,
    resolve_a1_axis_projection_basis,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine import (  # noqa: E402
    calculate_uniform_scale,
)
from run_bake_integration import PNG_SIGNATURE, _assert  # noqa: E402
from run_grenade_bump_displacement_normal_uv_integration import (  # noqa: E402
    _require_loaded_blend,
)


_MINIMUM_SELECTED_MESH_OBJECTS = 2
_POSITION_TOLERANCE = 1.0e-5


def _parse_arguments() -> argparse.Namespace:
    arguments = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expected-blend",
        required=True,
        help="Exact grenade.blend path Blender must already have loaded.",
    )
    parser.add_argument(
        "--output-directory",
        required=True,
        help="Empty persistent directory for production JSON and PNG outputs.",
    )
    return parser.parse_args(arguments)


def _register_steps() -> list[tuple]:
    completed: list[tuple] = []
    try:
        for step in addon.REGISTRATION_STEPS:
            step[1]()
            completed.append(step)
        return completed
    except Exception:
        for step in reversed(completed):
            try:
                step[2]()
            except Exception:
                traceback.print_exc()
        raise


def _unregister_steps(completed: list[tuple]) -> None:
    failures: list[str] = []
    for label, _register, unregister in reversed(completed):
        try:
            unregister()
        except Exception as exc:
            failures.append(f"{label}: {exc}")
    _assert(not failures, f"Rewrite unregister failures: {failures!r}")


def _prepare_output_directory(value: str) -> Path:
    output = Path(value).expanduser().resolve(strict=False)
    if output.exists() and not output.is_dir():
        raise ValueError(f"Output path is not a directory: {output}")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def _matrix_tuple(matrix: object) -> tuple[float, ...]:
    return tuple(
        float(matrix[row][column])
        for row in range(4)
        for column in range(4)
    )


def _selected_meshes(context) -> tuple:
    selected = tuple(
        obj
        for obj in getattr(context, "selected_objects", ())
        if getattr(obj, "type", None) == "MESH"
        and getattr(obj, "data", None) is not None
    )
    _assert(
        len(selected) >= _MINIMUM_SELECTED_MESH_OBJECTS,
        "grenade shared-pivot gate requires at least two artist-selected Mesh objects; "
        f"selected_mesh_count={len(selected)}",
    )
    active = getattr(context.view_layer.objects, "active", None)
    _assert(
        active in selected,
        "active Blender object must be one of the selected Mesh export objects",
    )
    return selected


def _object_fingerprint(obj) -> tuple:
    mesh = obj.data
    uv_layers = getattr(mesh, "uv_layers", None)
    return (
        obj.name_full,
        mesh.name_full,
        _matrix_tuple(obj.matrix_world),
        tuple(float(value) for value in obj.location),
        tuple(float(value) for value in obj.rotation_euler),
        tuple(float(value) for value in obj.scale),
        None if obj.parent is None else obj.parent.name_full,
        tuple(tuple(float(value) for value in vertex.co) for vertex in mesh.vertices),
        tuple(tuple(int(index) for index in polygon.vertices) for polygon in mesh.polygons),
        tuple(layer.name for layer in uv_layers) if uv_layers is not None else (),
        (
            None
            if uv_layers is None or getattr(uv_layers, "active", None) is None
            else uv_layers.active.name
        ),
        tuple(
            None if slot.material is None else slot.material.name_full
            for slot in obj.material_slots
        ),
    )


def _scene_fingerprint(scene, selected: tuple) -> tuple:
    active = getattr(bpy.context.view_layer.objects, "active", None)
    cursor = getattr(scene, "cursor", None)
    return (
        tuple(_object_fingerprint(obj) for obj in selected),
        tuple(obj.name_full for obj in bpy.context.selected_objects),
        None if active is None else active.name_full,
        None if cursor is None else tuple(float(value) for value in cursor.location),
        int(scene.frame_current),
        None if scene.camera is None else scene.camera.name_full,
        str(scene.render.engine),
        str(getattr(scene, "spine2d_json_path", "")),
        bool(getattr(scene, "spine2d_shared_selection_pivot", False)),
    )


def _datablock_fingerprint() -> tuple:
    return (
        tuple(sorted(item.name_full for item in bpy.data.objects)),
        tuple(sorted(item.name_full for item in bpy.data.meshes)),
        tuple(sorted(item.name_full for item in bpy.data.materials)),
        tuple(sorted(item.name_full for item in bpy.data.images)),
        tuple(sorted(item.name_full for item in bpy.data.collections)),
    )


def _bone_position(bone: dict[str, object]) -> tuple[float, float]:
    return float(bone.get("x", 0.0)), float(bone.get("y", 0.0))


def _assert_close_pair(
    actual: tuple[float, float],
    expected: tuple[float, float],
    *,
    label: str,
) -> None:
    deltas = tuple(
        abs(float(actual[index]) - float(expected[index])) for index in range(2)
    )
    _assert(
        max(deltas, default=0.0) <= _POSITION_TOLERANCE,
        f"{label} mismatch: expected={expected}, actual={actual}, deltas={deltas}",
    )


def _assert_production_outputs(
    result,
    *,
    selected_count: int,
    prefixes: tuple[str, ...],
    expected_main_position: tuple[float, float],
) -> tuple[Path, tuple[Path, ...]]:
    _assert(
        bool(result.success),
        "real shared-pivot export failed: "
        f"issues={result.issues!r}, statistics={dict(result.statistics)!r}",
    )
    outputs = tuple(Path(path).resolve(strict=False) for path in result.output_files)
    json_files = tuple(path for path in outputs if path.suffix.lower() == ".json")
    png_files = tuple(path for path in outputs if path.suffix.lower() == ".png")
    _assert(len(json_files) == 1, f"expected one production JSON: {json_files!r}")
    _assert(
        len(png_files) >= selected_count,
        "production export emitted fewer PNGs than selected Mesh objects: "
        f"selected={selected_count}, png={len(png_files)}",
    )
    for path in outputs:
        _assert(path.is_file() and path.stat().st_size > 8, f"missing/empty output: {path}")
    for path in png_files:
        _assert(path.read_bytes().startswith(PNG_SIGNATURE), f"output is not PNG: {path}")

    document = json.loads(json_files[0].read_text(encoding="utf-8"))
    _assert(isinstance(document, dict), "production JSON root must be mapping")
    bones = document.get("bones")
    _assert(isinstance(bones, list) and bones, "production JSON contains no bones")
    bone_by_name = {
        str(bone.get("name")): bone
        for bone in bones
        if isinstance(bone, dict) and isinstance(bone.get("name"), str)
    }

    main_positions: list[tuple[float, float]] = []
    for prefix in prefixes:
        main_name = f"{prefix}_main"
        x_control_name = f"{prefix}_rotation_X"
        y_control_name = f"{prefix}_rotation_Y"
        _assert(main_name in bone_by_name, f"missing shared-pivot main bone: {main_name}")
        _assert(x_control_name in bone_by_name, f"missing X control bone: {x_control_name}")
        _assert(y_control_name in bone_by_name, f"missing Y control bone: {y_control_name}")
        position = _bone_position(bone_by_name[main_name])
        _assert_close_pair(
            position,
            expected_main_position,
            label=f"{main_name} shared pivot",
        )
        main_positions.append(position)

    first = main_positions[0]
    for prefix, position in zip(prefixes[1:], main_positions[1:], strict=True):
        _assert_close_pair(
            position,
            first,
            label=f"{prefix}_main equality",
        )

    statistics = dict(result.statistics)
    _assert(
        int(statistics.get("shared_pivot_enabled", 0)) == 1,
        f"production statistics did not report shared pivot: {statistics!r}",
    )
    _assert(
        int(statistics.get("shared_pivot_vertex_count", 0)) > 0,
        f"production statistics lost shared pivot geometry count: {statistics!r}",
    )
    return json_files[0], outputs


def _run(expected_blend: str, output_directory_arg: str) -> None:
    loaded = _require_loaded_blend(expected_blend)
    output_directory = _prepare_output_directory(output_directory_arg)

    completed = _register_steps()
    try:
        scene = bpy.context.scene
        selected = _selected_meshes(bpy.context)

        # A newly introduced RNA property must be default-on even for an existing .blend
        # that has never stored a value for it.
        _assert(
            bool(getattr(scene, "spine2d_shared_selection_pivot", False)),
            "Shared Selection Pivot RNA default must be enabled",
        )

        original_output_path = str(getattr(scene, "spine2d_json_path", ""))
        original_shared_pivot = bool(
            getattr(scene, "spine2d_shared_selection_pivot", True)
        )
        try:
            scene.spine2d_json_path = str(output_directory)
            scene.spine2d_shared_selection_pivot = True

            before = _scene_fingerprint(scene, selected)
            datablocks_before = _datablock_fingerprint()

            plan = build_selected_ui_export_plan(bpy.context)
            _assert(
                plan.settings.shared_pivot_enabled,
                "public selected-object UI plan did not enable Shared Pivot",
            )
            _assert(
                len(plan.standalone_sources) == len(selected),
                "UI plan source count differs from selected Mesh object count",
            )
            first_settings = plan.standalone_sources[0].settings
            _assert(
                first_settings.bake_execution.texture_export_mode
                is A1TextureExportMode.NORMAL_UV_SEGMENTS,
                "real shared-pivot gate requires persisted Normal / UV Segments; "
                f"actual={first_settings.bake_execution.texture_export_mode.value!r}",
            )
            direction = first_settings.projection_direction
            _assert(
                isinstance(direction, A1ProjectionDirection) and direction.axis_aligned,
                "real shared-pivot gate requires one of +X/-X/+Y/-Y/+Z/-Z; "
                f"actual={direction!r}",
            )

            resolution = resolve_a1_shared_pivot_world(
                plan.standalone_sources,
                scene=scene,
            )
            basis = resolve_a1_axis_projection_basis(direction)
            projected_pivot = basis.project_point(resolution.pivot_world)
            uniform_scale = calculate_uniform_scale(
                first_settings.export.texture_width,
                first_settings.export.texture_height,
                first_settings.rig_scale_mode,
            )
            expected_main_position = (
                float(projected_pivot.u) * uniform_scale,
                float(projected_pivot.v) * uniform_scale,
            )
            prefixes = tuple(
                resolve_a1_names(
                    str(source.source_object.name_full),
                    source.settings,
                )[0]
                for source in plan.standalone_sources
            )
            _assert(
                len(prefixes) == len(set(prefixes)),
                f"selected objects produced duplicate rig prefixes: {prefixes!r}",
            )

            result = export_selected_objects_a1(bpy.context)
            json_path, outputs = _assert_production_outputs(
                result,
                selected_count=len(selected),
                prefixes=prefixes,
                expected_main_position=expected_main_position,
            )

            _assert(
                _scene_fingerprint(scene, selected) == before,
                "real shared-pivot export changed source object/scene/context state",
            )
            _assert(
                _datablock_fingerprint() == datablocks_before,
                "real shared-pivot export leaked or removed Blender datablocks",
            )

            print(
                "[GRENADE-SHARED-PIVOT-REAL-EXPORT] PASS "
                f"blend={loaded} selected_meshes={len(selected)} "
                f"projection={direction.value!r} "
                f"pivot_world={resolution.pivot_world!r} "
                f"projected_main={expected_main_position!r} "
                f"vertices={resolution.vertex_count} outputs={len(outputs)} "
                f"json={str(json_path)!r} source=unchanged",
                flush=True,
            )
        finally:
            scene.spine2d_json_path = original_output_path
            scene.spine2d_shared_selection_pivot = original_shared_pivot
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
