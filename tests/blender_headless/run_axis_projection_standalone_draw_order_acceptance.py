"""Validate six-axis standalone placement and complete object-block draw order.

The worker creates three real Blender cuboids whose Object Origins and nearest geometry
produce intentionally different depth orders. Every direction runs production object
preparation and standalone document composition, then verifies far-to-near object blocks,
unchanged per-object segment order, projected pivots, and exact source immutability.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
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
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.a1_multi_object_composition import (  # noqa: E402
    compose_a1_multi_object_document,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import (  # noqa: E402
    BakeExecutionSettings,
    BakeMode,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.projection import (  # noqa: E402
    A1ProjectionDirection,
    resolve_a1_axis_projection_basis,
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
_MATRIX_TOLERANCE = 1.0e-7
_AXIS_DIRECTIONS = tuple(
    direction
    for direction in A1ProjectionDirection
    if direction is not A1ProjectionDirection.ACTIVE_CAMERA
)


@dataclass(frozen=True)
class _ObjectSpecification:
    name: str
    component_id: str
    location: tuple[float, float, float]
    local_minimum: tuple[float, float, float]
    local_maximum: tuple[float, float, float]


@dataclass(frozen=True)
class _SourceState:
    vertices: tuple[tuple[float, float, float], ...]
    matrix_world: tuple[float, ...]
    location: tuple[float, float, float]
    rotation_euler: tuple[float, float, float]
    scale: tuple[float, float, float]


# Source order is deliberately unrelated to most projected depth orders.
_SPECIFICATIONS = (
    _ObjectSpecification(
        name="DepthBeta",
        component_id="component_beta",
        location=(2.0, -4.0, 0.0),
        local_minimum=(-1.0, 0.0, -1.0),
        local_maximum=(1.0, 9.0, 1.0),
    ),
    _ObjectSpecification(
        name="DepthAlpha",
        component_id="component_alpha",
        location=(-3.0, 1.0, 4.0),
        local_minimum=(0.0, -1.0, -1.0),
        local_maximum=(8.0, 1.0, 1.0),
    ),
    _ObjectSpecification(
        name="DepthGamma",
        component_id="component_gamma",
        location=(0.0, 3.0, -5.0),
        local_minimum=(-1.0, -1.0, 0.0),
        local_maximum=(1.0, 1.0, 12.0),
    ),
)

_EXPECTED_COMPONENT_ORDER = {
    A1ProjectionDirection.POSITIVE_X: (
        "component_gamma",
        "component_beta",
        "component_alpha",
    ),
    A1ProjectionDirection.NEGATIVE_X: (
        "component_beta",
        "component_gamma",
        "component_alpha",
    ),
    A1ProjectionDirection.POSITIVE_Y: (
        "component_alpha",
        "component_gamma",
        "component_beta",
    ),
    A1ProjectionDirection.NEGATIVE_Y: (
        "component_gamma",
        "component_alpha",
        "component_beta",
    ),
    A1ProjectionDirection.POSITIVE_Z: (
        "component_beta",
        "component_alpha",
        "component_gamma",
    ),
    A1ProjectionDirection.NEGATIVE_Z: (
        "component_alpha",
        "component_beta",
        "component_gamma",
    ),
}


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


def _matrix_tuple(matrix: object) -> tuple[float, ...]:
    return tuple(
        float(matrix[row][column])
        for row in range(4)
        for column in range(4)
    )


def _cuboid_vertices(
    minimum: tuple[float, float, float],
    maximum: tuple[float, float, float],
) -> tuple[tuple[float, float, float], ...]:
    min_x, min_y, min_z = minimum
    max_x, max_y, max_z = maximum
    if not (min_x < max_x and min_y < max_y and min_z < max_z):
        raise ValueError("Cuboid minimum must be strictly below maximum")
    return (
        (min_x, min_y, min_z),
        (max_x, min_y, min_z),
        (max_x, max_y, min_z),
        (min_x, max_y, min_z),
        (min_x, min_y, max_z),
        (max_x, min_y, max_z),
        (max_x, max_y, max_z),
        (min_x, max_y, max_z),
    )


def _create_cuboid(specification: _ObjectSpecification) -> bpy.types.Object:
    vertices = _cuboid_vertices(
        specification.local_minimum,
        specification.local_maximum,
    )
    faces = (
        (0, 3, 2, 1),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    )
    mesh = bpy.data.meshes.new(f"{specification.name}Mesh")
    mesh.from_pydata(vertices, (), faces)
    mesh.update(calc_edges=True)
    source_object = bpy.data.objects.new(specification.name, mesh)
    bpy.context.scene.collection.objects.link(source_object)
    source_object.location = specification.location
    _create_emission_material(source_object)
    return source_object


def _capture_state(source_object: bpy.types.Object) -> _SourceState:
    bpy.context.view_layer.update()
    return _SourceState(
        vertices=tuple(
            tuple(float(value) for value in vertex.co)
            for vertex in source_object.data.vertices
        ),
        matrix_world=_matrix_tuple(source_object.matrix_world),
        location=tuple(float(value) for value in source_object.location),
        rotation_euler=tuple(float(value) for value in source_object.rotation_euler),
        scale=tuple(float(value) for value in source_object.scale),
    )


def _assert_state_unchanged(
    source_object: bpy.types.Object,
    before: _SourceState,
    *,
    label: str,
) -> float:
    after = _capture_state(source_object)
    _assert(after.vertices == before.vertices, f"{label} mutated source vertices")
    _assert(after.location == before.location, f"{label} mutated Object.location")
    _assert(
        after.rotation_euler == before.rotation_euler,
        f"{label} mutated Object.rotation_euler",
    )
    _assert(after.scale == before.scale, f"{label} mutated Object.scale")
    maximum_delta = max(
        (
            abs(actual - expected)
            for actual, expected in zip(
                after.matrix_world,
                before.matrix_world,
                strict=True,
            )
        ),
        default=0.0,
    )
    _assert(
        maximum_delta <= _MATRIX_TOLERANCE,
        f"{label} mutated matrix_world: max_delta={maximum_delta}",
    )
    return maximum_delta


def _settings(
    output_directory: Path,
    specification: _ObjectSpecification,
    direction: A1ProjectionDirection,
) -> A1SingleObjectExportSettings:
    direction_token = direction.value.lower()
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
        prefix=specification.name,
        output_stem=f"{specification.name}_{direction_token}",
        source_geometry_mode=A1SourceGeometryMode.ORIGINAL,
        uv=UvUnwrapSettings(layer_name="SpineBakeUV"),
        diffuse_mode=BakeMode.EMIT,
        procedural_mode=BakeMode.EMIT,
        bake_execution=BakeExecutionSettings(samples=1),
        rig_setup_pose_mode=A1RigSetupPoseMode.PRESERVE_COMPOSITION,
        projection_direction=direction,
    )


def _expected_nearest_depth(
    specification: _ObjectSpecification,
    direction: A1ProjectionDirection,
) -> float:
    basis = resolve_a1_axis_projection_basis(direction)
    world_vertices = tuple(
        (
            local[0] + specification.location[0],
            local[1] + specification.location[1],
            local[2] + specification.location[2],
        )
        for local in _cuboid_vertices(
            specification.local_minimum,
            specification.local_maximum,
        )
    )
    return max(basis.project_point(vertex).depth for vertex in world_vertices)


def _collapsed_owner_order(
    slot_names: tuple[str, ...],
    owner_by_slot: dict[str, str],
) -> tuple[str, ...]:
    owners: list[str] = []
    for slot_name in slot_names:
        owner = owner_by_slot.get(slot_name)
        if owner is None:
            raise AssertionError(f"Composed slot has no object owner: {slot_name}")
        if not owners or owners[-1] != owner:
            owners.append(owner)
    return tuple(owners)


def _run_direction(
    output_root: Path,
    direction: A1ProjectionDirection,
) -> dict[str, object]:
    _clear_scene()
    _configure_cycles_scene()
    direction_root = output_root / direction.value.lower()
    direction_root.mkdir(parents=True, exist_ok=True)

    sources: list[A1MultiObjectSource] = []
    state_by_component: dict[str, tuple[bpy.types.Object, _SourceState]] = {}
    for specification in _SPECIFICATIONS:
        source_object = _create_cuboid(specification)
        _activate_only(source_object)
        state = _capture_state(source_object)
        sources.append(
            A1MultiObjectSource(
                source_object=source_object,
                component_id=specification.component_id,
                animation_namespace=specification.component_id,
                settings=_settings(direction_root, specification, direction),
            )
        )
        state_by_component[specification.component_id] = (source_object, state)

    multi_settings = A1MultiObjectExportSettings(
        output_directory=direction_root,
        output_stem=f"standalone_{direction.value.lower()}",
        mode=A1MultiObjectMode.STANDALONE,
        z_tolerance=1.0e-4,
    )
    prepared = prepare_a1_multi_object(
        tuple(sources),
        multi_settings,
        context=bpy.context,
        scene=bpy.context.scene,
    )
    composition = compose_a1_multi_object_document(
        prepared.sources,
        prepared.objects,
        multi_settings,
    )

    expected_order = _EXPECTED_COMPONENT_ORDER[direction]
    computed_expected_order = tuple(
        specification.component_id
        for specification in sorted(
            _SPECIFICATIONS,
            key=lambda item: _expected_nearest_depth(item, direction),
        )
    )
    _assert(
        computed_expected_order == expected_order,
        f"Fixture depth order drifted for {direction.value}: "
        f"computed={computed_expected_order}, expected={expected_order}",
    )

    owner_by_slot: dict[str, str] = {}
    component_slot_names: dict[str, tuple[str, ...]] = {}
    object_report: list[dict[str, object]] = []
    maximum_matrix_delta = 0.0
    for source, item, specification in zip(
        prepared.sources,
        prepared.objects,
        _SPECIFICATIONS,
        strict=True,
    ):
        slots = tuple(slot.name for slot in item.document.slots)
        _assert(
            len(slots) >= 2,
            f"{direction.value} {source.component_id} did not retain multiple segments",
        )
        component_slot_names[source.component_id] = slots
        for slot_name in slots:
            previous = owner_by_slot.get(slot_name)
            _assert(
                previous is None,
                f"Slot {slot_name} is shared by {previous} and {source.component_id}",
            )
            owner_by_slot[slot_name] = source.component_id

        source_object, state = state_by_component[source.component_id]
        maximum_matrix_delta = max(
            maximum_matrix_delta,
            _assert_state_unchanged(
                source_object,
                state,
                label=f"{direction.value} {source.component_id}",
            ),
        )

        main_bone = next(
            bone
            for bone in item.document.bones
            if bone.name == item.rig.info.main_bone_name
        )
        nearest_depth = _expected_nearest_depth(specification, direction)
        object_report.append(
            {
                "componentId": source.component_id,
                "objectName": specification.name,
                "nearestVertexDepth": nearest_depth,
                "mainPosition": [
                    0.0 if main_bone.x is None else float(main_bone.x),
                    0.0 if main_bone.y is None else float(main_bone.y),
                ],
                "slotNames": list(slots),
            }
        )

    final_slot_names = tuple(slot.name for slot in composition.document.slots)
    actual_order = _collapsed_owner_order(final_slot_names, owner_by_slot)
    _assert(
        actual_order == expected_order,
        f"{direction.value} object-block order mismatch: "
        f"actual={actual_order}, expected={expected_order}",
    )
    _assert(
        len(actual_order) == len(_SPECIFICATIONS)
        and len(set(actual_order)) == len(_SPECIFICATIONS),
        f"{direction.value} split or duplicated an object slot block",
    )

    for component_id in expected_order:
        actual_component_slots = tuple(
            slot_name
            for slot_name in final_slot_names
            if owner_by_slot[slot_name] == component_id
        )
        _assert(
            actual_component_slots == component_slot_names[component_id],
            f"{direction.value} changed internal slot order for {component_id}",
        )

    return {
        "direction": direction.value,
        "sourceInputOrder": [item.component_id for item in sources],
        "expectedComponentOrder": list(expected_order),
        "actualComponentOrder": list(actual_order),
        "slotNames": list(final_slot_names),
        "objects": object_report,
        "sourceUnchanged": True,
        "maximumMatrixDelta": maximum_matrix_delta,
    }


def run(output_directory: Path) -> Path:
    output_root = _prepare_output_directory(output_directory)
    directions = tuple(
        _run_direction(output_root, direction) for direction in _AXIS_DIRECTIONS
    )
    report = {
        "status": "passed",
        "blenderVersion": bpy.app.version_string,
        "directionCount": len(directions),
        "directions": list(directions),
    }
    report_path = output_root / "axis_projection_standalone_draw_order_acceptance.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report_path


def main() -> None:
    arguments = _parse_arguments()
    print(f"Blender version: {bpy.app.version_string}")
    print("[AXIS_STANDALONE_DRAW_ORDER] RUN six signed-axis compositions")
    report_path = run(arguments.output)
    print(f"[AXIS_STANDALONE_DRAW_ORDER] REPORT {report_path}")
    print("[AXIS_STANDALONE_DRAW_ORDER] PASS")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
