"""Validate connected and mixed projection normalization for all six signed axes."""

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

import run_axis_projection_standalone_draw_order_acceptance as base  # noqa: E402
import run_projection_connected_mixed_acceptance as shared  # noqa: E402

from Blender_to_Spine2D_Mesh_Exporter.application import (  # noqa: E402
    A1MultiObjectExportSettings,
    A1MultiObjectMode,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter import (  # noqa: E402
    A1MultiObjectSource,
    prepare_a1_mixed_object,
    prepare_a1_multi_object,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.a1_mixed_composition import (  # noqa: E402
    compose_a1_mixed_document,
    partition_mixed_prepared_objects,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.a1_multi_object_composition import (  # noqa: E402
    compose_a1_multi_object_document,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.projection import (  # noqa: E402
    A1ProjectionDirection,
    resolve_a1_axis_projection_basis,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.connected_group_contracts import (  # noqa: E402
    ConnectedGroupBuildResult,
)


_ANCHOR_COMPONENT_ID = "component_alpha"
_POSITION_TOLERANCE = 0.011


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else ()
    return parser.parse_args(arguments)


def _projected_origin_pixels(specification, direction, uniform_scale):
    projected = resolve_a1_axis_projection_basis(direction).project_point(
        specification.location
    )
    return (
        float(projected.u) * float(uniform_scale),
        float(projected.v) * float(uniform_scale),
        float(projected.depth),
    )


def _origin_front_order(direction):
    basis = resolve_a1_axis_projection_basis(direction)
    return tuple(
        specification.component_id
        for specification in sorted(
            base._SPECIFICATIONS,
            key=lambda item: basis.project_point(item.location).depth,
            reverse=True,
        )
    )


def _build_sources(direction_root, direction):
    sources: list[A1MultiObjectSource] = []
    state_by_component = {}
    specification_by_component = {}
    for specification in base._SPECIFICATIONS:
        source_object = base._create_cuboid(specification)
        base._activate_only(source_object)
        state = base._capture_state(source_object)
        sources.append(
            A1MultiObjectSource(
                source_object=source_object,
                component_id=specification.component_id,
                animation_namespace=specification.component_id,
                settings=base._settings(direction_root, specification, direction),
            )
        )
        state_by_component[specification.component_id] = (source_object, state)
        specification_by_component[specification.component_id] = specification
    return tuple(sources), state_by_component, specification_by_component


def _run_direction(output_root: Path, direction: A1ProjectionDirection):
    base._clear_scene()
    base._configure_cycles_scene()
    direction_root = output_root / direction.value.lower()
    direction_root.mkdir(parents=True, exist_ok=True)
    sources, state_by_component, specification_by_component = _build_sources(
        direction_root,
        direction,
    )

    connected_sources = sources[:2]
    standalone_sources = sources[2:]
    connected_settings = A1MultiObjectExportSettings(
        output_directory=direction_root,
        output_stem=f"connected_{direction.value.lower()}",
        mode=A1MultiObjectMode.CONNECTED,
        anchor_component_id=_ANCHOR_COMPONENT_ID,
        z_tolerance=1.0e-4,
    )
    connected_prepared = prepare_a1_multi_object(
        connected_sources,
        connected_settings,
        context=bpy.context,
        scene=bpy.context.scene,
    )
    connected = compose_a1_multi_object_document(
        connected_prepared.sources,
        connected_prepared.objects,
        connected_settings,
    )
    base._assert(
        isinstance(connected, ConnectedGroupBuildResult),
        f"{direction.value} connected composition returned unexpected result",
    )

    expected_all_order = base._EXPECTED_COMPONENT_ORDER[direction]
    connected_ids = {source.component_id for source in connected_sources}
    expected_connected_order = tuple(
        component_id
        for component_id in expected_all_order
        if component_id in connected_ids
    )
    connected_owner, connected_slots = shared._owner_maps(
        connected_prepared.sources,
        connected_prepared.objects,
    )
    connected_actual_order = shared._assert_block_order(
        connected.document,
        connected_owner,
        connected_slots,
        expected_connected_order,
    )

    uniform_scale = float(connected.uniform_scale)
    anchor_expected = _projected_origin_pixels(
        specification_by_component[_ANCHOR_COMPONENT_ID],
        direction,
        uniform_scale,
    )
    group_main_name = connected_prepared.objects[0].rig.profile.main_bone(
        connected_settings.connected_group_prefix
    )
    group_main = shared._bone_by_name(connected.document, group_main_name)
    group_main_position = (
        0.0 if group_main.x is None else float(group_main.x),
        0.0 if group_main.y is None else float(group_main.y),
    )
    group_main_delta = max(
        abs(group_main_position[0] - anchor_expected[0]),
        abs(group_main_position[1] - anchor_expected[1]),
    )
    base._assert(
        group_main_delta <= _POSITION_TOLERANCE,
        f"{direction.value} group main mismatch: actual={group_main_position}, "
        f"expected={anchor_expected[:2]}, delta={group_main_delta}",
    )

    maximum_connected_position_delta = 0.0
    for source, item in zip(
        connected_prepared.sources,
        connected_prepared.objects,
        strict=True,
    ):
        actual = shared._setup_world_position(
            connected.document,
            item.rig.info.main_bone_name,
        )
        expected = _projected_origin_pixels(
            specification_by_component[source.component_id],
            direction,
            uniform_scale,
        )
        delta = max(abs(actual[0] - expected[0]), abs(actual[1] - expected[1]))
        maximum_connected_position_delta = max(
            maximum_connected_position_delta,
            delta,
        )
        base._assert(
            delta <= _POSITION_TOLERANCE,
            f"{direction.value} connected position mismatch for "
            f"{source.component_id}: actual={actual}, expected={expected[:2]}",
        )

    origin_front_order = _origin_front_order(direction)
    expected_connected_origin_front = tuple(
        component_id for component_id in origin_front_order if component_id in connected_ids
    )
    actual_layer_front_order = tuple(
        component_id
        for layer in connected.layers
        for component_id in layer.component_ids
    )
    base._assert(
        actual_layer_front_order == expected_connected_origin_front,
        f"{direction.value} connected layers mismatch: "
        f"actual={actual_layer_front_order}, "
        f"expected={expected_connected_origin_front}",
    )

    mixed_settings = A1MultiObjectExportSettings(
        output_directory=direction_root,
        output_stem=f"mixed_{direction.value.lower()}",
        mode=A1MultiObjectMode.MIXED,
        anchor_component_id=_ANCHOR_COMPONENT_ID,
        z_tolerance=1.0e-4,
    )
    mixed_prepared = prepare_a1_mixed_object(
        connected_sources,
        standalone_sources,
        mixed_settings,
        context=bpy.context,
        scene=bpy.context.scene,
    )
    partition = partition_mixed_prepared_objects(
        mixed_prepared.objects,
        connected_sources,
        standalone_sources,
    )
    mixed = compose_a1_mixed_document(
        connected_sources,
        standalone_sources,
        partition,
        mixed_settings,
    )
    mixed_owner, mixed_slots = shared._owner_maps(
        mixed_prepared.sources,
        mixed_prepared.objects,
    )
    mixed_actual_order = shared._assert_block_order(
        mixed.document,
        mixed_owner,
        mixed_slots,
        expected_all_order,
    )

    maximum_matrix_delta = 0.0
    for component_id, (source_object, state) in state_by_component.items():
        maximum_matrix_delta = max(
            maximum_matrix_delta,
            base._assert_state_unchanged(
                source_object,
                state,
                label=f"{direction.value} {component_id}",
            ),
        )

    return {
        "direction": direction.value,
        "anchorComponentId": _ANCHOR_COMPONENT_ID,
        "setupTransformModel": shared._SETUP_TRANSFORM_MODEL,
        "expectedAllObjectOrder": list(expected_all_order),
        "connectedObjectOrder": list(connected_actual_order),
        "mixedObjectOrder": list(mixed_actual_order),
        "originFrontOrder": list(origin_front_order),
        "connectedLayerFrontOrder": list(actual_layer_front_order),
        "groupMainPosition": list(group_main_position),
        "expectedAnchorPosition": list(anchor_expected[:2]),
        "maximumGroupMainDelta": group_main_delta,
        "maximumConnectedPositionDelta": maximum_connected_position_delta,
        "maximumMatrixDelta": maximum_matrix_delta,
        "sourceUnchanged": True,
    }


def run(output_directory: Path) -> Path:
    output_root = base._prepare_output_directory(output_directory)
    directions = tuple(
        _run_direction(output_root, direction)
        for direction in base._AXIS_DIRECTIONS
    )
    report = {
        "status": "passed",
        "blenderVersion": bpy.app.version_string,
        "directionCount": len(directions),
        "directions": list(directions),
    }
    report_path = output_root / "axis_projection_connected_mixed_acceptance.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report_path


def main() -> None:
    arguments = _parse_arguments()
    print(f"Blender version: {bpy.app.version_string}")
    print("[AXIS_CONNECTED_MIXED] RUN six signed-axis connected/mixed compositions")
    report_path = run(arguments.output)
    print(f"[AXIS_CONNECTED_MIXED] REPORT {report_path}")
    print("[AXIS_CONNECTED_MIXED] PASS")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
