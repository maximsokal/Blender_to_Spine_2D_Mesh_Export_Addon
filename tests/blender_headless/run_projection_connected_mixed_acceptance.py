"""Validate projected connected and mixed Normal / UV Segments composition in Blender.

The worker reuses the accepted Active Camera fixture but sends two objects through the
connected hierarchy and one object through the standalone side of MIXED composition.
It proves that projected Object Origin owns placement/layers while nearest evaluated
vertex owns setup slot order across subgroup boundaries.
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

import run_active_camera_normal_uv_acceptance as base  # noqa: E402

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
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.connected_group_contracts import (  # noqa: E402
    ConnectedGroupBuildResult,
)
from spine_setup_transform import (  # noqa: E402
    evaluate_spine_setup_bone_position,
)


_POSITION_TOLERANCE = 0.011
_ANCHOR_COMPONENT_ID = "component_beta"
_SETUP_TRANSFORM_MODEL = "SPINE_AFFINE_NORMAL_ONLY_TRANSLATION"


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else ()
    return parser.parse_args(arguments)


def _bone_by_name(document, name: str):
    matches = tuple(bone for bone in document.bones if bone.name == name)
    base._assert(
        len(matches) == 1,
        f"Expected exactly one bone {name!r}, found {len(matches)}",
    )
    return matches[0]


def _setup_world_position(document, bone_name: str) -> tuple[float, float]:
    """Evaluate one object main through the complete Spine setup hierarchy."""

    return evaluate_spine_setup_bone_position(document, bone_name)


def _expected_order(expected_by_component) -> tuple[str, ...]:
    return tuple(
        component_id
        for component_id, expected in sorted(
            expected_by_component.items(),
            key=lambda item: max(vertex.camera_z for vertex in item[1]),
        )
    )


def _origin_front_order(state_by_component, camera_view_matrix) -> tuple[str, ...]:
    return tuple(
        component_id
        for component_id, (_source, state) in sorted(
            state_by_component.items(),
            key=lambda item: base._camera_z(
                camera_view_matrix,
                base._matrix_translation(item[1][1].matrix_world),
            ),
            reverse=True,
        )
    )


def _owner_maps(sources, objects):
    owner_by_slot: dict[str, str] = {}
    slots_by_component: dict[str, tuple[str, ...]] = {}
    for source, item in zip(sources, objects, strict=True):
        slot_names = tuple(slot.name for slot in item.document.slots)
        base._assert(slot_names, f"{source.component_id} produced no slots")
        slots_by_component[source.component_id] = slot_names
        for slot_name in slot_names:
            base._assert(slot_name not in owner_by_slot, f"Duplicate slot {slot_name}")
            owner_by_slot[slot_name] = source.component_id
    return owner_by_slot, slots_by_component


def _assert_block_order(document, owner_by_slot, slots_by_component, expected_order):
    slot_names = tuple(slot.name for slot in document.slots)
    actual_order = base._collapsed_owner_order(slot_names, owner_by_slot)
    base._assert(
        actual_order == expected_order,
        f"Object-block order mismatch: actual={actual_order}, expected={expected_order}",
    )
    for component_id in expected_order:
        actual_slots = tuple(
            slot_name
            for slot_name in slot_names
            if owner_by_slot[slot_name] == component_id
        )
        base._assert(
            actual_slots == slots_by_component[component_id],
            f"Internal slot order changed for {component_id}",
        )
    return actual_order


def _build_fixture(output_root: Path, camera_kind: str):
    base._clear_scene()
    base._configure_cycles_scene()
    scene = bpy.context.scene
    scene.render.resolution_x = base._SCENE_RENDER_WIDTH
    scene.render.resolution_y = base._SCENE_RENDER_HEIGHT
    scene.render.resolution_percentage = 63
    scene.render.pixel_aspect_x = 1.25
    scene.render.pixel_aspect_y = 0.8
    camera = base._create_camera(camera_kind)
    bpy.context.view_layer.update()

    camera_matrix_before = base._matrix_tuple(camera.matrix_world)
    camera_data_before = (
        camera.data.type,
        float(camera.data.clip_start),
        float(camera.data.clip_end),
        float(camera.data.shift_x),
        float(camera.data.shift_y),
        float(camera.data.lens),
        float(camera.data.ortho_scale),
    )
    render_before = (
        int(scene.render.resolution_x),
        int(scene.render.resolution_y),
        int(scene.render.resolution_percentage),
        float(scene.render.pixel_aspect_x),
        float(scene.render.pixel_aspect_y),
    )
    camera_view_matrix = base._rotation_only_camera_view_matrix(camera)

    kind_root = output_root / camera_kind.lower()
    kind_root.mkdir(parents=True, exist_ok=True)
    sources: list[A1MultiObjectSource] = []
    state_by_component = {}
    expected_by_component = {}
    for specification in base._SPECIFICATIONS:
        source_object = base._create_cuboid(specification)
        base._activate_only(source_object)
        bpy.context.view_layer.update()
        state = base._capture_state(source_object)
        expected = base._expected_vertices(
            scene,
            camera,
            state,
            camera_view_matrix,
        )
        sources.append(
            A1MultiObjectSource(
                source_object=source_object,
                component_id=specification.component_id,
                animation_namespace=specification.component_id,
                settings=base._settings(kind_root, specification, camera_kind),
            )
        )
        state_by_component[specification.component_id] = (source_object, state)
        expected_by_component[specification.component_id] = expected

    expected_order = _expected_order(expected_by_component)
    origin_front_order = _origin_front_order(state_by_component, camera_view_matrix)
    base._assert(
        expected_order != tuple(reversed(origin_front_order)),
        f"{camera_kind} fixture no longer distinguishes origin layers from draw order",
    )
    return (
        scene,
        camera,
        tuple(sources),
        state_by_component,
        expected_by_component,
        expected_order,
        origin_front_order,
        camera_matrix_before,
        camera_data_before,
        render_before,
    )


def _assert_source_and_camera_unchanged(
    camera_kind,
    scene,
    camera,
    state_by_component,
    camera_matrix_before,
    camera_data_before,
    render_before,
):
    maximum_matrix_delta = 0.0
    for component_id, (source_object, state) in state_by_component.items():
        maximum_matrix_delta = max(
            maximum_matrix_delta,
            base._assert_state_unchanged(
                source_object,
                state,
                label=f"{camera_kind} {component_id}",
            ),
        )
    base._assert(
        base._matrix_tuple(camera.matrix_world) == camera_matrix_before,
        f"{camera_kind} mutated camera matrix",
    )
    base._assert(
        (
            camera.data.type,
            float(camera.data.clip_start),
            float(camera.data.clip_end),
            float(camera.data.shift_x),
            float(camera.data.shift_y),
            float(camera.data.lens),
            float(camera.data.ortho_scale),
        )
        == camera_data_before,
        f"{camera_kind} mutated Camera data",
    )
    base._assert(
        (
            int(scene.render.resolution_x),
            int(scene.render.resolution_y),
            int(scene.render.resolution_percentage),
            float(scene.render.pixel_aspect_x),
            float(scene.render.pixel_aspect_y),
        )
        == render_before,
        f"{camera_kind} mutated render settings",
    )
    return maximum_matrix_delta


def _run_kind(output_root: Path, camera_kind: str) -> dict[str, object]:
    (
        scene,
        camera,
        sources,
        state_by_component,
        expected_by_component,
        expected_order,
        origin_front_order,
        camera_matrix_before,
        camera_data_before,
        render_before,
    ) = _build_fixture(output_root, camera_kind)

    connected_sources = sources[:2]
    standalone_sources = sources[2:]
    connected_settings = A1MultiObjectExportSettings(
        output_directory=output_root / camera_kind.lower(),
        output_stem=f"connected_{camera_kind.lower()}",
        mode=A1MultiObjectMode.CONNECTED,
        anchor_component_id=_ANCHOR_COMPONENT_ID,
        z_tolerance=1.0e-5,
    )
    connected_prepared = prepare_a1_multi_object(
        connected_sources,
        connected_settings,
        context=bpy.context,
        scene=scene,
    )
    connected = compose_a1_multi_object_document(
        connected_prepared.sources,
        connected_prepared.objects,
        connected_settings,
    )
    base._assert(
        isinstance(connected, ConnectedGroupBuildResult),
        "Connected composition returned unexpected result",
    )

    connected_expected_order = tuple(
        component_id
        for component_id in expected_order
        if component_id in {item.component_id for item in connected_sources}
    )
    connected_owner, connected_slots = _owner_maps(
        connected_prepared.sources,
        connected_prepared.objects,
    )
    connected_actual_order = _assert_block_order(
        connected.document,
        connected_owner,
        connected_slots,
        connected_expected_order,
    )

    anchor_expected = base._expected_screen_point(
        scene,
        camera,
        base._matrix_translation(
            state_by_component[_ANCHOR_COMPONENT_ID][1].matrix_world
        ),
    )
    group_main_name = connected_prepared.objects[0].rig.profile.main_bone(
        connected_settings.connected_group_prefix
    )
    group_main = _bone_by_name(connected.document, group_main_name)
    group_main_position = (
        0.0 if group_main.x is None else float(group_main.x),
        0.0 if group_main.y is None else float(group_main.y),
    )
    group_main_delta = max(
        abs(actual - expected)
        for actual, expected in zip(
            group_main_position,
            anchor_expected,
            strict=True,
        )
    )
    base._assert(
        group_main_delta <= _POSITION_TOLERANCE,
        f"{camera_kind} group main does not match projected anchor: "
        f"actual={group_main_position}, expected={anchor_expected}",
    )

    maximum_connected_position_delta = 0.0
    for source, item in zip(
        connected_prepared.sources,
        connected_prepared.objects,
        strict=True,
    ):
        actual_position = _setup_world_position(
            connected.document,
            item.rig.info.main_bone_name,
        )
        expected_position = base._expected_screen_point(
            scene,
            camera,
            base._matrix_translation(
                state_by_component[source.component_id][1].matrix_world
            ),
        )
        delta = max(
            abs(actual - expected)
            for actual, expected in zip(
                actual_position,
                expected_position,
                strict=True,
            )
        )
        maximum_connected_position_delta = max(
            maximum_connected_position_delta,
            delta,
        )
        base._assert(
            delta <= _POSITION_TOLERANCE,
            f"{camera_kind} connected setup position mismatch for "
            f"{source.component_id}: actual={actual_position}, "
            f"expected={expected_position}, delta={delta}",
        )

    actual_layer_front_order = tuple(
        component_id
        for layer in connected.layers
        for component_id in layer.component_ids
    )
    expected_connected_origin_front = tuple(
        component_id
        for component_id in origin_front_order
        if component_id in {item.component_id for item in connected_sources}
    )
    base._assert(
        actual_layer_front_order == expected_connected_origin_front,
        f"{camera_kind} connected layers do not use projected origin depth: "
        f"actual={actual_layer_front_order}, expected={expected_connected_origin_front}",
    )

    mixed_settings = A1MultiObjectExportSettings(
        output_directory=output_root / camera_kind.lower(),
        output_stem=f"mixed_{camera_kind.lower()}",
        mode=A1MultiObjectMode.MIXED,
        anchor_component_id=_ANCHOR_COMPONENT_ID,
        z_tolerance=1.0e-5,
    )
    mixed_prepared = prepare_a1_mixed_object(
        connected_sources,
        standalone_sources,
        mixed_settings,
        context=bpy.context,
        scene=scene,
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
    mixed_owner, mixed_slots = _owner_maps(
        mixed_prepared.sources,
        mixed_prepared.objects,
    )
    mixed_actual_order = _assert_block_order(
        mixed.document,
        mixed_owner,
        mixed_slots,
        expected_order,
    )

    maximum_matrix_delta = _assert_source_and_camera_unchanged(
        camera_kind,
        scene,
        camera,
        state_by_component,
        camera_matrix_before,
        camera_data_before,
        render_before,
    )
    return {
        "cameraType": camera_kind,
        "anchorComponentId": _ANCHOR_COMPONENT_ID,
        "setupTransformModel": _SETUP_TRANSFORM_MODEL,
        "expectedAllObjectOrder": list(expected_order),
        "connectedObjectOrder": list(connected_actual_order),
        "mixedObjectOrder": list(mixed_actual_order),
        "originFrontOrder": list(origin_front_order),
        "connectedLayerFrontOrder": list(actual_layer_front_order),
        "groupMainPosition": list(group_main_position),
        "expectedAnchorPosition": list(anchor_expected),
        "maximumGroupMainDelta": group_main_delta,
        "maximumConnectedPositionDelta": maximum_connected_position_delta,
        "maximumMatrixDelta": maximum_matrix_delta,
        "sourceUnchanged": True,
        "cameraUnchanged": True,
        "sceneRenderUnchanged": True,
    }


def run(output_directory: Path) -> Path:
    output_root = base._prepare_output_directory(output_directory)
    camera_results = tuple(
        _run_kind(output_root, camera_kind)
        for camera_kind in ("PERSP", "ORTHO")
    )
    report = {
        "status": "passed",
        "blenderVersion": bpy.app.version_string,
        "cameraCount": len(camera_results),
        "cameras": list(camera_results),
    }
    report_path = output_root / "projection_connected_mixed_acceptance.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report_path


def main() -> None:
    arguments = _parse_arguments()
    print(f"Blender version: {bpy.app.version_string}")
    print("[PROJECTION_CONNECTED_MIXED] RUN Perspective + Orthographic")
    report_path = run(arguments.output)
    print(f"[PROJECTION_CONNECTED_MIXED] REPORT {report_path}")
    print("[PROJECTION_CONNECTED_MIXED] PASS")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
