"""Real Blender 5.2 connected multi-object integration for the two-axis scale rig."""

from __future__ import annotations

from dataclasses import replace
import json
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
    A1MultiObjectMode,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter import (  # noqa: E402
    export_a1_multi_object,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine import (  # noqa: E402
    A1RigProfile,
)
from run_multi_object_export_integration import (  # noqa: E402
    _assert,
    _assert_state_restored,
    _clear_scene,
    _multi_settings,
    _prepare_state,
)


def _two_axis_sources(sources):
    """Return immutable source contracts using the connected two-axis profile."""

    return tuple(
        replace(
            source,
            settings=replace(
                source.settings,
                export=replace(
                    source.settings.export,
                    rig_profile=A1RigProfile.TWO_AXIS_ROTATION_SCALE.value,
                ),
            ),
        )
        for source in sources
    )


def _assert_neutral_connected_setup(document: dict) -> None:
    """Validate the setup-pose fields that Spine evaluates before animation."""

    bones = {bone["name"]: bone for bone in document["bones"]}
    for control_name in ("all_objects_rotation_X", "all_objects_rotation_Y"):
        _assert(
            float(bones[control_name].get("rotation", 0.0)) == 0.0,
            f"global connected control has non-neutral setup rotation: {bones[control_name]}",
        )

    global_scale = bones["all_objects_scale"]
    expected_local_x = float(global_scale.get("x", 0.0))
    expected_local_y = float(global_scale.get("y", 0.0))
    for prefix in ("ObjectA", "ObjectB"):
        main = bones[f"{prefix}_main"]
        scale = bones[f"{prefix}_scale"]
        _assert(
            scale.get("parent") == f"{prefix}_main",
            f"{prefix} scale control is outside object control space: {scale}",
        )
        # Every object uses the profile layout in its own main-local space. The
        # local offset must match the neutral global control and must not include
        # the object's connected world placement.
        _assert(
            float(scale.get("x", 0.0)) == expected_local_x,
            f"{prefix} scale control local X is wrong: main={main}, scale={scale}",
        )
        _assert(
            float(scale.get("y", 0.0)) == expected_local_y,
            f"{prefix} scale control local Y is wrong: main={main}, scale={scale}",
        )


def test_connected_two_axis_export_builds_global_and_object_controls() -> None:
    _clear_scene()
    with tempfile.TemporaryDirectory(
        prefix="spine2d-multi-connected-two-axis-"
    ) as directory:
        output_directory = Path(directory)
        (
            legacy_sources,
            materials,
            context_before,
            scene_before,
            material_fingerprints,
        ) = _prepare_state(output_directory)
        sources = _two_axis_sources(legacy_sources)
        settings = _multi_settings(
            output_directory,
            mode=A1MultiObjectMode.CONNECTED,
            output_stem="ConnectedTwoAxisGroup",
        )

        result = export_a1_multi_object(sources, settings)

        _assert(
            result.success,
            f"connected two-axis multi export failed: {result.issues}",
        )
        expected_json = (output_directory / "ConnectedTwoAxisGroup.json").resolve()
        expected_a = (output_directory / "images" / "ObjectA_Baked.png").resolve()
        expected_b = (output_directory / "images" / "ObjectB_Baked.png").resolve()
        _assert(
            result.output_files == (expected_json, expected_a, expected_b),
            f"unexpected connected two-axis outputs: {result.output_files}",
        )

        document = json.loads(expected_json.read_text(encoding="utf-8"))
        bones = {bone["name"]: bone for bone in document["bones"]}
        required_controls = {
            "all_objects_rotation_X",
            "all_objects_rotation_Y",
            "all_objects_scale",
            "ObjectA_rotation_X",
            "ObjectA_rotation_Y",
            "ObjectA_scale",
            "ObjectB_rotation_X",
            "ObjectB_rotation_Y",
            "ObjectB_scale",
        }
        _assert(
            required_controls <= set(bones),
            f"connected two-axis controls missing: {sorted(required_controls - set(bones))}",
        )
        _assert(
            "all_objects_rotation_Z" not in bones,
            "two-axis connected global rig unexpectedly contains Rotation Z",
        )
        _assert(
            "ObjectA_rotation_Z" not in bones and "ObjectB_rotation_Z" not in bones,
            "two-axis connected object rigs unexpectedly contain Rotation Z",
        )
        _assert(
            bones["ObjectA_main"]["parent"] == "all_objects_layer_1",
            f"anchor object layer is wrong: {bones['ObjectA_main']}",
        )
        _assert(
            bones["ObjectB_main"]["parent"] == "all_objects_layer_0",
            f"elevated object layer is wrong: {bones['ObjectB_main']}",
        )
        _assert(
            float(bones["ObjectA_main"].get("x", 0.0)) == 0.0,
            "two-axis anchor X moved",
        )
        _assert(
            float(bones["ObjectA_main"].get("y", 0.0)) == 0.0,
            "two-axis anchor Y moved",
        )
        _assert(
            float(bones["ObjectB_main"].get("x", 0.0)) == 64.0,
            "two-axis ObjectB X offset wrong",
        )
        _assert(
            float(bones["ObjectB_main"].get("y", 0.0)) == 32.0,
            "two-axis ObjectB Y offset wrong",
        )
        _assert_neutral_connected_setup(document)

        constraints = tuple(document.get("ik", ())) + tuple(
            document.get("transform", ())
        )
        orders = tuple(int(item["order"]) for item in constraints)
        names = {item["name"] for item in constraints}
        required_constraints = {
            "all_objects_rotation_X_constraint",
            "all_objects_IK",
            "all_objects_scale",
            "all_objects_scale_rotate_X_constraint",
            "all_objects_rotation_Y",
            "ObjectA_rotation_X_constraint",
            "ObjectA_IK",
            "ObjectA_scale",
            "ObjectA_scale_rotate_X_constraint",
            "ObjectA_rotation_Y",
            "ObjectB_rotation_X_constraint",
            "ObjectB_IK",
            "ObjectB_scale",
            "ObjectB_scale_rotate_X_constraint",
            "ObjectB_rotation_Y",
        }
        _assert(
            names == required_constraints,
            f"unexpected connected two-axis constraints: {sorted(names)}",
        )
        _assert(
            len(orders) == len(set(orders)) == 15,
            f"connected two-axis constraint orders collide: {orders}",
        )
        _assert(
            set(orders) == set(range(15)),
            f"connected two-axis schedule is not contiguous: {orders}",
        )
        _assert(
            result.statistics["connected_layer_count"] == 2,
            "connected two-axis Z layer count is wrong",
        )
        _assert_state_restored(
            context_before=context_before,
            scene_before=scene_before,
            materials=materials,
            material_fingerprints=material_fingerprints,
        )


def main() -> None:
    print(f"Blender version: {bpy.app.version_string}")
    print(
        "[TWO_AXIS_CONNECTED_MULTI] RUN "
        "test_connected_two_axis_export_builds_global_and_object_controls"
    )
    test_connected_two_axis_export_builds_global_and_object_controls()
    print(
        "[TWO_AXIS_CONNECTED_MULTI] PASS "
        "test_connected_two_axis_export_builds_global_and_object_controls"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
