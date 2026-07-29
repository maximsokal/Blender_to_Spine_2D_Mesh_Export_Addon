"""Real Blender 5.2 parity gate for the Legacy connected three-axis rig."""

from __future__ import annotations

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

from Blender_to_Spine2D_Mesh_Exporter.application import A1MultiObjectMode  # noqa: E402
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter import export_a1_multi_object  # noqa: E402
from run_multi_object_export_integration import (  # noqa: E402
    _assert,
    _assert_state_restored,
    _clear_scene,
    _multi_settings,
    _prepare_state,
)


def _constraint(document: dict, name: str) -> dict:
    matches = tuple(
        item
        for item in (*document.get("ik", ()), *document.get("transform", ()))
        if item.get("name") == name
    )
    _assert(len(matches) == 1, f"expected one constraint {name!r}, found {len(matches)}")
    return matches[0]


def _assert_exact_fields(actual: dict, expected: dict, label: str) -> None:
    for key, value in expected.items():
        _assert(actual.get(key) == value, f"{label}.{key}: {actual.get(key)!r} != {value!r}")


def test_connected_three_axis_matches_legacy_main_wrapper() -> None:
    _clear_scene()
    with tempfile.TemporaryDirectory(
        prefix="spine2d-connected-three-axis-main-parity-"
    ) as directory:
        output_directory = Path(directory)
        (
            sources,
            materials,
            context_before,
            scene_before,
            material_fingerprints,
        ) = _prepare_state(output_directory)
        settings = _multi_settings(
            output_directory,
            mode=A1MultiObjectMode.CONNECTED,
            output_stem="ConnectedThreeAxisMainParity",
        )

        result = export_a1_multi_object(sources, settings)
        _assert(result.success, f"connected three-axis export failed: {result.issues}")

        output_json = (output_directory / "ConnectedThreeAxisMainParity.json").resolve()
        document = json.loads(output_json.read_text(encoding="utf-8"))
        bones = {bone["name"]: bone for bone in document["bones"]}

        for control in (
            "all_objects_rotation_X",
            "all_objects_rotation_Y",
            "all_objects_rotation_Z",
        ):
            _assert(
                bones[control].get("parent") == "root",
                f"Legacy global control is not root-space: {bones[control]}",
            )

        for name in (
            "all_objects_0_scale",
            "all_objects_layer_0",
            "all_objects_1_scale",
            "all_objects_layer_1",
        ):
            _assert(float(bones[name].get("y", 0.0)) == 0.0, f"{name} has setup Y")
            _assert(
                float(bones[name].get("rotation", 0.0)) == 0.0,
                f"{name} has setup rotation",
            )
            _assert("inherit" not in bones[name], f"{name} has object-rig inherit mode")

        rotation_x = _constraint(document, "all_objects_rotation_X")
        _assert_exact_fields(
            rotation_x,
            {
                "order": 0,
                "bones": [
                    "all_objects_0_scale",
                    "all_objects_1_scale",
                    "all_objects",
                ],
                "target": "all_objects_rotation_X",
                "rotation": 90,
                "local": True,
                "relative": True,
                "x": -64.0,
                "y": -16.0,
                "scaleX": -1,
                "scaleY": -1,
                "mixX": 0,
                "mixScaleX": 0,
                "mixShearY": 0,
            },
            "all_objects_rotation_X",
        )

        rotation_y = _constraint(document, "all_objects_rotation_Y")
        _assert_exact_fields(
            rotation_y,
            {
                "order": 1,
                "bones": [
                    "all_objects_rotate_X",
                    "all_objects_rotate_X_constraint_rotate_IK",
                ],
                "target": "all_objects_rotation_Y",
                "local": True,
                "relative": True,
                "x": 32.0,
                "scaleX": -1,
                "mixX": 0,
                "mixScaleX": 0,
                "mixShearY": 0,
            },
            "all_objects_rotation_Y",
        )

        rotation_z = _constraint(document, "all_objects_rotation_Z")
        _assert_exact_fields(
            rotation_z,
            {
                "order": 2,
                "bones": ["ObjectA", "ObjectB"],
                "target": "all_objects_rotation_Z",
                "local": True,
                "mixX": 0,
                "mixScaleX": 0,
                "mixShearY": 0,
            },
            "all_objects_rotation_Z",
        )

        scale = _constraint(document, "all_objects_scale_constraint")
        _assert_exact_fields(
            scale,
            {
                "order": 10,
                "bones": ["all_objects_0_scale", "all_objects_1_scale"],
                "target": "all_objects_rotate_X_constraint",
                "scaleX": -1,
                "mixX": 0,
                "mixScaleX": 0,
                "mixShearY": 0,
            },
            "all_objects_scale_constraint",
        )

        _assert(float(bones["ObjectB_main"].get("x", 0.0)) == 64.0, "ObjectB X wrong")
        _assert(float(bones["ObjectB_main"].get("y", 0.0)) == 32.0, "ObjectB Y wrong")
        _assert(
            float(bones["all_objects_0_scale"].get("y", 0.0)) == 0.0,
            "Legacy wrapper added Z to visible Y",
        )

        _assert(_constraint(document, "ObjectA_scale_compensator")["order"] == 6, "A compensator order")
        _assert(_constraint(document, "ObjectB_scale_compensator")["order"] == 6, "B compensator order")

        _assert_state_restored(
            context_before=context_before,
            scene_before=scene_before,
            materials=materials,
            material_fingerprints=material_fingerprints,
        )


def main() -> None:
    print(f"Blender version: {bpy.app.version_string}")
    print("[CONNECTED_MAIN_PARITY] RUN test_connected_three_axis_matches_legacy_main_wrapper")
    test_connected_three_axis_matches_legacy_main_wrapper()
    print("[CONNECTED_MAIN_PARITY] PASS test_connected_three_axis_matches_legacy_main_wrapper")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
