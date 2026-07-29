"""Real Blender 5.2 setup-pose regression for connected three-axis composition."""

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


def _assert_zero_relative_local_delta(document: dict, name: str) -> None:
    constraint = _constraint(document, name)
    _assert(constraint.get("local") is True, f"{name} is not local")
    _assert(constraint.get("relative") is True, f"{name} is not relative")
    for field_name in ("rotation", "x", "y", "scaleX", "scaleY", "shearY"):
        _assert(
            float(constraint.get(field_name, 0.0)) == 0.0,
            f"{name} has non-neutral {field_name}: {constraint}",
        )
    for field_name in ("mixX", "mixY", "mixScaleX", "mixScaleY", "mixShearY"):
        _assert(
            float(constraint.get(field_name, 1.0)) == 0.0,
            f"{name} still mixes {field_name}: {constraint}",
        )


def test_connected_three_axis_setup_pose_is_not_collapsed() -> None:
    _clear_scene()
    with tempfile.TemporaryDirectory(
        prefix="spine2d-connected-three-axis-setup-"
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
            output_stem="ConnectedThreeAxisSetup",
        )

        result = export_a1_multi_object(sources, settings)
        _assert(result.success, f"connected three-axis export failed: {result.issues}")

        output_json = (output_directory / "ConnectedThreeAxisSetup.json").resolve()
        document = json.loads(output_json.read_text(encoding="utf-8"))
        bones = {bone["name"]: bone for bone in document["bones"]}

        for name in (
            "all_objects_rotation_X",
            "all_objects_rotation_Y",
            "all_objects_rotation_Z",
        ):
            _assert_zero_relative_local_delta(document, name)

        rotation_z = _constraint(document, "all_objects_rotation_Z")
        _assert(
            tuple(rotation_z.get("bones", ()))
            == ("all_objects_layer_0", "all_objects_layer_1"),
            f"global Z constraint must operate on wrapper layers: {rotation_z}",
        )

        scale = _constraint(document, "all_objects_scale_constraint")
        _assert(float(scale.get("mixRotate", 1.0)) == 0.0, f"scale rotates: {scale}")
        _assert(float(scale.get("mixX", 1.0)) == 0.0, f"scale translates X: {scale}")
        _assert(float(scale.get("mixY", 1.0)) == 0.0, f"scale translates Y: {scale}")
        _assert(float(scale.get("mixScaleX", 0.0)) == 1.0, f"scale X disabled: {scale}")
        _assert(float(scale.get("mixScaleY", 0.0)) == 1.0, f"scale Y disabled: {scale}")
        _assert(float(scale.get("mixShearY", 1.0)) == 0.0, f"scale shears: {scale}")

        layer_y = float(bones["all_objects_0_scale"].get("y", 0.0))
        object_y = float(bones["ObjectB_main"].get("y", 0.0))
        _assert(layer_y == 16.0, f"ObjectB layer depth is wrong: {layer_y}")
        _assert(object_y == 16.0, f"ObjectB local Y is wrong: {object_y}")
        _assert(
            layer_y + object_y == 32.0,
            "ObjectB setup world Y no longer matches Blender placement",
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
        "[CONNECTED_SETUP_POSE] RUN "
        "test_connected_three_axis_setup_pose_is_not_collapsed"
    )
    test_connected_three_axis_setup_pose_is_not_collapsed()
    print(
        "[CONNECTED_SETUP_POSE] PASS "
        "test_connected_three_axis_setup_pose_is_not_collapsed"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
