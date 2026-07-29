"""Real Blender 5.2 setup-pose regression for both connected rig profiles."""

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
        rotation_x = _constraint(document, "all_objects_rotation_X")
        rotation_z = _constraint(document, "all_objects_rotation_Z")

        _assert(
            "scaleY" not in rotation_x,
            f"global X constraint still writes a destructive Y-scale offset: {rotation_x}",
        )
        _assert(
            float(rotation_x.get("mixScaleY", 1.0)) == 0.0,
            f"global X constraint still mixes Y scale in setup pose: {rotation_x}",
        )
        _assert(
            tuple(rotation_z.get("bones", ()))
            == ("all_objects_layer_0", "all_objects_layer_1"),
            f"global Z constraint must operate on wrapper layers: {rotation_z}",
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
