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


def _constraint(document: dict, name: str) -> dict:
    matches = tuple(
        item
        for item in (*document.get("ik", ()), *document.get("transform", ()))
        if item.get("name") == name
    )
    _assert(len(matches) == 1, f"expected one constraint {name!r}, found {len(matches)}")
    return matches[0]


def _assert_fields(actual: dict, expected: dict, label: str) -> None:
    for key, value in expected.items():
        _assert(
            actual.get(key) == value,
            f"{label}.{key}: {actual.get(key)!r} != {value!r}; actual={actual}",
        )


def _assert_zero_relative_local_delta(document: dict, constraint_name: str) -> None:
    """Assert that one global wrapper rotation is identity in setup pose."""

    constraint = _constraint(document, constraint_name)
    _assert(constraint.get("local") is True, f"{constraint_name} is not local")
    _assert(constraint.get("relative") is True, f"{constraint_name} is not relative")
    for field_name in ("rotation", "x", "y", "scaleX", "scaleY", "shearY"):
        _assert(
            float(constraint.get(field_name, 0.0)) == 0.0,
            f"{constraint_name} has non-neutral {field_name}: {constraint}",
        )
    for field_name in ("mixX", "mixY", "mixScaleX", "mixScaleY", "mixShearY"):
        _assert(
            float(constraint.get(field_name, 1.0)) == 0.0,
            f"{constraint_name} still mixes {field_name}: {constraint}",
        )


def _assert_connected_control_space(document: dict) -> None:
    """Validate the setup-pose fields that Spine evaluates before animation."""

    bones = {bone["name"]: bone for bone in document["bones"]}
    for control_name in ("all_objects_rotation_X", "all_objects_rotation_Y"):
        _assert(
            float(bones[control_name].get("rotation", 0.0)) == 0.0,
            f"global connected control has non-neutral setup rotation: {bones[control_name]}",
        )

    _assert_zero_relative_local_delta(document, "all_objects_rotation_X_constraint")
    _assert_zero_relative_local_delta(document, "all_objects_rotation_Y")

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
        layer_y = float(bones["all_objects_0_scale"].get("y", 0.0))
        object_y = float(bones["ObjectB_main"].get("y", 0.0))
        _assert(layer_y == 16.0, f"two-axis ObjectB layer depth is wrong: {layer_y}")
        _assert(object_y == 16.0, f"two-axis ObjectB local Y is wrong: {object_y}")
        _assert(
            layer_y + object_y == 32.0,
            "two-axis ObjectB setup world Y no longer matches Blender placement",
        )
        _assert_connected_control_space(document)

        # Spine 4.2 uses the global-first two-axis schedule. The complete global
        # wrapper must run before any object-local constraints so a later depth phase
        # cannot reset a subtree below the historical zero-scale setup parent.
        _assert_fields(
            _constraint(document, "all_objects_rotation_X_constraint"),
            {
                "order": 0,
                "bones": [
                    "all_objects_rotate_X_constraint_rotate_IK",
                    "all_objects_rotate_X",
                ],
                "target": "all_objects_rotation_X",
                "local": True,
                "relative": True,
                "mixX": 0,
                "mixY": 0,
                "mixScaleX": 0,
                "mixScaleY": 0,
                "mixShearY": 0,
            },
            "all_objects_rotation_X_constraint",
        )
        _assert_fields(
            _constraint(document, "all_objects_IK"),
            {
                "order": 1,
                "bones": ["all_objects_rotate_X_constraint"],
                "target": "all_objects_rotate_X_constraint_IK",
                "compress": True,
                "stretch": True,
            },
            "all_objects_IK",
        )
        _assert_fields(
            _constraint(document, "all_objects_scale"),
            {
                "order": 2,
                "bones": [
                    "all_objects_rotate_X",
                    "all_objects_layer_0",
                    "all_objects_layer_1",
                ],
                "target": "all_objects_scale",
                "relative": True,
                "mixRotate": 0,
                "mixX": 0,
                "mixY": 0,
                "mixShearY": 0,
            },
            "all_objects_scale",
        )
        _assert_fields(
            _constraint(document, "all_objects_scale_rotate_X_constraint"),
            {
                "order": 3,
                "bones": ["all_objects_0_scale", "all_objects_1_scale"],
                "target": "all_objects_rotate_X_constraint",
            },
            "all_objects_scale_rotate_X_constraint",
        )
        _assert_fields(
            _constraint(document, "all_objects_rotation_Y"),
            {
                "order": 4,
                "bones": ["all_objects_layer_0", "all_objects_layer_1"],
                "target": "all_objects_rotation_Y",
                "local": True,
                "relative": True,
                "mixX": 0,
                "mixY": 0,
                "mixScaleX": 0,
                "mixScaleY": 0,
                "mixShearY": 0,
            },
            "all_objects_rotation_Y",
        )

        expected_orders = {
            "all_objects_rotation_X_constraint": 0,
            "all_objects_IK": 1,
            "all_objects_scale": 2,
            "all_objects_scale_rotate_X_constraint": 3,
            "all_objects_rotation_Y": 4,
            "ObjectB_rotation_X_constraint": 5,
            "ObjectA_rotation_X_constraint": 6,
            "ObjectB_IK": 7,
            "ObjectA_IK": 8,
            "ObjectB_scale": 9,
            "ObjectA_scale": 10,
            "ObjectB_scale_rotate_X_constraint": 11,
            "ObjectA_scale_rotate_X_constraint": 12,
            "ObjectB_rotation_Y": 13,
            "ObjectA_rotation_Y": 14,
        }
        constraints = tuple(document.get("ik", ())) + tuple(
            document.get("transform", ())
        )
        names = {item["name"] for item in constraints}
        _assert(
            names == set(expected_orders),
            f"unexpected connected two-axis constraints: {sorted(names)}",
        )
        for name, expected_order in expected_orders.items():
            constraint = _constraint(document, name)
            _assert(
                int(constraint["order"]) == expected_order,
                f"{name} order {constraint['order']} != {expected_order}",
            )
        orders = tuple(int(item["order"]) for item in constraints)
        _assert(
            len(orders) == len(set(orders)) == 15,
            f"connected two-axis constraint orders collide: {orders}",
        )
        _assert(
            set(orders) == set(range(15)),
            f"connected two-axis schedule is not dense: {orders}",
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
