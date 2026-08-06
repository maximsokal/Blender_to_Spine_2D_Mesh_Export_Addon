"""Validate Object Root depth setup compensation on the real coin asset.

This gate exists because comparing projected geometry and baked texture alone is not
sufficient. Active Camera Object Root may retain the correct projected vertices while its
Spine depth constraint places every camera-depth group directly into the setup pose,
visibly stretching or flattening the mesh. The real exported rig must therefore preserve
neutral camera-facing X/Y rotations while retaining ordinary Object Root depth
compensation.
"""

from __future__ import annotations

import argparse
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

from Blender_to_Spine2D_Mesh_Exporter.blender_adapter import (  # noqa: E402
    export_a1_single_object,
    prepare_a1_object,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.rig_profiles import (  # noqa: E402
    A1RigSetupPoseMode,
)
from run_bake_integration import (  # noqa: E402
    _assert,
    _capture_scene_bake_state,
    _temporary_datablock_names,
)
from run_coin_star_normal_camera_root_modes_integration import (  # noqa: E402
    _two_axis_settings,
)
from run_coin_star_normal_projection_parity_integration import (  # noqa: E402
    _single_json_and_png,
)
from run_coin_star_real_blend_shader_capability_integration import (  # noqa: E402
    _datablock_fingerprint,
    _object_fingerprint,
    _require_loaded_blend,
    _require_source_object,
    _scene_fingerprint,
)


_FLOAT_TOLERANCE = 0.011


def _parse_arguments() -> argparse.Namespace:
    arguments = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(
        description="Validate real coin Active Camera Object Root setup compensation."
    )
    parser.add_argument(
        "--expected-blend",
        required=True,
        help="Exact coin_star.blend path Blender must already have loaded.",
    )
    return parser.parse_args(arguments)


def _constraint_by_name(rig, name: str):
    matches = tuple(item for item in rig.transform if item.name == name)
    _assert(
        len(matches) == 1,
        f"expected one transform constraint {name!r}, found {len(matches)}",
    )
    return matches[0]


def _serialized_constraint(document: dict[str, object], name: str) -> dict[str, object]:
    constraints = document.get("transform")
    _assert(isinstance(constraints, list), "serialized transform constraints are missing")
    matches = tuple(
        item
        for item in constraints
        if isinstance(item, dict) and item.get("name") == name
    )
    _assert(
        len(matches) == 1,
        f"expected one serialized transform constraint {name!r}, found {len(matches)}",
    )
    return matches[0]


def _assert_close(actual: object, expected: float, message: str) -> None:
    _assert(
        isinstance(actual, (int, float)) and not isinstance(actual, bool),
        f"{message}: non-numeric value {actual!r}",
    )
    _assert(
        abs(float(actual) - float(expected)) <= _FLOAT_TOLERANCE,
        f"{message}: actual={actual!r}, expected={expected!r}",
    )


def _run(expected_blend: str) -> None:
    loaded = _require_loaded_blend(expected_blend)
    source = _require_source_object()
    _assert(
        bpy.context.scene.camera is not None,
        "Object Root setup gate requires an active scene camera",
    )

    scene_before = _scene_fingerprint()
    bake_before = _capture_scene_bake_state()
    object_before = _object_fingerprint(source)
    datablocks_before = _datablock_fingerprint()
    temporary_before = _temporary_datablock_names()

    with tempfile.TemporaryDirectory(
        prefix="spine2d-coin-normal-object-root-setup-"
    ) as directory:
        output_directory = Path(directory)
        settings = _two_axis_settings(
            output_directory,
            A1RigSetupPoseMode.PRESERVE_COMPOSITION,
        )
        prepared = prepare_a1_object(
            source,
            settings,
            context=bpy.context,
            scene=bpy.context.scene,
        )

        rig = prepared.rig
        profile = rig.profile
        _assert(
            rig.request.setup_pose_mode is A1RigSetupPoseMode.CAMERA_VIEW_NORMAL,
            "Active Camera Object Root did not normalize to CAMERA_VIEW_NORMAL",
        )
        _assert(
            len(rig.info.z_groups) > 1,
            "Active Camera Object Root lost its per-depth groups",
        )

        rotation_x = _constraint_by_name(
            rig,
            profile.rotation_x_constraint(rig.request.prefix),
        )
        rotation_y = _constraint_by_name(
            rig,
            profile.rotation_y_constraint(rig.request.prefix),
        )
        depth = _constraint_by_name(
            rig,
            profile.scale_depth_constraint(rig.request.prefix),
        )
        minimum_depth_y = min(
            float(group.y_offset_pixels) for group in rig.info.z_groups
        )

        _assert_close(
            rotation_x.extras.get("rotation"),
            0.0,
            "Object Root setup X rotation is not neutral",
        )
        _assert_close(
            rotation_y.extras.get("rotation"),
            0.0,
            "Object Root setup Y rotation is not neutral",
        )
        _assert_close(
            depth.extras.get("x"),
            minimum_depth_y,
            "Object Root depth translation compensation is wrong",
        )
        _assert_close(
            depth.extras.get("scaleX"),
            -1.0,
            "Object Root depth scale compensation is wrong",
        )

        result = export_a1_single_object(
            source,
            settings,
            context=bpy.context,
            scene=bpy.context.scene,
        )
        _assert(result.success, f"Object Root export failed: {result.issues}")
        json_path, _png_path = _single_json_and_png(result)
        document = json.loads(json_path.read_text(encoding="utf-8"))
        serialized_depth = _serialized_constraint(document, depth.name)
        _assert_close(
            serialized_depth.get("x"),
            minimum_depth_y,
            "serialized Object Root depth translation compensation is wrong",
        )
        _assert_close(
            serialized_depth.get("scaleX"),
            -1.0,
            "serialized Object Root depth scale compensation is wrong",
        )

        print(
            "[COIN-NORMAL-OBJECT-ROOT-SETUP] PASS "
            f"blend={loaded} object={source.name_full!r} "
            f"depth_groups={len(rig.info.z_groups)} "
            f"minimum_depth_y={minimum_depth_y:.6f} "
            "rotation_x=0 rotation_y=0 depth_x=minDepth depth_scale_x=-1",
            flush=True,
        )

    _assert(_scene_fingerprint() == scene_before, "setup gate changed Blender context")
    _assert(_capture_scene_bake_state() == bake_before, "setup gate changed bake state")
    _assert(_object_fingerprint(source) == object_before, "setup gate changed source data")
    _assert(
        _datablock_fingerprint() == datablocks_before,
        "setup gate created or removed persistent Blender datablocks",
    )
    _assert(
        _temporary_datablock_names() == temporary_before,
        "setup gate leaked temporary Blender datablocks",
    )


def main() -> None:
    arguments = _parse_arguments()
    try:
        _run(arguments.expected_blend)
    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
