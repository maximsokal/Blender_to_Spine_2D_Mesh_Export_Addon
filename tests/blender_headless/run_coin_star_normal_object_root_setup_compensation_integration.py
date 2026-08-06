"""Validate complete Active Camera Object Root setup chains on the real coin.

Geometry/texture parity is insufficient for this mode: weighted vertices are evaluated
through ``depth scale -> depth rotation -> inverse setup -> vertex``. This gate checks the
entire typed and serialized hierarchy, requires a full-rank neutral depth constraint, and
proves that every group's authored depth translation is exactly cancelled before the
projected vertex X/Y is applied.
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
        description="Validate real coin Active Camera Object Root inverse setup."
    )
    parser.add_argument(
        "--expected-blend",
        required=True,
        help="Exact coin_star.blend path Blender must already have loaded.",
    )
    return parser.parse_args(arguments)


def _assert_close(actual: object, expected: float, message: str) -> None:
    _assert(
        isinstance(actual, (int, float)) and not isinstance(actual, bool),
        f"{message}: non-numeric value {actual!r}",
    )
    _assert(
        abs(float(actual) - float(expected)) <= _FLOAT_TOLERANCE,
        f"{message}: actual={actual!r}, expected={expected!r}",
    )


def _typed_bones_by_name(prepared) -> dict[str, object]:
    document = prepared.document_assembly.document_build.document
    result = {bone.name: bone for bone in document.bones}
    _assert(
        len(result) == len(document.bones),
        "typed Object Root document contains duplicate bone names",
    )
    return result


def _serialized_bones_by_name(document: dict[str, object]) -> dict[str, dict[str, object]]:
    raw_bones = document.get("bones")
    _assert(isinstance(raw_bones, list), "serialized Object Root bones are missing")
    result: dict[str, dict[str, object]] = {}
    for index, raw in enumerate(raw_bones):
        _assert(isinstance(raw, dict), f"bones[{index}] must be a mapping")
        name = raw.get("name")
        _assert(
            isinstance(name, str) and bool(name.strip()),
            f"bones[{index}].name must be non-empty",
        )
        _assert(name not in result, f"duplicate serialized bone name: {name}")
        result[name] = raw
    return result


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


def _assert_typed_inverse_setup(prepared) -> tuple[int, int]:
    rig = prepared.rig
    bones = _typed_bones_by_name(prepared)
    groups_by_index = {group.index: group for group in rig.info.z_groups}
    compensation_names: dict[int, str] = {}

    for group in rig.info.z_groups:
        scale_bone = bones[group.scale_bone_name]
        depth_bone = bones[group.bone_name]
        compensation_name = rig.profile.z_camera_setup_bone(
            rig.info.prefix,
            group.index,
        )
        compensation = bones[compensation_name]
        compensation_names[group.index] = compensation_name

        _assert(scale_bone.parent == rig.info.main_rotation_bone_name, "wrong depth scale parent")
        _assert_close(scale_bone.y, group.y_offset_pixels, "wrong depth setup translation")
        _assert_close(scale_bone.rotation, 90.0, "wrong depth scale setup rotation")
        _assert(
            scale_bone.extras.get("inherit") == "onlyTranslation",
            "depth scale bone lost onlyTranslation inheritance",
        )
        _assert(depth_bone.parent == group.scale_bone_name, "wrong depth rotation parent")
        _assert_close(depth_bone.rotation, -90.0, "wrong depth child setup rotation")
        _assert(compensation.parent == group.bone_name, "wrong inverse setup parent")
        _assert_close(
            compensation.y,
            -float(group.y_offset_pixels),
            "inverse setup translation does not cancel depth",
        )
        _assert_close(
            float(scale_bone.y or 0.0) + float(compensation.y or 0.0),
            0.0,
            "typed depth and inverse setup translations do not sum to zero",
        )

    vertex_count = 0
    for component in prepared.document_assembly.document_build.components:
        _assert(
            len(component.request.vertices) == len(component.vertex_bones),
            "typed component vertex/request counts differ",
        )
        for request_vertex, vertex_bone in zip(
            component.request.vertices,
            component.vertex_bones,
            strict=True,
        ):
            _assert(
                request_vertex.z_group_index in groups_by_index,
                "typed vertex references unknown depth group",
            )
            _assert(
                vertex_bone.parent
                == compensation_names[request_vertex.z_group_index],
                "typed vertex bypasses its inverse setup bone",
            )
            _assert_close(
                vertex_bone.x,
                request_vertex.bone_position_pixels[0],
                "typed vertex X changed after projection",
            )
            _assert_close(
                vertex_bone.y,
                request_vertex.bone_position_pixels[1],
                "typed vertex Y changed after projection",
            )
            vertex_count += 1

    _assert(vertex_count > 0, "typed Object Root export contains no weighted vertices")
    return len(compensation_names), vertex_count


def _assert_serialized_inverse_setup(
    prepared,
    document: dict[str, object],
) -> tuple[int, int]:
    rig = prepared.rig
    bones = _serialized_bones_by_name(document)
    compensation_names: dict[int, str] = {}

    for group in rig.info.z_groups:
        scale_bone = bones[group.scale_bone_name]
        depth_bone = bones[group.bone_name]
        compensation_name = rig.profile.z_camera_setup_bone(
            rig.info.prefix,
            group.index,
        )
        compensation = bones[compensation_name]
        compensation_names[group.index] = compensation_name

        _assert(scale_bone.get("parent") == rig.info.main_rotation_bone_name, "serialized wrong depth scale parent")
        _assert_close(scale_bone.get("y", 0.0), group.y_offset_pixels, "serialized wrong depth translation")
        _assert_close(scale_bone.get("rotation", 0.0), 90.0, "serialized wrong depth scale rotation")
        _assert(scale_bone.get("inherit") == "onlyTranslation", "serialized depth scale lost onlyTranslation")
        _assert(depth_bone.get("parent") == group.scale_bone_name, "serialized wrong depth child parent")
        _assert_close(depth_bone.get("rotation", 0.0), -90.0, "serialized wrong depth child rotation")
        _assert(compensation.get("parent") == group.bone_name, "serialized wrong inverse setup parent")
        _assert_close(
            compensation.get("y", 0.0),
            -float(group.y_offset_pixels),
            "serialized inverse setup translation is wrong",
        )
        _assert_close(
            float(scale_bone.get("y", 0.0)) + float(compensation.get("y", 0.0)),
            0.0,
            "serialized depth and inverse setup translations do not sum to zero",
        )

    vertex_count = 0
    for component in prepared.document_assembly.document_build.components:
        for request_vertex, vertex_bone in zip(
            component.request.vertices,
            component.vertex_bones,
            strict=True,
        ):
            serialized = bones[vertex_bone.name]
            _assert(
                serialized.get("parent")
                == compensation_names[request_vertex.z_group_index],
                "serialized vertex bypasses inverse setup bone",
            )
            _assert_close(
                serialized.get("x", 0.0),
                request_vertex.bone_position_pixels[0],
                "serialized vertex X changed after projection",
            )
            _assert_close(
                serialized.get("y", 0.0),
                request_vertex.bone_position_pixels[1],
                "serialized vertex Y changed after projection",
            )
            vertex_count += 1

    return len(compensation_names), vertex_count


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
        settings = _two_axis_settings(
            Path(directory),
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
        _assert(len(rig.info.z_groups) > 1, "Object Root lost per-depth groups")

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
        _assert_close(rotation_x.extras.get("rotation"), 0.0, "setup X rotation is not neutral")
        _assert_close(rotation_y.extras.get("rotation"), 0.0, "setup Y rotation is not neutral")
        _assert_close(depth.extras.get("x"), 0.0, "depth setup translation is not neutral")
        _assert_close(depth.extras.get("scaleX"), 0.0, "depth setup scale is not full-rank")

        typed_compensations, typed_vertices = _assert_typed_inverse_setup(prepared)

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
        _assert_close(serialized_depth.get("x", 0.0), 0.0, "serialized depth translation is not neutral")
        _assert_close(serialized_depth.get("scaleX", 0.0), 0.0, "serialized depth setup is singular")

        serialized_compensations, serialized_vertices = _assert_serialized_inverse_setup(
            prepared,
            document,
        )
        _assert(typed_compensations == serialized_compensations, "typed/serialized compensation counts differ")
        _assert(typed_vertices == serialized_vertices, "typed/serialized vertex counts differ")

        print(
            "[COIN-NORMAL-OBJECT-ROOT-SETUP] PASS "
            f"blend={loaded} object={source.name_full!r} "
            f"depth_groups={len(rig.info.z_groups)} "
            f"inverse_setup_bones={serialized_compensations} "
            f"weighted_vertices={serialized_vertices} "
            "rotation_x=0 rotation_y=0 depth_x=0 depth_scale_x=0 "
            "setup_chain=depth+inverse+projected_xy",
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
