"""Validate both Normal / UV Active Camera rig roots on the real coin asset.

Blender must open the caller-provided ``coin_star.blend`` before this script starts.
Both variants use identical active-camera projected geometry and projection-independent
material-bake geometry. Only the generated Spine root/depth hierarchy may differ:

* Object Root Bone keeps Blender Object Origin and all camera-depth groups;
* Camera Root Bone keeps main at camera-space zero and one rigid object layer.
"""

from __future__ import annotations

import argparse
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

from Blender_to_Spine2D_Mesh_Exporter.blender_adapter import (  # noqa: E402
    export_a1_single_object,
    prepare_a1_object,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.projection import (  # noqa: E402
    A1ProjectionDirection,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.rig_profiles import (  # noqa: E402
    A1CameraLayerProjectionKind,
    A1RigProfile,
    A1RigSetupPoseMode,
)
from run_bake_integration import (  # noqa: E402
    _assert,
    _capture_scene_bake_state,
    _temporary_datablock_names,
)
from run_coin_star_normal_projection_parity_integration import (  # noqa: E402
    _assert_luminance_parity,
    _mesh_uv_stream_count,
    _settings,
    _single_json_and_png,
    _source_material_geometry_fingerprint,
    _visible_luminance_metrics,
)
from run_coin_star_real_blend_shader_capability_integration import (  # noqa: E402
    _datablock_fingerprint,
    _object_fingerprint,
    _require_loaded_blend,
    _require_source_object,
    _scene_fingerprint,
)


_POSITION_TOLERANCE = 0.011


def _parse_arguments() -> argparse.Namespace:
    arguments = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(
        description="Validate real coin Normal Active Camera Object/Camera roots."
    )
    parser.add_argument(
        "--expected-blend",
        required=True,
        help="Exact coin_star.blend path Blender must already have loaded.",
    )
    return parser.parse_args(arguments)


def _two_axis_settings(
    output_directory: Path,
    setup_pose_mode: A1RigSetupPoseMode,
):
    """Reuse the parity fixture while selecting the public production rig profile."""

    if not isinstance(output_directory, Path):
        raise TypeError("output_directory must be pathlib.Path")
    if not isinstance(setup_pose_mode, A1RigSetupPoseMode):
        raise TypeError("setup_pose_mode must be A1RigSetupPoseMode")
    base = _settings(
        output_directory,
        A1ProjectionDirection.ACTIVE_CAMERA,
    )
    return replace(
        base,
        export=replace(
            base.export,
            rig_profile=A1RigProfile.TWO_AXIS_ROTATION_SCALE.value,
        ),
        rig_setup_pose_mode=setup_pose_mode,
    )


def _bone_by_name(rig, name: str):
    matches = tuple(bone for bone in rig.bones if bone.name == name)
    _assert(len(matches) == 1, f"expected one bone {name!r}, found {len(matches)}")
    return matches[0]


def _camera_root_world_position(prepared) -> tuple[float, float]:
    rig = prepared.rig
    _assert(
        len(rig.info.z_groups) == 1,
        f"Camera Root rig must have one depth group: {rig.info.z_groups}",
    )
    group = rig.info.z_groups[0]
    base = _bone_by_name(rig, rig.info.base_bone_name)
    _assert(
        base.parent == group.bone_name,
        "Camera Root base is not parented to its rigid depth group",
    )
    return (
        float(base.x or 0.0),
        float(group.y_offset_pixels) + float(base.y or 0.0),
    )


def _run(expected_blend: str) -> None:
    loaded = _require_loaded_blend(expected_blend)
    source = _require_source_object()
    _assert(
        bpy.context.scene.camera is not None,
        "Active Camera root-mode gate requires an active scene camera",
    )

    scene_before = _scene_fingerprint()
    bake_before = _capture_scene_bake_state()
    object_before = _object_fingerprint(source)
    datablocks_before = _datablock_fingerprint()
    temporary_before = _temporary_datablock_names()

    with tempfile.TemporaryDirectory(
        prefix="spine2d-coin-normal-camera-root-modes-"
    ) as directory:
        root = Path(directory)
        object_root_settings = _two_axis_settings(
            root / "object-root",
            A1RigSetupPoseMode.PRESERVE_COMPOSITION,
        )
        camera_root_settings = _two_axis_settings(
            root / "camera-root",
            A1RigSetupPoseMode.PREPROJECTED_SCREEN,
        )

        object_root = prepare_a1_object(
            source,
            object_root_settings,
            context=bpy.context,
            scene=bpy.context.scene,
        )
        camera_root = prepare_a1_object(
            source,
            camera_root_settings,
            context=bpy.context,
            scene=bpy.context.scene,
        )

        _assert(
            object_root.source_snapshot == camera_root.source_snapshot,
            "Active Camera root choice changed projected export geometry",
        )
        _assert(
            _source_material_geometry_fingerprint(object_root.bake_target_snapshot)
            == _source_material_geometry_fingerprint(camera_root.bake_target_snapshot),
            "Active Camera root choice changed material-bake geometry",
        )
        _assert(
            len(object_root.document_assembly.projections)
            == len(camera_root.document_assembly.projections)
            and len(object_root.document_assembly.projections) > 8,
            "Active Camera root choice changed retained Normal regions: "
            f"object={len(object_root.document_assembly.projections)}, "
            f"camera={len(camera_root.document_assembly.projections)}",
        )

        _assert(
            object_root.rig.request.setup_pose_mode
            is A1RigSetupPoseMode.CAMERA_VIEW_NORMAL,
            "Object Root mode did not retain CAMERA_VIEW_NORMAL",
        )
        _assert(
            object_root.rig.request.camera_layer_projection_kind is None,
            "Object Root mode retained rigid camera-layer semantics",
        )
        _assert(
            len(object_root.rig.info.z_groups) > 1,
            "Object Root mode lost per-vertex camera-depth groups",
        )
        _assert(
            len(object_root.z_groups.groups) == len(object_root.rig.info.z_groups),
            "Object Root public Z plan disagrees with its rig",
        )

        _assert(
            camera_root.rig.request.setup_pose_mode
            is A1RigSetupPoseMode.PREPROJECTED_SCREEN,
            "Camera Root mode did not use PREPROJECTED_SCREEN",
        )
        _assert(
            isinstance(
                camera_root.rig.request.camera_layer_projection_kind,
                A1CameraLayerProjectionKind,
            ),
            "Camera Root mode lost Perspective/Orthographic layer semantics",
        )
        _assert(
            len(camera_root.rig.info.z_groups) == 1
            and len(camera_root.z_groups.groups) == 1,
            "Camera Root mode did not collapse to one rigid depth layer",
        )
        main = _bone_by_name(camera_root.rig, camera_root.rig.info.main_bone_name)
        _assert(
            abs(float(main.x or 0.0)) <= _POSITION_TOLERANCE
            and abs(float(main.y or 0.0)) <= _POSITION_TOLERANCE,
            f"Camera Root main is not at camera-space zero: {(main.x, main.y)}",
        )
        actual_origin = _camera_root_world_position(camera_root)
        expected_origin = camera_root.rig.request.main_position_pixels
        _assert(expected_origin is not None, "Camera Root lost projected Object Origin")
        _assert(
            abs(actual_origin[0] - float(expected_origin[0])) <= _POSITION_TOLERANCE
            and abs(actual_origin[1] - float(expected_origin[1]))
            <= _POSITION_TOLERANCE,
            "Camera Root base does not reconstruct projected Object Origin: "
            f"actual={actual_origin}, expected={expected_origin}",
        )

        object_result = export_a1_single_object(
            source,
            object_root_settings,
            context=bpy.context,
            scene=bpy.context.scene,
        )
        camera_result = export_a1_single_object(
            source,
            camera_root_settings,
            context=bpy.context,
            scene=bpy.context.scene,
        )
        _assert(
            object_result.success,
            f"Object Root export failed: {object_result.issues}",
        )
        _assert(
            camera_result.success,
            f"Camera Root export failed: {camera_result.issues}",
        )

        object_json, object_png = _single_json_and_png(object_result)
        camera_json, camera_png = _single_json_and_png(camera_result)
        object_document = json.loads(object_json.read_text(encoding="utf-8"))
        camera_document = json.loads(camera_json.read_text(encoding="utf-8"))
        object_meshes = _mesh_uv_stream_count(object_document)
        camera_meshes = _mesh_uv_stream_count(camera_document)
        _assert(
            object_meshes == camera_meshes
            and object_meshes == len(object_root.document_assembly.projections),
            "Serialized mesh ownership differs between Active Camera roots: "
            f"object={object_meshes}, camera={camera_meshes}",
        )

        _assert(
            object_result.statistics.get("normal_active_camera_root_mode")
            == "OBJECT_ROOT",
            f"Object Root statistics are wrong: {object_result.statistics}",
        )
        _assert(
            camera_result.statistics.get("normal_active_camera_root_mode")
            == "CAMERA_ROOT",
            f"Camera Root statistics are wrong: {camera_result.statistics}",
        )
        _assert(
            camera_result.statistics.get("camera_relative_depth_group_count") == 1,
            f"Camera Root relative depth statistics are wrong: {camera_result.statistics}",
        )
        _assert(
            camera_result.statistics.get("normal_active_camera_depth_group_count") == 0,
            f"Camera Root retained object-root depth statistics: {camera_result.statistics}",
        )
        _assert(
            camera_result.statistics.get("depth_setup_y_compensated") == 1,
            f"Camera Root setup compensation was not reported: {camera_result.statistics}",
        )
        _assert(
            object_result.statistics.get("depth_setup_y_compensated") == 0,
            f"Object Root incorrectly enabled camera compensation: {object_result.statistics}",
        )
        _assert(
            object_result.statistics.get("bake_strategy_ids")
            == camera_result.statistics.get("bake_strategy_ids"),
            "Active Camera root choice changed material bake strategy",
        )

        object_luma = _visible_luminance_metrics(object_png)
        camera_luma = _visible_luminance_metrics(camera_png)
        _assert_luminance_parity(object_luma, camera_luma)

        print(
            "[COIN-NORMAL-ACTIVE-CAMERA-ROOTS] PASS "
            f"blend={loaded} object={source.name_full!r} "
            f"segments={object_meshes} "
            f"object_depth_groups={len(object_root.rig.info.z_groups)} "
            f"camera_depth_groups={len(camera_root.rig.info.z_groups)} "
            f"camera_kind={camera_root.rig.request.camera_layer_projection_kind.value} "
            f"object_luma=({object_luma[1]:.6f},{object_luma[2]:.6f},"
            f"{object_luma[3]:.6f}) "
            f"camera_luma=({camera_luma[1]:.6f},{camera_luma[2]:.6f},"
            f"{camera_luma[3]:.6f}) "
            "geometry=shared material_geometry=projection-independent "
            "object_setup=CAMERA_VIEW_NORMAL camera_setup=PREPROJECTED_SCREEN",
            flush=True,
        )

    _assert(_scene_fingerprint() == scene_before, "root-mode export changed Blender context")
    _assert(_capture_scene_bake_state() == bake_before, "root-mode export changed bake state")
    _assert(_object_fingerprint(source) == object_before, "root-mode export changed source data")
    _assert(
        _datablock_fingerprint() == datablocks_before,
        "root-mode export created or removed persistent Blender datablocks",
    )
    _assert(
        _temporary_datablock_names() == temporary_before,
        "root-mode export leaked temporary Blender datablocks",
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
