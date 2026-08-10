"""Validate standalone signed-axis Object-Root setup placement on real grenade.blend.

This focused gate performs preparation only: no texture bake, serialization, or file
commit. It exists to catch whole-object setup drift that can be visually mistaken for a
wrong root position after otherwise-correct Normal / UV geometry and textures are merged
into one Spine document.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
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

import Blender_to_Spine2D_Mesh_Exporter as addon  # noqa: E402
from Blender_to_Spine2D_Mesh_Exporter.application import (  # noqa: E402
    A1MultiObjectExportSettings,
    A1MultiObjectMode,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter import (  # noqa: E402
    A1MultiObjectSource,
    prepare_a1_multi_object,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.a1_ui_scene_capture import (  # noqa: E402
    _capture_scene_profile,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.a1_ui_selection import (  # noqa: E402
    _capture_object_profile,
    _connect_enabled,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.a1_ui_settings import (  # noqa: E402
    _settings_from_profiles,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import (  # noqa: E402
    A1TextureExportMode,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.projection import (  # noqa: E402
    A1ProjectionDirection,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine import (  # noqa: E402
    A1RigSetupPoseMode,
    calculate_uniform_scale,
)
from run_bake_integration import _assert  # noqa: E402
from run_grenade_bump_displacement_normal_uv_integration import (  # noqa: E402
    _require_loaded_blend,
)


_MINIMUM_MESH_OBJECTS = 2


def _parse_arguments() -> argparse.Namespace:
    arguments = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(
        description="Validate real grenade standalone projected Object-Root setup pose."
    )
    parser.add_argument(
        "--expected-blend",
        required=True,
        help="Exact grenade.blend path Blender must already have loaded.",
    )
    return parser.parse_args(arguments)


def _register_steps() -> list[tuple]:
    completed: list[tuple] = []
    try:
        for step in addon.REGISTRATION_STEPS:
            step[1]()
            completed.append(step)
        return completed
    except Exception:
        for step in reversed(completed):
            try:
                step[2]()
            except Exception:
                traceback.print_exc()
        raise


def _unregister_steps(completed: list[tuple]) -> None:
    failures: list[str] = []
    for label, _register, unregister in reversed(completed):
        try:
            unregister()
        except Exception as exc:
            failures.append(f"{label}: {exc}")
    _assert(not failures, f"Rewrite unregister failures: {failures!r}")


def _mesh_objects(scene) -> tuple:
    objects = tuple(
        sorted(
            (
                obj
                for obj in scene.objects
                if getattr(obj, "type", None) == "MESH"
                and getattr(obj, "data", None) is not None
            ),
            key=lambda obj: obj.name_full,
        )
    )
    _assert(
        len(objects) >= _MINIMUM_MESH_OBJECTS,
        f"grenade fixture contains too few Mesh objects: {len(objects)}",
    )
    return objects


def _object_fingerprint(obj) -> tuple:
    uv_layers = getattr(obj.data, "uv_layers", None)
    layer_names = tuple(layer.name for layer in uv_layers) if uv_layers is not None else ()
    active_uv = (
        None
        if uv_layers is None or getattr(uv_layers, "active", None) is None
        else uv_layers.active.name
    )
    return (
        obj.name_full,
        obj.data.name_full,
        tuple(tuple(float(value) for value in row) for row in obj.matrix_world),
        None if obj.parent is None else obj.parent.name_full,
        tuple(
            None if slot.material is None else slot.material.name_full
            for slot in obj.material_slots
        ),
        layer_names,
        active_uv,
        len(obj.data.vertices),
        len(obj.data.edges),
        len(obj.data.polygons),
    )


def _scene_fingerprint(scene, objects: tuple) -> tuple:
    return (
        tuple(_object_fingerprint(obj) for obj in objects),
        None if scene.camera is None else scene.camera.name_full,
        int(scene.frame_current),
        str(scene.render.engine),
        tuple(sorted(obj.name_full for obj in bpy.context.selected_objects)),
        (
            None
            if bpy.context.view_layer.objects.active is None
            else bpy.context.view_layer.objects.active.name_full
        ),
    )


def _datablock_fingerprint() -> tuple:
    return (
        tuple(sorted(item.name_full for item in bpy.data.objects)),
        tuple(sorted(item.name_full for item in bpy.data.meshes)),
        tuple(sorted(item.name_full for item in bpy.data.materials)),
        tuple(sorted(item.name_full for item in bpy.data.images)),
    )


def _object_settings(obj, scene_profile):
    bake = getattr(obj, "spine2d_bake_settings", None)
    object_profile = _capture_object_profile(
        obj,
        sequence_start_frame=int(getattr(bake, "bake_frame_start", 0)),
        sequence_frame_count=int(getattr(bake, "frames_for_render", 0)),
        connect_enabled=_connect_enabled(obj),
    )
    return _settings_from_profiles(
        object_profile,
        scene_profile,
        rig_setup_pose_mode=A1RigSetupPoseMode.PRESERVE_COMPOSITION,
    )


def _source(obj, settings, index: int) -> A1MultiObjectSource:
    return A1MultiObjectSource(
        source_object=obj,
        component_id=f"object_{index}:{obj.name_full}",
        animation_namespace=f"object_{index}",
        settings=settings,
    )


def _bone_by_name(prepared, name: str):
    matches = tuple(bone for bone in prepared.rig.bones if bone.name == name)
    _assert(len(matches) == 1, f"expected one rig bone {name!r}; found={len(matches)}")
    return matches[0]


def _assert_neutral_projected_object(prepared) -> tuple[float, float, float, float]:
    settings = prepared.settings
    _assert(
        settings.rig_setup_pose_mode is A1RigSetupPoseMode.CAMERA_VIEW_NORMAL,
        "standalone signed-axis Normal/UV preparation did not select neutral projected "
        f"Object-Root setup for {prepared.object_id!r}: "
        f"actual={settings.rig_setup_pose_mode.value}",
    )
    _assert(
        settings.projection_direction.axis_aligned,
        f"focused grenade setup gate requires signed-axis projection: {settings.projection_direction}",
    )

    statistics = dict(prepared.statistics)
    projected_u = float(statistics["projected_origin_u"])
    projected_v = float(statistics["projected_origin_v"])
    uniform_scale = calculate_uniform_scale(
        settings.export.texture_width,
        settings.export.texture_height,
        settings.rig_scale_mode,
    )
    expected_x = round(projected_u * uniform_scale, 2)
    expected_y = round(projected_v * uniform_scale, 2)

    main = _bone_by_name(prepared, f"{prepared.prefix}_main")
    actual_x = 0.0 if main.x is None else float(main.x)
    actual_y = 0.0 if main.y is None else float(main.y)
    _assert(
        (actual_x, actual_y) == (expected_x, expected_y),
        "projected Object Origin and generated main bone disagree: "
        f"object={prepared.object_id!r}, projected=({projected_u}, {projected_v}), "
        f"scale={uniform_scale}, expected=({expected_x}, {expected_y}), "
        f"actual=({actual_x}, {actual_y})",
    )

    transform = tuple(prepared.rig.transform)
    _assert(len(transform) >= 4, f"unexpected transform constraint count: {len(transform)}")
    rotate_x, rotate_y, _scale, depth = transform[:4]
    _assert(
        float(rotate_x.extras.get("rotation", 0.0)) == 0.0,
        f"{prepared.object_id!r} has non-neutral X setup rotation: {rotate_x.extras!r}",
    )
    _assert(
        float(rotate_y.extras.get("rotation", 0.0)) == 0.0,
        f"{prepared.object_id!r} has non-neutral Y setup rotation: {rotate_y.extras!r}",
    )
    _assert(
        float(depth.extras.get("x", 0.0)) == 0.0,
        f"{prepared.object_id!r} has non-neutral depth setup translation: {depth.extras!r}",
    )
    _assert(
        float(depth.extras.get("scaleX", 0.0)) == 0.0,
        f"{prepared.object_id!r} has non-neutral depth setup scale: {depth.extras!r}",
    )

    available_bones = {bone.name for bone in prepared.rig.bones}
    inverse_setup_names = tuple(
        prepared.rig.profile.z_camera_setup_bone(prepared.prefix, group.index)
        for group in prepared.rig.info.z_groups
    )
    _assert(
        inverse_setup_names
        and all(name in available_bones for name in inverse_setup_names),
        f"{prepared.object_id!r} is missing inverse projected setup bones",
    )
    return projected_u, projected_v, actual_x, actual_y


def _run(expected_blend: str) -> None:
    loaded = _require_loaded_blend(expected_blend)
    completed = _register_steps()
    try:
        scene = bpy.context.scene
        objects = _mesh_objects(scene)
        before = _scene_fingerprint(scene, objects)
        datablocks_before = _datablock_fingerprint()

        with tempfile.TemporaryDirectory(prefix="spine2d_grenade_setup_pose_") as root:
            output_directory = Path(root).resolve(strict=False)
            scene_profile = _capture_scene_profile(
                scene,
                output_directory=output_directory,
                images_relative_path="images",
            )
            _assert(
                scene_profile.texture_export_mode is A1TextureExportMode.NORMAL_UV_SEGMENTS,
                "grenade setup regression requires persisted Normal / UV Segments mode; "
                f"actual={scene_profile.texture_export_mode.value}",
            )
            _assert(
                isinstance(scene_profile.projection_direction, A1ProjectionDirection)
                and scene_profile.projection_direction.axis_aligned,
                "grenade setup regression requires a persisted signed-axis projection; "
                f"actual={scene_profile.projection_direction!r}",
            )

            sources = tuple(
                _source(obj, _object_settings(obj, scene_profile), index)
                for index, obj in enumerate(objects, start=1)
            )
            prepared = prepare_a1_multi_object(
                sources,
                A1MultiObjectExportSettings(
                    output_directory=output_directory,
                    output_stem="Grenade_Standalone_Setup_Pose",
                    mode=A1MultiObjectMode.STANDALONE,
                ),
                context=bpy.context,
                scene=scene,
            )

            _assert(
                len(prepared.objects) == len(objects),
                "prepared object count differs from source Mesh object count",
            )
            placements = tuple(
                (item.object_id, *_assert_neutral_projected_object(item))
                for item in prepared.objects
            )

        _assert(
            _scene_fingerprint(scene, objects) == before,
            "standalone setup preparation changed source object/scene/context state",
        )
        _assert(
            _datablock_fingerprint() == datablocks_before,
            "standalone setup preparation leaked or removed Blender datablocks",
        )

        print(
            "[GRENADE-STANDALONE-SETUP-POSE] PASS "
            f"blend={loaded} mesh_objects={len(objects)} "
            f"projection={scene_profile.projection_direction.value!r} "
            f"setup={A1RigSetupPoseMode.CAMERA_VIEW_NORMAL.value!r} "
            f"placements={placements!r} source=unchanged",
            flush=True,
        )
    finally:
        _unregister_steps(completed)


def main() -> None:
    arguments = _parse_arguments()
    _run(arguments.expected_blend)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
