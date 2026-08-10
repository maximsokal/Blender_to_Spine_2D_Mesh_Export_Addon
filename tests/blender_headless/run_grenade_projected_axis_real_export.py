"""Export the current real grenade.blend through projected-axis standalone Normal/UV.

This is a manual-release regression runner, not a synthetic fixture. It captures the
persisted UI profile from the loaded Blender Scene, captures every current Mesh object's
UI profile, validates the generated two-axis rig before baking, and then executes the
same production standalone multi-object exporter used by the UI. Outputs are intentionally
kept in a caller-provided directory so the resulting JSON can be opened in Spine and its
live X/Y controls can be inspected manually.
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

import Blender_to_Spine2D_Mesh_Exporter as addon  # noqa: E402
from Blender_to_Spine2D_Mesh_Exporter.application import (  # noqa: E402
    A1MultiObjectExportSettings,
    A1MultiObjectMode,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter import (  # noqa: E402
    A1MultiObjectSource,
    export_a1_multi_object,
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
)
from run_bake_integration import PNG_SIGNATURE, _assert  # noqa: E402
from run_grenade_bump_displacement_normal_uv_integration import (  # noqa: E402
    _require_loaded_blend,
)


_MINIMUM_MESH_OBJECTS = 2


def _parse_arguments() -> argparse.Namespace:
    arguments = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(
        description="Export current real grenade project for projected-axis Spine smoke."
    )
    parser.add_argument(
        "--expected-blend",
        required=True,
        help="Exact grenade.blend path Blender must already have loaded.",
    )
    parser.add_argument(
        "--output-directory",
        required=True,
        help="Persistent directory where production JSON and baked PNG files are kept.",
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
        f"grenade project contains too few Mesh objects: {len(objects)}",
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


def _settings_for_object(obj, scene_profile):
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


def _sources(objects: tuple, scene_profile) -> tuple[A1MultiObjectSource, ...]:
    return tuple(
        A1MultiObjectSource(
            source_object=obj,
            component_id=f"object_{index}:{obj.name_full}",
            animation_namespace=f"object_{index}",
            settings=_settings_for_object(obj, scene_profile),
        )
        for index, obj in enumerate(objects, start=1)
    )


def _assert_projected_axis_rig(prepared) -> tuple[str, int]:
    settings = prepared.settings
    _assert(
        settings.rig_setup_pose_mode is A1RigSetupPoseMode.PROJECTED_AXIS_NORMAL,
        "standalone signed-axis Normal/UV did not select PROJECTED_AXIS_NORMAL: "
        f"object={prepared.object_id!r}, actual={settings.rig_setup_pose_mode.value!r}",
    )
    _assert(
        settings.projection_direction.axis_aligned,
        f"projected-axis runner requires signed axis: {settings.projection_direction!r}",
    )

    transform = tuple(prepared.rig.transform)
    _assert(len(transform) >= 4, f"unexpected transform count for {prepared.object_id!r}")
    rotate_x, rotate_y, _scale, depth = transform[:4]
    _assert(
        float(rotate_x.extras.get("rotation", 0.0)) == 0.0,
        f"{prepared.object_id!r} X setup baseline is not neutral: {rotate_x.extras!r}",
    )
    _assert(
        float(rotate_y.extras.get("rotation", 0.0)) == 0.0,
        f"{prepared.object_id!r} Y setup baseline is not neutral: {rotate_y.extras!r}",
    )

    expected_minimum_depth = min(
        float(group.y_offset_pixels) for group in prepared.rig.info.z_groups
    )
    _assert(
        float(depth.extras.get("x")) == expected_minimum_depth,
        "projected-axis depth translation changed: "
        f"object={prepared.object_id!r}, expected={expected_minimum_depth}, "
        f"actual={depth.extras.get('x')!r}",
    )
    _assert(
        float(depth.extras.get("scaleX")) == -1.0,
        f"{prepared.object_id!r} projected-axis depth scale mapping changed: {depth.extras!r}",
    )

    rig_bone_names = {bone.name for bone in prepared.rig.bones}
    camera_setup_names = {
        prepared.rig.profile.z_camera_setup_bone(prepared.prefix, group.index)
        for group in prepared.rig.info.z_groups
    }
    _assert(
        rig_bone_names.isdisjoint(camera_setup_names),
        f"{prepared.object_id!r} unexpectedly contains camera setup bones",
    )

    ordinary_depth_names = {group.bone_name for group in prepared.rig.info.z_groups}
    vertex_bone_count = 0
    for component in prepared.document_assembly.document_build.components:
        for vertex_bone in component.vertex_bones:
            vertex_bone_count += 1
            _assert(
                vertex_bone.parent in ordinary_depth_names,
                "projected-axis vertex bone is not parented directly to ordinary depth: "
                f"object={prepared.object_id!r}, bone={vertex_bone.name!r}, "
                f"parent={vertex_bone.parent!r}",
            )
    _assert(vertex_bone_count > 0, f"{prepared.object_id!r} produced no vertex bones")
    return settings.projection_direction.value, vertex_bone_count


def _assert_outputs(result, expected_texture_count: int) -> tuple[Path, tuple[Path, ...]]:
    _assert(
        bool(result.success),
        "real projected-axis export failed: "
        f"issues={result.issues!r}, statistics={dict(result.statistics)!r}",
    )
    outputs = tuple(Path(path).resolve(strict=False) for path in result.output_files)
    json_files = tuple(path for path in outputs if path.suffix.lower() == ".json")
    png_files = tuple(path for path in outputs if path.suffix.lower() == ".png")
    _assert(len(json_files) == 1, f"expected exactly one production JSON: {json_files!r}")
    _assert(
        len(png_files) == expected_texture_count,
        "texture count differs from current Mesh object count: "
        f"expected={expected_texture_count}, actual={len(png_files)}",
    )
    for path in outputs:
        _assert(path.is_file() and path.stat().st_size > 8, f"missing/empty output: {path}")
    for path in png_files:
        _assert(path.read_bytes().startswith(PNG_SIGNATURE), f"output is not PNG: {path}")

    document = json.loads(json_files[0].read_text(encoding="utf-8"))
    _assert(isinstance(document, dict), "production JSON root must be mapping")
    _assert(bool(document.get("bones")), "production JSON contains no bones")
    _assert(bool(document.get("skins")), "production JSON contains no skins")
    return json_files[0], outputs


def _run(expected_blend: str, output_directory_arg: str) -> None:
    loaded = _require_loaded_blend(expected_blend)
    output_directory = Path(output_directory_arg).expanduser().resolve(strict=False)
    output_directory.mkdir(parents=True, exist_ok=True)

    completed = _register_steps()
    try:
        scene = bpy.context.scene
        objects = _mesh_objects(scene)
        before = _scene_fingerprint(scene, objects)
        datablocks_before = _datablock_fingerprint()

        scene_profile = _capture_scene_profile(
            scene,
            output_directory=output_directory,
            images_relative_path="images",
        )
        _assert(
            scene_profile.texture_export_mode is A1TextureExportMode.NORMAL_UV_SEGMENTS,
            "real projected-axis export requires persisted Normal / UV Segments; "
            f"actual={scene_profile.texture_export_mode.value!r}",
        )
        _assert(
            isinstance(scene_profile.projection_direction, A1ProjectionDirection)
            and scene_profile.projection_direction.axis_aligned,
            "real projected-axis export requires persisted signed-axis projection; "
            f"actual={scene_profile.projection_direction!r}",
        )

        sources = _sources(objects, scene_profile)
        multi_settings = A1MultiObjectExportSettings(
            output_directory=output_directory,
            output_stem="grenade_projected_axis_real",
            mode=A1MultiObjectMode.STANDALONE,
        )

        prepared = prepare_a1_multi_object(
            sources,
            multi_settings,
            context=bpy.context,
            scene=scene,
        )
        _assert(
            len(prepared.objects) == len(objects),
            "prepared object count differs from current Mesh object count",
        )
        rig_reports = tuple(
            (item.object_id, *_assert_projected_axis_rig(item))
            for item in prepared.objects
        )

        result = export_a1_multi_object(
            sources,
            multi_settings,
            context=bpy.context,
            scene=scene,
        )
        json_path, outputs = _assert_outputs(result, len(objects))

        _assert(
            _scene_fingerprint(scene, objects) == before,
            "real projected-axis export changed source object/scene/context state",
        )
        _assert(
            _datablock_fingerprint() == datablocks_before,
            "real projected-axis export leaked or removed Blender datablocks",
        )

        print(
            "[GRENADE-PROJECTED-AXIS-REAL-EXPORT] PASS "
            f"blend={loaded} mesh_objects={len(objects)} "
            f"projection={scene_profile.projection_direction.value!r} "
            f"setup={A1RigSetupPoseMode.PROJECTED_AXIS_NORMAL.value!r} "
            f"rig_reports={rig_reports!r} outputs={len(outputs)} "
            f"json={str(json_path)!r} source=unchanged",
            flush=True,
        )
    finally:
        _unregister_steps(completed)


def main() -> None:
    arguments = _parse_arguments()
    _run(arguments.expected_blend, arguments.output_directory)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
