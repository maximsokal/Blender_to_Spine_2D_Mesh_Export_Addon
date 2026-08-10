"""Serialize the real grenade standalone document without rebaking textures.

This runner is a visual-smoke bridge between the fast setup-pose preparation regression
and the expensive full Blender export. It uses the persisted Scene output directory and
requires every texture file predicted by the current Normal / UV settings to already exist.
No texture task is executed. The prepared standalone document is composed and serialized
through the production Spine codec, then committed atomically as one additional JSON file.

The runner never changes the source .blend, object transforms, UV layers, materials, or
selection state. Existing PNG files are read only by existence checks; they are never
opened or modified.
"""

from __future__ import annotations

import argparse
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
    prepare_a1_multi_object,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.a1_multi_object_composition import (  # noqa: E402
    compose_a1_multi_object_document,
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
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.version_codecs import (  # noqa: E402
    serialize_spine_document,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.version_target import (  # noqa: E402
    spine_json_version_filename_token,
)
from Blender_to_Spine2D_Mesh_Exporter.infrastructure import (  # noqa: E402
    atomic_file_transaction,
    write_staged_utf8_text,
)
from run_bake_integration import _assert  # noqa: E402
from run_grenade_bump_displacement_normal_uv_integration import (  # noqa: E402
    _require_loaded_blend,
)
from run_grenade_standalone_setup_pose_integration import (  # noqa: E402
    _assert_neutral_projected_object,
    _datablock_fingerprint,
    _mesh_objects,
    _scene_fingerprint,
)


_OUTPUT_PREFIX = "grenade_setup_pose_smoke"


def _parse_arguments() -> argparse.Namespace:
    arguments = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(
        description="Serialize grenade standalone Spine JSON using existing PNG files."
    )
    parser.add_argument(
        "--expected-blend",
        required=True,
        help="Exact grenade.blend path Blender must already have loaded.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help=(
            "Optional explicit JSON path. By default the runner writes a versioned "
            "smoke JSON into the persisted Scene export directory."
        ),
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


def _expected_texture_paths(prepared) -> tuple[Path, ...]:
    paths: list[Path] = []
    for item in prepared.objects:
        plans = (
            item.bake_plan,
            *tuple(getattr(item, "reserve_bake_plans", ())),
        )
        for plan in plans:
            for task in plan.frame_tasks:
                paths.append(task.output_path.expanduser().resolve(strict=False))
    return tuple(paths)


def _require_existing_textures(paths: tuple[Path, ...]) -> None:
    _assert(paths, "prepared standalone document predicts no texture outputs")
    missing = tuple(path for path in paths if not path.is_file())
    _assert(
        not missing,
        "JSON-only smoke requires textures from a previous export. Missing files:\n"
        + "\n".join(str(path) for path in missing),
    )


def _resolve_output_json(scene_profile, explicit_path: str | None) -> Path:
    if explicit_path is not None:
        value = explicit_path.strip()
        _assert(value, "--output-json cannot be empty")
        path = Path(value).expanduser().resolve(strict=False)
    else:
        token = spine_json_version_filename_token(scene_profile.spine_target)
        path = (
            scene_profile.output_directory
            / f"{_OUTPUT_PREFIX}_{token}.json"
        ).expanduser().resolve(strict=False)
    _assert(path.suffix.casefold() == ".json", "output JSON must use .json extension")
    return path


def _write_json_atomically(output_json: Path, json_text: str) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with atomic_file_transaction(
        operation_name="grenade-standalone-json-only-smoke"
    ) as transaction:
        reservation = transaction.reserve(output_json)
        write_staged_utf8_text(
            reservation.staged_path,
            json_text,
            ensure_trailing_newline=True,
        )
        committed = transaction.commit()
    _assert(
        tuple(committed) == (reservation.final_path,),
        "JSON-only smoke committed an unexpected output set",
    )


def _run(expected_blend: str, explicit_output_json: str | None) -> None:
    loaded = _require_loaded_blend(expected_blend)
    completed = _register_steps()
    try:
        scene = bpy.context.scene
        objects = _mesh_objects(scene)
        before = _scene_fingerprint(scene, objects)
        datablocks_before = _datablock_fingerprint()

        scene_profile = _capture_scene_profile(scene)
        _assert(
            scene_profile.texture_export_mode is A1TextureExportMode.NORMAL_UV_SEGMENTS,
            "JSON-only grenade smoke requires persisted Normal / UV Segments mode; "
            f"actual={scene_profile.texture_export_mode.value}",
        )
        _assert(
            isinstance(scene_profile.projection_direction, A1ProjectionDirection)
            and scene_profile.projection_direction.axis_aligned,
            "JSON-only grenade smoke requires a persisted signed-axis projection; "
            f"actual={scene_profile.projection_direction!r}",
        )

        sources = tuple(
            _source(obj, _object_settings(obj, scene_profile), index)
            for index, obj in enumerate(objects, start=1)
        )
        multi_settings = A1MultiObjectExportSettings(
            output_directory=scene_profile.output_directory,
            output_stem=_OUTPUT_PREFIX,
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
            "prepared object count differs from source Mesh object count",
        )

        placements = tuple(
            (item.object_id, *_assert_neutral_projected_object(item))
            for item in prepared.objects
        )
        texture_paths = _expected_texture_paths(prepared)
        _require_existing_textures(texture_paths)

        composition = compose_a1_multi_object_document(
            prepared.sources,
            prepared.objects,
            multi_settings,
        )
        spine_target = prepared.objects[0].settings.export.spine_target
        _assert(
            all(item.settings.export.spine_target == spine_target for item in prepared.objects),
            "prepared objects do not share one Spine target",
        )
        json_text = serialize_spine_document(
            composition.document,
            spine_target,
            indent=2,
        )
        output_json = _resolve_output_json(scene_profile, explicit_output_json)
        _write_json_atomically(output_json, json_text)

        _assert(output_json.is_file(), f"smoke JSON was not created: {output_json}")
        _assert(output_json.stat().st_size > 0, "smoke JSON is empty")
        _assert(
            _scene_fingerprint(scene, objects) == before,
            "JSON-only smoke changed source object/scene/context state",
        )
        _assert(
            _datablock_fingerprint() == datablocks_before,
            "JSON-only smoke leaked or removed Blender datablocks",
        )

        print(
            "[GRENADE-STANDALONE-JSON-ONLY] PASS "
            f"blend={loaded} mesh_objects={len(objects)} "
            f"projection={scene_profile.projection_direction.value!r} "
            f"spine={spine_target.exact_version!r} "
            f"textures={len(texture_paths)} "
            f"json={str(output_json)!r} bytes={output_json.stat().st_size} "
            f"placements={placements!r} source=unchanged",
            flush=True,
        )
    finally:
        _unregister_steps(completed)


def main() -> None:
    arguments = _parse_arguments()
    _run(arguments.expected_blend, arguments.output_json)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
