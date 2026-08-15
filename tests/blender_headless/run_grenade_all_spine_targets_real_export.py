"""Real grenade.blend export matrix for every production-ready Spine JSON target.

The target list is derived from the production codec registry. Every target is exported
with a deliberately non-default exact project patch version from AddonPreferences. This
proves that codec family selection and exact JSON metadata are independent while the same
artist-authored multi-object selection, Shared Pivot contract, and production exporter are
used for every registered target.
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

from Blender_to_Spine2D_Mesh_Exporter.application import resolve_a1_names  # noqa: E402
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.a1_shared_pivot import (  # noqa: E402
    resolve_a1_shared_pivot_world,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.a1_ui_export_plan import (  # noqa: E402
    build_selected_ui_export_plan,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.a1_ui_router import (  # noqa: E402
    export_selected_objects_a1,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.spine_version_preferences import (  # noqa: E402
    SPINE_EXACT_VERSION_PREFERENCE_SPECS,
    assign_spine_project_exact_version,
    get_spine_addon_preferences,
    spine_exact_version_preference_spec,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import (  # noqa: E402
    A1TextureExportMode,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.projection import (  # noqa: E402
    A1ProjectionDirection,
    resolve_a1_axis_projection_basis,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine import calculate_uniform_scale  # noqa: E402
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.version_codecs.registry import (  # noqa: E402
    registered_spine_json_codecs,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.version_target import (  # noqa: E402
    validate_spine_json_exact_version_for_target,
)
from run_grenade_shared_pivot_real_export import (  # noqa: E402
    _assert,
    _assert_production_outputs,
    _canonical_legacy_main_position,
    _datablock_fingerprint,
    _register_steps,
    _require_loaded_blend,
    _scene_fingerprint,
    _selected_meshes,
    _unregister_steps,
)


_SCENE_TARGET_PROPERTY = "spine2d_target_spine_version"


def _parse_arguments() -> argparse.Namespace:
    arguments = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expected-blend",
        required=True,
        help="Exact grenade.blend path Blender must already have loaded.",
    )
    parser.add_argument(
        "--output-directory",
        required=True,
        help="Empty root directory that will receive one subdirectory per Spine target.",
    )
    return parser.parse_args(arguments)


def _prepare_output_root(value: str) -> Path:
    output = Path(value).expanduser().resolve(strict=False)
    if output.exists() and not output.is_dir():
        raise ValueError(f"Output path is not a directory: {output}")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output root must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def _target_output_directory(root: Path, target) -> Path:
    family_slug = target.family.replace(".", "_")
    output = root / f"spine_{family_slug}"
    output.mkdir(parents=False, exist_ok=False)
    return output


def _custom_exact_version(target) -> str:
    """Choose a deterministic same-family patch different from the registry default."""

    major_text, minor_text, patch_text = target.exact_version.split(".")
    patch = int(patch_text)
    custom_patch = patch - 1 if patch > 0 else patch + 1
    value = f"{int(major_text)}.{int(minor_text)}.{custom_patch}"
    normalized = validate_spine_json_exact_version_for_target(target, value)
    _assert(
        normalized != target.exact_version,
        f"custom exact version must differ from default for {target.value}",
    )
    return normalized


def _assert_target_plan(plan, target, expected_exact_version: str) -> None:
    _assert(
        plan.settings.shared_pivot_enabled,
        f"Shared Pivot was not enabled for target {expected_exact_version}",
    )
    _assert(plan.standalone_sources, "selected-object UI plan contains no sources")

    for source_index, source in enumerate(plan.standalone_sources):
        export_settings = source.settings.export
        _assert(
            export_settings.spine_target is target,
            "public UI plan resolved the wrong Spine target: "
            f"index={source_index}, expected={target.value}, "
            f"actual={export_settings.spine_target.value}",
        )
        _assert(
            export_settings.spine_version == expected_exact_version,
            "public UI plan resolved the wrong exact Spine project version: "
            f"index={source_index}, expected={expected_exact_version!r}, "
            f"actual={export_settings.spine_version!r}",
        )

    first_settings = plan.standalone_sources[0].settings
    _assert(
        first_settings.bake_execution.texture_export_mode
        is A1TextureExportMode.NORMAL_UV_SEGMENTS,
        "grenade target matrix requires persisted Normal / UV Segments; "
        f"actual={first_settings.bake_execution.texture_export_mode.value!r}",
    )
    direction = first_settings.projection_direction
    _assert(
        isinstance(direction, A1ProjectionDirection) and direction.axis_aligned,
        "grenade target matrix requires one signed axis; "
        f"actual={direction!r}",
    )


def _assert_serialized_exact_version(
    json_path: Path,
    target,
    expected_exact_version: str,
) -> None:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    _assert(isinstance(payload, dict), "production JSON root must be mapping")
    skeleton = payload.get("skeleton")
    _assert(
        isinstance(skeleton, dict),
        f"target {expected_exact_version} JSON contains no skeleton object",
    )
    actual = skeleton.get("spine")
    _assert(
        actual == expected_exact_version,
        "serialized skeleton.spine does not match configured project version: "
        f"target={target.value}, expected={expected_exact_version!r}, actual={actual!r}",
    )
    _assert(
        expected_exact_version in json_path.name,
        "production JSON filename does not contain configured exact project version: "
        f"expected={expected_exact_version!r}, json={json_path.name!r}",
    )


def _assert_source_state_unchanged(
    scene,
    selected,
    before,
    datablocks_before,
    *,
    label: str,
) -> None:
    _assert(
        _scene_fingerprint(scene, selected) == before,
        f"{label} changed source object/scene/context state",
    )
    _assert(
        _datablock_fingerprint() == datablocks_before,
        f"{label} leaked or removed Blender datablocks",
    )


def _run(expected_blend: str, output_directory_arg: str) -> None:
    loaded = _require_loaded_blend(expected_blend)
    output_root = _prepare_output_root(output_directory_arg)

    codecs = registered_spine_json_codecs()
    _assert(codecs, "production Spine JSON codec registry is empty")

    completed = _register_steps()
    try:
        scene = bpy.context.scene
        selected = _selected_meshes(bpy.context)
        preferences = get_spine_addon_preferences(required=True)

        _assert(
            bool(getattr(scene, "spine2d_shared_selection_pivot", False)),
            "Shared Selection Pivot RNA default must be enabled",
        )
        _assert(
            hasattr(scene, _SCENE_TARGET_PROPERTY),
            f"Spine JSON target Scene property is not registered: {_SCENE_TARGET_PROPERTY}",
        )

        original_output_path = str(getattr(scene, "spine2d_json_path", ""))
        original_shared_pivot = bool(
            getattr(scene, "spine2d_shared_selection_pivot", True)
        )
        original_target = str(getattr(scene, _SCENE_TARGET_PROPERTY))
        original_exact_versions = {
            spec.target: str(getattr(preferences, spec.property_name))
            for spec in SPINE_EXACT_VERSION_PREFERENCE_SPECS
        }

        try:
            for target, codec in codecs.items():
                _assert(
                    codec.target is target,
                    "codec registry key/codec target mismatch reached Blender matrix gate",
                )
                expected_exact_version = _custom_exact_version(target)
                assigned = assign_spine_project_exact_version(
                    preferences,
                    target,
                    expected_exact_version,
                )
                _assert(
                    assigned == expected_exact_version,
                    "AddonPreferences exact-version assignment changed the canonical value",
                )
                spec = spine_exact_version_preference_spec(target)
                _assert(
                    str(getattr(preferences, spec.property_name)) == expected_exact_version,
                    "AddonPreferences rejected the configured exact project version",
                )

                target_output = _target_output_directory(output_root, target)
                scene.spine2d_json_path = str(target_output)
                scene.spine2d_shared_selection_pivot = True
                setattr(scene, _SCENE_TARGET_PROPERTY, target.value)
                actual_scene_target = str(getattr(scene, _SCENE_TARGET_PROPERTY))
                _assert(
                    actual_scene_target == target.value,
                    "Blender Scene rejected a registered Spine JSON target: "
                    f"expected={target.value!r}, actual={actual_scene_target!r}",
                )

                before = _scene_fingerprint(scene, selected)
                datablocks_before = _datablock_fingerprint()

                plan = build_selected_ui_export_plan(bpy.context)
                _assert_target_plan(plan, target, expected_exact_version)
                _assert(
                    len(plan.standalone_sources) == len(selected),
                    "UI plan source count differs from selected Mesh object count: "
                    f"target={expected_exact_version}, selected={len(selected)}, "
                    f"sources={len(plan.standalone_sources)}",
                )

                first_settings = plan.standalone_sources[0].settings
                direction = first_settings.projection_direction
                resolution = resolve_a1_shared_pivot_world(
                    plan.standalone_sources,
                    scene=scene,
                )
                basis = resolve_a1_axis_projection_basis(direction)
                projected_pivot = basis.project_point(resolution.pivot_world)
                uniform_scale = calculate_uniform_scale(
                    first_settings.export.texture_width,
                    first_settings.export.texture_height,
                    first_settings.rig_scale_mode,
                )
                raw_expected_main_position = (
                    float(projected_pivot.u) * uniform_scale,
                    float(projected_pivot.v) * uniform_scale,
                )
                expected_main_position = _canonical_legacy_main_position(
                    raw_expected_main_position
                )
                prefixes = tuple(
                    resolve_a1_names(
                        str(source.source_object.name_full),
                        source.settings,
                    )[0]
                    for source in plan.standalone_sources
                )
                _assert(
                    len(prefixes) == len(set(prefixes)),
                    f"selected objects produced duplicate rig prefixes: {prefixes!r}",
                )

                result = export_selected_objects_a1(bpy.context)
                json_path, outputs = _assert_production_outputs(
                    result,
                    selected_count=len(selected),
                    prefixes=prefixes,
                    expected_main_position=expected_main_position,
                )
                _assert_serialized_exact_version(
                    json_path,
                    target,
                    expected_exact_version,
                )
                _assert(
                    str(getattr(scene, _SCENE_TARGET_PROPERTY)) == target.value,
                    "production export changed the persisted Spine target during the transaction",
                )
                _assert(
                    str(getattr(preferences, spec.property_name)) == expected_exact_version,
                    "production export changed the exact project version preference",
                )
                _assert_source_state_unchanged(
                    scene,
                    selected,
                    before,
                    datablocks_before,
                    label=f"Spine {expected_exact_version} export",
                )

                print(
                    "[GRENADE-ALL-SPINE-TARGETS] TARGET PASS "
                    f"target={target.value} exact={expected_exact_version} "
                    f"default_exact={target.exact_version} "
                    f"selected_meshes={len(selected)} outputs={len(outputs)} "
                    f"json={str(json_path)!r} source=unchanged",
                    flush=True,
                )
        finally:
            scene.spine2d_json_path = original_output_path
            scene.spine2d_shared_selection_pivot = original_shared_pivot
            setattr(scene, _SCENE_TARGET_PROPERTY, original_target)
            for target, original_value in original_exact_versions.items():
                assign_spine_project_exact_version(
                    preferences,
                    target,
                    original_value,
                )

        print(
            "[GRENADE-ALL-SPINE-TARGETS] PASS "
            f"blend={loaded} targets={len(codecs)} selected_meshes={len(selected)} "
            f"output_root={str(output_root)!r}",
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
