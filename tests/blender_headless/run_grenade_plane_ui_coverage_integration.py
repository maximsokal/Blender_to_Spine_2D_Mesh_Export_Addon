"""Reproduce the installed-UI Normal/UV coverage path on real grenade.blend.

The manual 0.129.0 smoke exposed a staging-only coverage failure for
``Plane_Segment_56``. This runner deliberately does not hard-code a texture size: the
artist-authored ``grenade.blend`` owns its persisted Scene setting, and the production
Scene/Object UI profile builders must carry that exact value into immutable export
settings.

The production multi-object contract requires at least two sources, so the focused
regression exports ``Plane`` together with ``Cube``. ``Plane`` remains the target that
must contain the historical ``Plane_Segment_56`` triangles, while ``Cube`` is the
smallest deterministic companion already covered by the grenade Bump regression.
Cycles samples are reduced to one only to keep this gate fast; geometry, UV layout,
texture dimensions, bake margin, staging and coverage validation remain on the
production UI path.

The historical triangles 44/45 are additionally proven to be finite but to contain no
output texel sample centre at the persisted resolution. Successful staging must report
those triangles through the resolution-unrepresentable coverage statistic instead of
silently disabling validation.
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

import Blender_to_Spine2D_Mesh_Exporter as addon  # noqa: E402
from Blender_to_Spine2D_Mesh_Exporter.application import (  # noqa: E402
    A1MultiObjectExportSettings,
    A1MultiObjectMode,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter import (  # noqa: E402
    A1MultiObjectSource,
    export_a1_multi_object,
    prepare_a1_object,
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
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.bake_uv_raster_coverage import (  # noqa: E402
    raster_sample_pixels,
    triangle_twice_area_pixels,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import (  # noqa: E402
    A1TextureExportMode,
)
from run_bake_integration import PNG_SIGNATURE, _assert  # noqa: E402
from run_grenade_bump_displacement_normal_uv_integration import (  # noqa: E402
    _require_loaded_blend,
)


_EXPECTED_OBJECT_NAME = "Plane"
_COMPANION_OBJECT_NAME = "Cube"
_EXPECTED_ATTACHMENT_NAME = "Plane_Segment_56"
_EXPECTED_PROBLEM_TRIANGLE_INDICES = (44, 45)
_EXPECTED_OUTPUT_PNG_COUNT = 2


def _parse_arguments() -> argparse.Namespace:
    arguments = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(
        description="Run real grenade Plane UI-path bake coverage regression."
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


def _require_mesh_object(name: str):
    source = bpy.data.objects.get(name)
    _assert(source is not None, f"missing grenade object: {name!r}")
    _assert(source.type == "MESH", f"grenade object {name!r} must be a Mesh")
    _assert(getattr(source, "data", None) is not None, f"{name!r} has no Mesh data")
    return source


def _source_fingerprint(source, scene) -> tuple:
    return (
        source.name_full,
        source.data.name_full,
        tuple(tuple(float(value) for value in row) for row in source.matrix_world),
        tuple(
            None if slot.material is None else slot.material.name_full
            for slot in source.material_slots
        ),
        tuple(layer.name for layer in source.data.uv_layers),
        (
            None
            if source.data.uv_layers.active is None
            else source.data.uv_layers.active.name
        ),
        int(scene.frame_current),
        str(scene.render.engine),
        None if scene.camera is None else scene.camera.name_full,
    )


def _datablock_fingerprint() -> tuple:
    return (
        tuple(sorted(item.name_full for item in bpy.data.objects)),
        tuple(sorted(item.name_full for item in bpy.data.meshes)),
        tuple(sorted(item.name_full for item in bpy.data.materials)),
        tuple(sorted(item.name_full for item in bpy.data.images)),
    )


def _persisted_texture_size(scene) -> int:
    raw = getattr(scene, "spine2d_texture_size", None)
    _assert(raw is not None, "grenade scene has no persisted spine2d_texture_size")
    _assert(not isinstance(raw, bool), "spine2d_texture_size cannot be bool")
    try:
        value = int(raw)
    except (TypeError, ValueError, OverflowError) as exc:
        raise AssertionError(
            f"invalid persisted spine2d_texture_size: {raw!r}"
        ) from exc
    _assert(
        64 <= value <= 4096 and value % 2 == 0,
        f"persisted spine2d_texture_size is outside the public UI contract: {value}",
    )
    return value


def _capture_ui_scene_profile(scene, output_directory: Path):
    persisted_texture_size = _persisted_texture_size(scene)
    profile = _capture_scene_profile(
        scene,
        output_directory=output_directory,
        images_relative_path="images",
    )
    _assert(
        profile.texture_export_mode is A1TextureExportMode.NORMAL_UV_SEGMENTS,
        "grenade UI regression must remain Normal / UV Segments; "
        f"actual={profile.texture_export_mode.value}",
    )
    _assert(
        profile.texture_size == persisted_texture_size,
        "Scene profile changed the persisted UI texture size; "
        f"scene={persisted_texture_size}, profile={profile.texture_size}",
    )
    return profile, persisted_texture_size


def _settings_from_ui_profile(source, scene_profile, persisted_texture_size: int):
    bake = getattr(source, "spine2d_bake_settings", None)
    object_profile = _capture_object_profile(
        source,
        sequence_start_frame=int(getattr(bake, "bake_frame_start", 0)),
        sequence_frame_count=int(getattr(bake, "frames_for_render", 0)),
        connect_enabled=_connect_enabled(source),
    )
    settings = _settings_from_profiles(object_profile, scene_profile)
    _assert(
        settings.export.texture_width == persisted_texture_size
        and settings.export.texture_height == persisted_texture_size,
        "UI settings changed the captured Scene texture size: "
        f"object={source.name_full!r}, scene={persisted_texture_size}, "
        f"export=({settings.export.texture_width}, {settings.export.texture_height})",
    )
    _assert(
        settings.export.bake_margin >= 0,
        f"UI bake margin must be non-negative: {settings.export.bake_margin}",
    )
    _assert(
        settings.bake_execution.samples >= 1,
        "captured UI bake samples must be positive",
    )

    # Sampling count affects shader convergence, not generated UV geometry or the alpha
    # raster footprint of this opaque object bake. Keep every other captured UI setting.
    return replace(
        settings,
        bake_execution=replace(settings.bake_execution, samples=1),
        json_output_stem=None,
    )


def _triangle_uvs(request, triangle_index: int) -> tuple[tuple[float, float], ...]:
    offset = triangle_index * 3
    indices = tuple(request.triangles[offset : offset + 3])
    _assert(
        len(indices) == 3,
        f"attachment triangle {triangle_index} is incomplete: {indices!r}",
    )
    return tuple(
        tuple(float(value) for value in request.vertices[index].uv)
        for index in indices
    )


def _assert_problem_attachment(
    prepared,
    texture_size: int,
) -> tuple[int, tuple[float, ...]]:
    matching = tuple(
        projection
        for projection in prepared.document_assembly.projections
        if projection.request.attachment_name == _EXPECTED_ATTACHMENT_NAME
    )
    _assert(
        len(matching) == 1,
        "real grenade UI regression no longer contains the attachment that exposed "
        f"the staging failure: expected={_EXPECTED_ATTACHMENT_NAME!r}, "
        f"available={tuple(p.request.attachment_name for p in prepared.document_assembly.projections)!r}",
    )
    request = matching[0].request
    triangle_count = len(request.triangles) // 3
    _assert(
        len(request.triangles) % 3 == 0,
        f"{_EXPECTED_ATTACHMENT_NAME} has malformed triangle stream",
    )
    _assert(
        triangle_count > max(_EXPECTED_PROBLEM_TRIANGLE_INDICES),
        f"{_EXPECTED_ATTACHMENT_NAME} no longer contains historical triangles 44/45",
    )

    areas: list[float] = []
    for triangle_index in _EXPECTED_PROBLEM_TRIANGLE_INDICES:
        uvs = _triangle_uvs(request, triangle_index)
        area = triangle_twice_area_pixels(
            uvs,
            width=texture_size,
            height=texture_size,
        )
        samples = raster_sample_pixels(
            uvs,
            width=texture_size,
            height=texture_size,
        )
        _assert(
            area > 0.0,
            f"historical triangle {triangle_index} became UV-degenerate",
        )
        _assert(
            samples == (),
            "historical regression triangle unexpectedly gained a texel sample centre; "
            f"triangle={triangle_index}, texture={texture_size}, "
            f"area_twice_pixels={area}, samples={samples!r}, uvs={uvs!r}",
        )
        areas.append(area)

    return triangle_count, tuple(areas)


def _multi_source(source, settings, index: int) -> A1MultiObjectSource:
    return A1MultiObjectSource(
        source_object=source,
        component_id=f"object_{index}:{source.name_full}",
        animation_namespace=f"object_{index}",
        settings=settings,
    )


def _run(expected_blend: str) -> None:
    loaded = _require_loaded_blend(expected_blend)
    completed = _register_steps()
    try:
        scene = bpy.context.scene
        target = _require_mesh_object(_EXPECTED_OBJECT_NAME)
        companion = _require_mesh_object(_COMPANION_OBJECT_NAME)

        target_before = _source_fingerprint(target, scene)
        companion_before = _source_fingerprint(companion, scene)
        datablocks_before = _datablock_fingerprint()

        with tempfile.TemporaryDirectory(prefix="spine2d_grenade_plane_ui_coverage_") as root:
            output_directory = Path(root).resolve(strict=False)
            scene_profile, texture_size = _capture_ui_scene_profile(
                scene,
                output_directory,
            )
            target_settings = _settings_from_ui_profile(
                target,
                scene_profile,
                texture_size,
            )
            companion_settings = _settings_from_ui_profile(
                companion,
                scene_profile,
                texture_size,
            )
            _assert(
                target_settings.export.bake_margin == companion_settings.export.bake_margin,
                "UI profile produced different bake margins for the two sources",
            )

            # Prepare the exact target first so fixture drift and raster assumptions are
            # proven before the broader two-object transaction starts.
            prepared = prepare_a1_object(
                target,
                target_settings,
                context=bpy.context,
                scene=scene,
            )
            triangle_count, problem_areas = _assert_problem_attachment(
                prepared,
                texture_size,
            )

            sources = (
                _multi_source(target, target_settings, 1),
                _multi_source(companion, companion_settings, 2),
            )
            multi_settings = A1MultiObjectExportSettings(
                output_directory=output_directory,
                output_stem="Grenade_Plane_UI_Coverage",
                mode=A1MultiObjectMode.STANDALONE,
            )
            result = export_a1_multi_object(
                sources,
                multi_settings,
                context=bpy.context,
                scene=scene,
            )
            _assert(
                bool(result.success),
                "real grenade Plane UI export failed: "
                f"issues={result.issues!r}, statistics={dict(result.statistics)!r}",
            )

            result_statistics = dict(result.statistics)
            subpixel_key = (
                "component.object_1:Plane."
                "bake_uv_resolution_unrepresentable_triangle_count"
            )
            _assert(
                subpixel_key in result_statistics,
                "successful UI export did not expose subpixel coverage statistics; "
                f"available={tuple(sorted(result_statistics))!r}",
            )
            subpixel_count = int(result_statistics[subpixel_key])
            _assert(
                subpixel_count >= len(_EXPECTED_PROBLEM_TRIANGLE_INDICES),
                "successful UI export did not classify the historical subpixel "
                f"triangles; expected_at_least={len(_EXPECTED_PROBLEM_TRIANGLE_INDICES)}, "
                f"actual={subpixel_count}",
            )

            outputs = tuple(Path(path).resolve(strict=False) for path in result.output_files)
            json_files = tuple(path for path in outputs if path.suffix.lower() == ".json")
            png_files = tuple(path for path in outputs if path.suffix.lower() == ".png")
            _assert(len(json_files) == 1, f"expected one JSON output: {json_files!r}")
            _assert(
                len(png_files) == _EXPECTED_OUTPUT_PNG_COUNT,
                "two-source UI regression produced unexpected PNG count: "
                f"expected={_EXPECTED_OUTPUT_PNG_COUNT}, actual={len(png_files)}, "
                f"files={png_files!r}",
            )
            _assert(
                all(path.read_bytes().startswith(PNG_SIGNATURE) for path in png_files),
                f"UI coverage outputs contain a non-PNG file: {png_files!r}",
            )
            parsed = json.loads(json_files[0].read_text(encoding="utf-8"))
            _assert(isinstance(parsed, dict) and bool(parsed.get("skins")), "invalid JSON")

        _assert(
            _source_fingerprint(target, scene) == target_before,
            "Plane UI export changed target source/scene state",
        )
        _assert(
            _source_fingerprint(companion, scene) == companion_before,
            "Plane UI export changed companion source/scene state",
        )
        _assert(
            _datablock_fingerprint() == datablocks_before,
            "Plane UI export leaked or removed Blender datablocks",
        )

        print(
            "[GRENADE-PLANE-UI-COVERAGE] PASS "
            f"blend={loaded} object={target.name_full!r} "
            f"companion={companion.name_full!r} "
            f"attachment={_EXPECTED_ATTACHMENT_NAME!r} "
            f"triangle_count={triangle_count} "
            f"problem_twice_area_pixels={problem_areas!r} "
            f"subpixel_triangles={subpixel_count} "
            f"texture={texture_size} margin={target_settings.export.bake_margin} "
            "staging=passed source=unchanged",
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
