"""Reproduce the installed-UI 1024 Normal/UV coverage path on real grenade.blend.

The manual 0.129.0 smoke exposed a staging-only false negative for
``Plane_Segment_56`` at the public UI's 1024 texture size.  Earlier grenade runners used
256px test settings, where the same tiny UV triangles were hidden by the much larger
relative bake margin.  This runner captures settings through the production Scene/Object
UI profile builders, verifies the public 1024 configuration, then reduces Cycles samples
to one only to keep the regression fast.  Geometry, UV layout, texture dimensions,
bake margin, staging and coverage validation remain on the production UI path.
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
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import (  # noqa: E402
    A1TextureExportMode,
)
from run_bake_integration import PNG_SIGNATURE, _assert  # noqa: E402
from run_grenade_bump_displacement_normal_uv_integration import (  # noqa: E402
    _require_loaded_blend,
)


_EXPECTED_OBJECT_NAME = "Plane"
_EXPECTED_ATTACHMENT_NAME = "Plane_Segment_56"
_EXPECTED_TEXTURE_SIZE = 1024
_EXPECTED_BAKE_MARGIN = 4


def _parse_arguments() -> argparse.Namespace:
    arguments = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(
        description="Run real grenade Plane 1024 UI-path bake coverage regression."
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


def _settings_from_ui(source, scene, output_directory: Path):
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
        profile.texture_size == _EXPECTED_TEXTURE_SIZE,
        "grenade fixture/UI texture size changed; "
        f"expected={_EXPECTED_TEXTURE_SIZE}, actual={profile.texture_size}",
    )

    bake = getattr(source, "spine2d_bake_settings", None)
    object_profile = _capture_object_profile(
        source,
        sequence_start_frame=int(getattr(bake, "bake_frame_start", 0)),
        sequence_frame_count=int(getattr(bake, "frames_for_render", 0)),
        connect_enabled=_connect_enabled(source),
    )
    settings = _settings_from_profiles(object_profile, profile)
    _assert(
        settings.export.texture_width == _EXPECTED_TEXTURE_SIZE
        and settings.export.texture_height == _EXPECTED_TEXTURE_SIZE,
        f"UI settings lost 1024 texture size: {settings.export!r}",
    )
    _assert(
        settings.export.bake_margin == _EXPECTED_BAKE_MARGIN,
        "UI bake margin changed; "
        f"expected={_EXPECTED_BAKE_MARGIN}, actual={settings.export.bake_margin}",
    )
    _assert(
        settings.bake_execution.samples >= 1,
        "captured UI bake samples must be positive",
    )

    # Sampling count affects shader convergence, not generated UV geometry or the alpha
    # raster footprint of an opaque object bake. Keep every other captured UI setting.
    return replace(
        settings,
        bake_execution=replace(settings.bake_execution, samples=1),
        prefix=_EXPECTED_OBJECT_NAME,
        output_stem="Grenade_Plane_UI_1024",
        json_output_stem=None,
    )


def _assert_problem_attachment(prepared) -> tuple[int, ...]:
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
    triangles = tuple(matching[0].request.triangles)
    _assert(
        len(triangles) >= 3 * 46,
        f"{_EXPECTED_ATTACHMENT_NAME} no longer contains historical triangles 44/45",
    )
    return triangles


def _run(expected_blend: str) -> None:
    loaded = _require_loaded_blend(expected_blend)
    completed = _register_steps()
    try:
        scene = bpy.context.scene
        source = bpy.data.objects.get(_EXPECTED_OBJECT_NAME)
        _assert(source is not None, f"missing grenade object: {_EXPECTED_OBJECT_NAME!r}")
        _assert(source.type == "MESH", f"{_EXPECTED_OBJECT_NAME} must be a Mesh")

        before = _source_fingerprint(source, scene)
        datablocks_before = _datablock_fingerprint()

        with tempfile.TemporaryDirectory(prefix="spine2d_grenade_plane_ui_1024_") as root:
            output_directory = Path(root).resolve(strict=False)
            settings = _settings_from_ui(source, scene, output_directory)
            prepared = prepare_a1_object(
                source,
                settings,
                context=bpy.context,
                scene=scene,
            )
            triangles = _assert_problem_attachment(prepared)

            multi_source = A1MultiObjectSource(
                source_object=source,
                component_id="object_1:Plane",
                animation_namespace="object_1",
                settings=settings,
            )
            multi_settings = A1MultiObjectExportSettings(
                output_directory=output_directory,
                output_stem="Grenade_Plane_UI_1024",
                mode=A1MultiObjectMode.STANDALONE,
            )
            result = export_a1_multi_object(
                (multi_source,),
                multi_settings,
                context=bpy.context,
                scene=scene,
            )
            _assert(
                bool(result.success),
                "real grenade Plane UI 1024 export failed: "
                f"issues={result.issues!r}, statistics={dict(result.statistics)!r}",
            )
            outputs = tuple(Path(path).resolve(strict=False) for path in result.output_files)
            json_files = tuple(path for path in outputs if path.suffix.lower() == ".json")
            png_files = tuple(path for path in outputs if path.suffix.lower() == ".png")
            _assert(len(json_files) == 1, f"expected one JSON output: {json_files!r}")
            _assert(len(png_files) == 1, f"expected one PNG output: {png_files!r}")
            _assert(
                png_files[0].read_bytes().startswith(PNG_SIGNATURE),
                f"Plane UI output is not PNG: {png_files[0]}",
            )
            parsed = json.loads(json_files[0].read_text(encoding="utf-8"))
            _assert(isinstance(parsed, dict) and bool(parsed.get("skins")), "invalid JSON")

        _assert(
            _source_fingerprint(source, scene) == before,
            "Plane UI 1024 export changed source/scene state",
        )
        _assert(
            _datablock_fingerprint() == datablocks_before,
            "Plane UI 1024 export leaked or removed Blender datablocks",
        )

        print(
            "[GRENADE-PLANE-UI-1024-COVERAGE] PASS "
            f"blend={loaded} object={source.name_full!r} "
            f"attachment={_EXPECTED_ATTACHMENT_NAME!r} "
            f"triangle_count={len(triangles) // 3} "
            f"texture={_EXPECTED_TEXTURE_SIZE} margin={_EXPECTED_BAKE_MARGIN} "
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
