"""Validate fail-closed Normal/UV routing for the real BlendKit coin material.

Blender must open the caller-provided ``coin_star.blend`` before this script starts. The
current artist-authored ``Gold metal`` graph contains true displacement. Production
Normal / UV Segments cannot reproduce that render-time geometry and must reject the
request at PLAN_BAKE while directing the caller to Camera Projection or Depth Camera
Projection.

This regression deliberately keeps the artist material untouched. Geometry/rig Normal
acceptance is covered by separate real-coin gates that temporarily install a deterministic
surface-only test material. Here the contract is strict rejection with no output files and
no mutation of Blender source state.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tempfile
from time import perf_counter
import traceback

import bpy


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIRECTORY.parents[1]
for path in (SCRIPT_DIRECTORY, REPOSITORY_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from Blender_to_Spine2D_Mesh_Exporter.application import (  # noqa: E402
    A1SingleObjectExportSettings,
    A1SourceGeometryMode,
    ExportSettings,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter import (  # noqa: E402
    export_a1_single_object,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import (  # noqa: E402
    A1TextureExportMode,
    BakeExecutionSettings,
    BakeMode,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.uv import UvUnwrapSettings  # noqa: E402
from run_bake_integration import (  # noqa: E402
    _assert,
    _capture_scene_bake_state,
    _temporary_datablock_names,
)
from run_coin_star_real_blend_shader_capability_integration import (  # noqa: E402
    _datablock_fingerprint,
    _object_fingerprint,
    _require_loaded_blend,
    _require_source_object,
    _scene_fingerprint,
)


_TEXTURE_SIZE = 256
_MAX_EXPORT_SECONDS = 240.0
_OUTPUT_STEM = "Game_Gold_Coin_Normal"
_EXPECTED_STAGE = "PLAN_BAKE"
_EXPECTED_CODE = "A1_PLAN_BAKE_FAILED"
_EXPECTED_BLOCKER = "DISPLACEMENT_RENDER_REQUIRED"


def _parse_arguments() -> argparse.Namespace:
    arguments = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(
        description="Validate fail-closed real-coin Normal UV displacement routing."
    )
    parser.add_argument(
        "--expected-blend",
        required=True,
        help="Exact coin_star.blend path Blender must already have loaded.",
    )
    return parser.parse_args(arguments)


def _settings(output_directory: Path) -> A1SingleObjectExportSettings:
    if not isinstance(output_directory, Path):
        raise TypeError("output_directory must be pathlib.Path")

    return A1SingleObjectExportSettings(
        export=ExportSettings(
            texture_width=_TEXTURE_SIZE,
            texture_height=_TEXTURE_SIZE,
            output_directory=output_directory,
            images_relative_path="images",
            bake_margin=4,
        ),
        prefix="Game Gold Coin",
        output_stem=_OUTPUT_STEM,
        json_output_stem=_OUTPUT_STEM,
        source_geometry_mode=A1SourceGeometryMode.ORIGINAL,
        uv=UvUnwrapSettings(layer_name="SpineBakeUV"),
        diffuse_mode=BakeMode.DIFFUSE,
        procedural_mode=BakeMode.COMBINED,
        bake_execution=BakeExecutionSettings(
            render_engine="CYCLES",
            samples=1,
            texture_export_mode=A1TextureExportMode.NORMAL_UV_SEGMENTS,
        ),
    )


def _assert_fail_closed_issue(result) -> tuple[object, ...]:
    issues = tuple(result.issues)
    _assert(issues, "failed real-coin Normal export returned no diagnostics")

    matching = tuple(
        issue
        for issue in issues
        if getattr(issue, "stage", None) == _EXPECTED_STAGE
        and getattr(issue, "code", None) == _EXPECTED_CODE
    )
    _assert(
        matching,
        "real-coin Normal rejection lost PLAN_BAKE/A1_PLAN_BAKE_FAILED diagnostic: "
        f"{issues}",
    )

    for issue in matching:
        message = str(getattr(issue, "message", "") or "")
        technical = str(getattr(issue, "technical_details", "") or "")
        combined = f"{message}\n{technical}"
        _assert(
            _EXPECTED_BLOCKER in combined,
            f"real-coin Normal rejection lost displacement blocker: {issue}",
        )
        _assert(
            "Camera Projection" in combined,
            f"real-coin Normal rejection does not direct to Camera Projection: {issue}",
        )
        _assert(
            "Depth Camera Projection" in combined,
            f"real-coin Normal rejection does not direct to Depth Camera Projection: {issue}",
        )

    return matching


def _run(expected_blend: str) -> None:
    loaded = _require_loaded_blend(expected_blend)
    source = _require_source_object()
    _assert(
        bpy.context.scene.camera is not None,
        "real coin fail-closed gate requires the authored active camera",
    )

    scene_before = _scene_fingerprint()
    bake_before = _capture_scene_bake_state()
    object_before = _object_fingerprint(source)
    datablocks_before = _datablock_fingerprint()
    temporary_before = _temporary_datablock_names()

    with tempfile.TemporaryDirectory(
        prefix="spine2d-coin-real-normal-rejection-"
    ) as directory:
        output_directory = Path(directory)
        started = perf_counter()
        result = export_a1_single_object(
            source,
            _settings(output_directory),
            context=bpy.context,
            scene=bpy.context.scene,
        )
        elapsed = perf_counter() - started

        _assert(
            not result.success,
            "real coin artist displacement unexpectedly became exportable through Normal/UV",
        )
        _assert(
            elapsed <= _MAX_EXPORT_SECONDS,
            f"real coin fail-closed routing exceeded {_MAX_EXPORT_SECONDS}s: {elapsed:.3f}s",
        )
        matching = _assert_fail_closed_issue(result)
        _assert(
            not tuple(result.output_files),
            f"failed real-coin Normal export reported output files: {result.output_files}",
        )
        residual_paths = tuple(sorted(path for path in output_directory.rglob("*") if path.exists()))
        _assert(
            not residual_paths,
            f"failed real-coin Normal export left filesystem artifacts: {residual_paths}",
        )

        print(
            "[COIN-REAL-NORMAL-FAIL-CLOSED] PASS "
            f"blend={loaded} object={source.name_full!r} "
            f"issues={len(result.issues)} matching={len(matching)} "
            f"elapsed={elapsed:.3f}s stage={_EXPECTED_STAGE} code={_EXPECTED_CODE} "
            f"blocker={_EXPECTED_BLOCKER} outputs=0",
            flush=True,
        )

    _assert(_scene_fingerprint() == scene_before, "coin rejection changed Blender context")
    _assert(_capture_scene_bake_state() == bake_before, "coin rejection changed bake state")
    _assert(_object_fingerprint(source) == object_before, "coin rejection changed source data")
    _assert(
        _datablock_fingerprint() == datablocks_before,
        "coin rejection created or removed persistent Blender datablocks",
    )
    _assert(
        _temporary_datablock_names() == temporary_before,
        "coin rejection leaked temporary Blender datablocks",
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
