"""Compare +Z and Active Camera through the current public Normal / UV Segments route.

Deep parity helpers are preserved in ``coin_star_normal_projection_parity_core`` so this
runner can track the production execution contract without duplicating hundreds of lines.
Standalone execution temporarily installs a deterministic surface-only material because
the real artist ``Gold metal`` now contains true displacement and is intentionally blocked
from Normal/UV. Callers such as the publication gate may invoke ``_run`` while owning their
own compatible material context.
"""

from __future__ import annotations

import json
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

from Blender_to_Spine2D_Mesh_Exporter.application import A1SourceGeometryMode  # noqa: E402
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter import (  # noqa: E402
    export_a1_single_object,
    prepare_a1_object,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import (  # noqa: E402
    A1TextureExportMode,
    BakeMode,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.projection import (  # noqa: E402
    A1ProjectionDirection,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.rig_profiles import (  # noqa: E402
    A1RigSetupPoseMode,
)
from coin_star_normal_projection_parity_core import (  # noqa: E402
    _MAX_EXPORT_SECONDS,
    _assert_luminance_parity,
    _assert_prepared_depth_groups,
    _assert_serialized_active_camera_normal_setup,
    _mesh_uv_stream_count,
    _parse_arguments,
    _settings,
    _single_json_and_png,
    _source_material_geometry_fingerprint,
    _visible_luminance_metrics,
)
from coin_star_normal_test_support import safe_coin_normal_material  # noqa: E402
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


_REQUIRED_CAMERA_STRATEGIES = frozenset({"CAMERA_SURFACE_COLOR", "ALPHA"})
_OBSOLETE_CAMERA_STRATEGY = "CAMERA_COMBINED"


def _strategy_ids(statistics: dict[str, object], *, label: str) -> frozenset[str]:
    if not isinstance(statistics, dict):
        raise TypeError("statistics must be a dict")
    if not isinstance(label, str) or not label.strip():
        raise ValueError("label must be non-empty")

    raw = str(statistics.get("bake_strategy_ids", "") or "")
    values = frozenset(token.strip() for token in raw.split(",") if token.strip())
    _assert(values, f"{label} export reported no bake strategy IDs: {statistics}")
    _assert(
        _REQUIRED_CAMERA_STRATEGIES.issubset(values),
        f"{label} export lost current camera surface-color/alpha strategies: {values}",
    )
    _assert(
        _OBSOLETE_CAMERA_STRATEGY not in values,
        f"{label} export regressed to obsolete CAMERA_COMBINED strategy: {values}",
    )
    return values


def _run(expected_blend: str) -> None:
    loaded = _require_loaded_blend(expected_blend)
    source = _require_source_object()
    _assert(
        bpy.context.scene.camera is not None,
        "real coin Active Camera Normal export requires an active scene camera",
    )

    scene_before = _scene_fingerprint()
    bake_before = _capture_scene_bake_state()
    object_before = _object_fingerprint(source)
    datablocks_before = _datablock_fingerprint()
    temporary_before = _temporary_datablock_names()

    with tempfile.TemporaryDirectory(
        prefix="spine2d-coin-normal-projection-parity-"
    ) as directory:
        root = Path(directory)
        axis_settings = _settings(root / "axis", A1ProjectionDirection.POSITIVE_Z)
        camera_settings = _settings(
            root / "active-camera",
            A1ProjectionDirection.ACTIVE_CAMERA,
        )

        axis_prepared = prepare_a1_object(
            source,
            axis_settings,
            context=bpy.context,
            scene=bpy.context.scene,
        )
        camera_prepared = prepare_a1_object(
            source,
            camera_settings,
            context=bpy.context,
            scene=bpy.context.scene,
        )

        _assert(
            axis_prepared.source_snapshot != camera_prepared.source_snapshot,
            "projection parity gate did not create distinct export geometry",
        )
        _assert(
            _source_material_geometry_fingerprint(axis_prepared.bake_target_snapshot)
            == _source_material_geometry_fingerprint(
                camera_prepared.bake_target_snapshot
            ),
            "Normal projection direction changed source-material bake geometry",
        )
        _assert(
            len(axis_prepared.document_assembly.projections)
            == len(camera_prepared.document_assembly.projections),
            "Normal projection directions produced different segment counts: "
            f"axis={len(axis_prepared.document_assembly.projections)}, "
            f"camera={len(camera_prepared.document_assembly.projections)}",
        )
        _assert(
            len(axis_prepared.document_assembly.projections) > 8,
            "real coin parity gate did not retain expected side regions",
        )
        _assert(
            camera_prepared.rig.request.setup_pose_mode
            is A1RigSetupPoseMode.CAMERA_VIEW_NORMAL,
            "Active Camera Normal did not use neutral object-pivot setup",
        )
        _assert(
            camera_prepared.rig.request.camera_layer_projection_kind is None,
            "Active Camera Normal retained Camera Projection layer semantics",
        )

        axis_depth_group_count = _assert_prepared_depth_groups(
            "+Z Normal",
            axis_prepared,
        )
        camera_depth_group_count = _assert_prepared_depth_groups(
            "Active Camera Normal",
            camera_prepared,
        )

        started = perf_counter()
        axis_result = export_a1_single_object(
            source,
            axis_settings,
            context=bpy.context,
            scene=bpy.context.scene,
        )
        axis_elapsed = perf_counter() - started

        started = perf_counter()
        camera_result = export_a1_single_object(
            source,
            camera_settings,
            context=bpy.context,
            scene=bpy.context.scene,
        )
        camera_elapsed = perf_counter() - started

        _assert(
            axis_result.success,
            f"real coin +Z Normal export failed: {axis_result.issues}",
        )
        _assert(
            camera_result.success,
            f"real coin Active Camera Normal export failed: {camera_result.issues}",
        )
        _assert(
            axis_elapsed <= _MAX_EXPORT_SECONDS,
            f"+Z Normal export exceeded limit: {axis_elapsed:.3f}s",
        )
        _assert(
            camera_elapsed <= _MAX_EXPORT_SECONDS,
            f"Active Camera export exceeded limit: {camera_elapsed:.3f}s",
        )

        axis_json, axis_png = _single_json_and_png(axis_result)
        camera_json, camera_png = _single_json_and_png(camera_result)
        axis_document = json.loads(axis_json.read_text(encoding="utf-8"))
        camera_document = json.loads(camera_json.read_text(encoding="utf-8"))
        axis_uv_streams = _mesh_uv_stream_count(axis_document)
        camera_uv_streams = _mesh_uv_stream_count(camera_document)
        _assert(
            axis_uv_streams == camera_uv_streams
            and axis_uv_streams
            == len(axis_prepared.document_assembly.projections),
            "serialized Normal mesh count differs by projection: "
            f"axis={axis_uv_streams}, camera={camera_uv_streams}",
        )
        neutral_constraint_count, inverse_setup_count = (
            _assert_serialized_active_camera_normal_setup(
                camera_document,
                camera_prepared,
            )
        )

        strategies_by_label: dict[str, frozenset[str]] = {}
        for label, statistics in (
            ("axis", axis_result.statistics),
            ("camera", camera_result.statistics),
        ):
            _assert(
                statistics.get("texture_export_mode")
                == A1TextureExportMode.NORMAL_UV_SEGMENTS.value,
                f"{label} export changed requested texture mode: {statistics}",
            )
            _assert(
                statistics.get("source_geometry_mode")
                == A1SourceGeometryMode.ORIGINAL.value,
                f"{label} export changed public Normal source route: {statistics}",
            )
            _assert(
                statistics.get("texture_pipeline") == "OBJECT_BAKE",
                f"{label} export did not use object bake: {statistics}",
            )
            _assert(
                statistics.get("bake_mode") == BakeMode.EMIT.value,
                f"{label} export did not use current EMIT surface-color bake: {statistics}",
            )
            strategies_by_label[label] = _strategy_ids(statistics, label=label)
            _assert(
                statistics.get("material_bake_projection_independent") == 1,
                f"{label} export lost projection-independent material geometry",
            )
            _assert(
                "projection_crop_width" not in statistics,
                f"{label} Normal export silently became Camera Projection",
            )

        _assert(
            strategies_by_label["axis"] == strategies_by_label["camera"],
            "Normal projection changed material bake strategy: "
            f"axis={strategies_by_label['axis']}, camera={strategies_by_label['camera']}",
        )
        _assert(
            axis_result.statistics.get("z_group_count") == axis_depth_group_count,
            "ordinary +Z Normal statistics lost depth-group count",
        )
        _assert(
            camera_result.statistics.get("z_group_count")
            == camera_depth_group_count,
            "Active Camera Normal statistics lost depth-group count",
        )
        _assert(
            camera_result.statistics.get("final_rig_setup_pose_mode")
            == A1RigSetupPoseMode.CAMERA_VIEW_NORMAL.value,
            "Active Camera Normal serialized the wrong setup pose",
        )
        _assert(
            camera_result.statistics.get("normal_active_camera_setup_neutral") == 1,
            "Active Camera Normal did not report neutral setup",
        )
        _assert(
            camera_result.statistics.get("camera_relative_depth_group_count") == 0,
            "Active Camera Normal retained camera-relative depth groups",
        )
        _assert(
            camera_result.statistics.get("normal_active_camera_depth_group_count")
            == camera_depth_group_count,
            "Active Camera depth statistics disagree with prepared plan",
        )

        axis_metrics = _visible_luminance_metrics(axis_png)
        camera_metrics = _visible_luminance_metrics(camera_png)
        _assert_luminance_parity(axis_metrics, camera_metrics)

        print(
            "[COIN-NORMAL-PROJECTION-PARITY] PASS "
            f"blend={loaded} object={source.name_full!r} "
            f"segments={axis_uv_streams} "
            f"axis_depth_groups={axis_depth_group_count} "
            f"camera_depth_groups={camera_depth_group_count} "
            f"neutral_constraints={neutral_constraint_count} "
            f"inverse_setup_bones={inverse_setup_count} "
            f"axis_luma=({axis_metrics[1]:.6f},{axis_metrics[2]:.6f},"
            f"{axis_metrics[3]:.6f}) "
            f"camera_luma=({camera_metrics[1]:.6f},{camera_metrics[2]:.6f},"
            f"{camera_metrics[3]:.6f}) "
            f"axis_elapsed={axis_elapsed:.3f}s "
            f"camera_elapsed={camera_elapsed:.3f}s "
            "setup=CAMERA_VIEW_NORMAL pivot=OBJECT_ORIGIN "
            "depth_setup=neutral+inverse material_geometry=projection-independent "
            "bake=EMIT strategy=CAMERA_SURFACE_COLOR+ALPHA",
            flush=True,
        )

    _assert(_scene_fingerprint() == scene_before, "parity export changed Blender context")
    _assert(_capture_scene_bake_state() == bake_before, "parity export changed bake state")
    _assert(_object_fingerprint(source) == object_before, "parity export changed source data")
    _assert(
        _datablock_fingerprint() == datablocks_before,
        "parity export created or removed persistent Blender datablocks",
    )
    _assert(
        _temporary_datablock_names() == temporary_before,
        "parity export leaked temporary Blender datablocks",
    )


def main() -> None:
    arguments = _parse_arguments()
    source = _require_source_object()
    try:
        with safe_coin_normal_material(source):
            _run(arguments.expected_blend)
    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
