"""Compare +Z and Active Camera through the public Normal / UV Segments route.

Blender must open the caller-provided ``coin_star.blend`` before this script starts. The
regression proves three independent contracts on one real production material:

* every prepared side region survives both projection directions;
* Active Camera keeps the ordinary object-pivot rig and per-vertex depth groups;
* material baking uses identical unprojected Blender-local geometry in both directions.

The generated destination UV layout may differ by projection, so image comparison uses
broad luminance distribution metrics rather than byte identity.
"""

from __future__ import annotations

import argparse
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

from Blender_to_Spine2D_Mesh_Exporter.application import (  # noqa: E402
    A1SingleObjectExportSettings,
    A1SourceGeometryMode,
    ExportSettings,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter import (  # noqa: E402
    export_a1_single_object,
    prepare_a1_object,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import (  # noqa: E402
    A1TextureExportMode,
    BakeExecutionSettings,
    BakeMode,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.projection import (  # noqa: E402
    A1ProjectionDirection,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.rig_profiles import (  # noqa: E402
    A1RigSetupPoseMode,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.uv import (  # noqa: E402
    UvUnwrapSettings,
)
from run_bake_integration import (  # noqa: E402
    PNG_SIGNATURE,
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
_MAX_EXPORT_SECONDS = 300.0
_GENERATED_UV_LAYER = "SpineBakeUV"
_MAX_LUMINANCE_RELATIVE_DELTA = 0.25


def _parse_arguments() -> argparse.Namespace:
    arguments = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(
        description="Compare real coin Normal +Z and Active Camera exports."
    )
    parser.add_argument(
        "--expected-blend",
        required=True,
        help="Exact coin_star.blend path Blender must already have loaded.",
    )
    return parser.parse_args(arguments)


def _settings(
    output_directory: Path,
    direction: A1ProjectionDirection,
) -> A1SingleObjectExportSettings:
    if not isinstance(output_directory, Path):
        raise TypeError("output_directory must be pathlib.Path")
    if not isinstance(direction, A1ProjectionDirection):
        raise TypeError("direction must be A1ProjectionDirection")

    token = (
        "ActiveCamera"
        if direction is A1ProjectionDirection.ACTIVE_CAMERA
        else direction.value
    )
    output_stem = f"Game_Gold_Coin_Normal_{token}"
    return A1SingleObjectExportSettings(
        export=ExportSettings(
            texture_width=_TEXTURE_SIZE,
            texture_height=_TEXTURE_SIZE,
            output_directory=output_directory,
            images_relative_path="images",
            bake_margin=4,
        ),
        prefix="Game Gold Coin",
        output_stem=output_stem,
        json_output_stem=output_stem,
        source_geometry_mode=A1SourceGeometryMode.ORIGINAL,
        uv=UvUnwrapSettings(layer_name=_GENERATED_UV_LAYER),
        diffuse_mode=BakeMode.DIFFUSE,
        procedural_mode=BakeMode.COMBINED,
        bake_execution=BakeExecutionSettings(
            render_engine="CYCLES",
            samples=1,
            texture_export_mode=A1TextureExportMode.NORMAL_UV_SEGMENTS,
        ),
        rig_setup_pose_mode=A1RigSetupPoseMode.PRESERVE_COMPOSITION,
        projection_direction=direction,
    )


def _source_material_geometry_fingerprint(snapshot) -> tuple:
    """Ignore generated destination UV while retaining all material inputs."""

    loop_records = tuple(
        (
            loop.source_id,
            loop.vertex_id,
            loop.edge_id,
            tuple(
                (uv.layer_name, uv.coordinate)
                for uv in loop.uvs
                if uv.layer_name != _GENERATED_UV_LAYER
            ),
        )
        for loop in snapshot.loops
    )
    return (
        snapshot.source_object_id,
        snapshot.vertices,
        snapshot.edges,
        loop_records,
        snapshot.faces,
        snapshot.world_matrix,
        snapshot.render_uv_layer,
    )


def _mesh_uv_stream_count(value: object) -> int:
    if isinstance(value, dict):
        count = 1 if isinstance(value.get("uvs"), list) and value["uvs"] else 0
        return count + sum(_mesh_uv_stream_count(child) for child in value.values())
    if isinstance(value, list):
        return sum(_mesh_uv_stream_count(child) for child in value)
    return 0


def _percentile(sorted_values: tuple[float, ...], fraction: float) -> float:
    if not sorted_values:
        raise ValueError("sorted_values cannot be empty")
    if fraction < 0.0 or fraction > 1.0:
        raise ValueError("fraction must be in [0, 1]")
    index = int(round((len(sorted_values) - 1) * fraction))
    return float(sorted_values[index])


def _visible_luminance_metrics(path: Path) -> tuple[int, float, float, float]:
    if not isinstance(path, Path):
        raise TypeError("path must be pathlib.Path")
    image = None
    try:
        image = bpy.data.images.load(str(path), check_existing=False)
        pixels = tuple(float(value) for value in image.pixels[:])
        expected = int(image.size[0]) * int(image.size[1]) * 4
        _assert(
            len(pixels) == expected,
            f"loaded Normal PNG has invalid RGBA length: {len(pixels)} != {expected}",
        )

        luminance: list[float] = []
        for index in range(0, len(pixels), 4):
            if pixels[index + 3] <= 1.0 / 255.0:
                continue
            red, green, blue = pixels[index : index + 3]
            luminance.append(
                0.2126 * red + 0.7152 * green + 0.0722 * blue
            )

        _assert(luminance, f"Normal texture contains no visible pixels: {path}")
        ordered = tuple(sorted(luminance))
        mean = sum(ordered) / float(len(ordered))
        median = _percentile(ordered, 0.5)
        upper = _percentile(ordered, 0.9)
        return len(ordered), mean, median, upper
    finally:
        if image is not None:
            bpy.data.images.remove(image)


def _relative_delta(first: float, second: float) -> float:
    return abs(float(first) - float(second)) / max(
        abs(float(first)),
        abs(float(second)),
        1.0e-6,
    )


def _assert_luminance_parity(
    axis_metrics: tuple[int, float, float, float],
    camera_metrics: tuple[int, float, float, float],
) -> None:
    labels = ("mean", "median", "p90")
    for label, axis_value, camera_value in zip(
        labels,
        axis_metrics[1:],
        camera_metrics[1:],
        strict=True,
    ):
        delta = _relative_delta(axis_value, camera_value)
        _assert(
            delta <= _MAX_LUMINANCE_RELATIVE_DELTA,
            "Normal projection changed material brightness beyond tolerance: "
            f"metric={label}, axis={axis_value:.6f}, "
            f"camera={camera_value:.6f}, relative_delta={delta:.6f}, "
            f"limit={_MAX_LUMINANCE_RELATIVE_DELTA:.6f}",
        )


def _single_json_and_png(result) -> tuple[Path, Path]:
    outputs = tuple(Path(path).resolve(strict=False) for path in result.output_files)
    json_files = tuple(path for path in outputs if path.suffix.lower() == ".json")
    png_files = tuple(path for path in outputs if path.suffix.lower() == ".png")
    _assert(len(outputs) == 2, f"expected one JSON and one PNG, got {outputs}")
    _assert(len(json_files) == 1, f"expected one JSON output, got {json_files}")
    _assert(len(png_files) == 1, f"expected one PNG output, got {png_files}")
    _assert(
        all(path.is_file() and path.stat().st_size > 8 for path in outputs),
        f"Normal export contains missing or empty files: {outputs}",
    )
    _assert(
        png_files[0].read_bytes().startswith(PNG_SIGNATURE),
        f"Normal texture is not a PNG: {png_files[0]}",
    )
    return json_files[0], png_files[0]


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
        axis_settings = _settings(
            root / "axis",
            A1ProjectionDirection.POSITIVE_Z,
        )
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
            _source_material_geometry_fingerprint(
                axis_prepared.bake_target_snapshot
            )
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
            is A1RigSetupPoseMode.PRESERVE_COMPOSITION,
            "Active Camera Normal changed to camera-zero setup pose",
        )
        _assert(
            camera_prepared.rig.request.camera_layer_projection_kind is None,
            "Active Camera Normal retained Camera Projection layer semantics",
        )
        _assert(
            len(camera_prepared.rig.info.z_groups)
            == len(axis_prepared.rig.info.z_groups)
            and len(camera_prepared.rig.info.z_groups) > 1,
            "Active Camera Normal lost ordinary per-depth rig groups",
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

        _assert(axis_result.success, f"real coin +Z Normal export failed: {axis_result.issues}")
        _assert(
            camera_result.success,
            f"real coin Active Camera Normal export failed: {camera_result.issues}",
        )
        _assert(
            axis_elapsed <= _MAX_EXPORT_SECONDS,
            f"+Z Normal export exceeded {_MAX_EXPORT_SECONDS}s: {axis_elapsed:.3f}s",
        )
        _assert(
            camera_elapsed <= _MAX_EXPORT_SECONDS,
            "Active Camera Normal export exceeded "
            f"{_MAX_EXPORT_SECONDS}s: {camera_elapsed:.3f}s",
        )

        axis_json, axis_png = _single_json_and_png(axis_result)
        camera_json, camera_png = _single_json_and_png(camera_result)
        axis_document = json.loads(axis_json.read_text(encoding="utf-8"))
        camera_document = json.loads(camera_json.read_text(encoding="utf-8"))
        axis_uv_streams = _mesh_uv_stream_count(axis_document)
        camera_uv_streams = _mesh_uv_stream_count(camera_document)
        _assert(
            axis_uv_streams == camera_uv_streams
            and axis_uv_streams == len(axis_prepared.document_assembly.projections),
            "serialized Normal mesh count differs by projection: "
            f"axis={axis_uv_streams}, camera={camera_uv_streams}, "
            f"prepared={len(axis_prepared.document_assembly.projections)}",
        )

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
                statistics.get("bake_mode") == BakeMode.COMBINED.value,
                f"{label} export did not use COMBINED bake: {statistics}",
            )
            _assert(
                statistics.get("material_bake_projection_independent") == 1,
                f"{label} export lost projection-independent material geometry: "
                f"{statistics}",
            )
            _assert(
                statistics.get("material_bake_uv_transfer_count")
                == len(axis_prepared.bake_target_snapshot.loops),
                f"{label} export did not transfer every generated UV loop: {statistics}",
            )
            _assert(
                "projection_crop_width" not in statistics,
                f"{label} Normal export silently became Camera Projection: {statistics}",
            )

        _assert(
            camera_result.statistics.get("final_rig_setup_pose_mode")
            == A1RigSetupPoseMode.PRESERVE_COMPOSITION.value,
            f"Active Camera Normal serialized the wrong setup pose: "
            f"{camera_result.statistics}",
        )
        _assert(
            camera_result.statistics.get("camera_relative_depth_group_count") == 0,
            f"Active Camera Normal retained camera-relative depth groups: "
            f"{camera_result.statistics}",
        )
        _assert(
            camera_result.statistics.get("depth_setup_y_compensated") == 0,
            f"Active Camera Normal retained camera-zero compensation: "
            f"{camera_result.statistics}",
        )
        _assert(
            camera_result.statistics.get("normal_active_camera_depth_group_count")
            == axis_result.statistics.get("z_group_count"),
            "Active Camera Normal depth-group count differs from ordinary Normal: "
            f"axis={axis_result.statistics}, camera={camera_result.statistics}",
        )
        _assert(
            axis_result.statistics.get("bake_strategy_ids")
            == camera_result.statistics.get("bake_strategy_ids"),
            "Normal projection changed material bake strategy: "
            f"axis={axis_result.statistics}, camera={camera_result.statistics}",
        )

        axis_metrics = _visible_luminance_metrics(axis_png)
        camera_metrics = _visible_luminance_metrics(camera_png)
        _assert_luminance_parity(axis_metrics, camera_metrics)

        print(
            "[COIN-NORMAL-PROJECTION-PARITY] PASS "
            f"blend={loaded} object={source.name_full!r} "
            f"segments={axis_uv_streams} depth_groups="
            f"{len(camera_prepared.rig.info.z_groups)} "
            f"axis_luma=({axis_metrics[1]:.6f},{axis_metrics[2]:.6f},"
            f"{axis_metrics[3]:.6f}) "
            f"camera_luma=({camera_metrics[1]:.6f},{camera_metrics[2]:.6f},"
            f"{camera_metrics[3]:.6f}) "
            f"axis_elapsed={axis_elapsed:.3f}s "
            f"camera_elapsed={camera_elapsed:.3f}s "
            "setup=PRESERVE_COMPOSITION pivot=OBJECT_ORIGIN "
            "material_geometry=projection-independent",
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
    try:
        _run(arguments.expected_blend)
    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
