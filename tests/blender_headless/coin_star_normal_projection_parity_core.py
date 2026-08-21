"""Compare +Z and Active Camera through the public Normal / UV Segments route.

The real-coin gate proves that projection direction does not change retained Normal
regions or source-material bake geometry. Active Camera Object Root must additionally
keep every camera-depth group, use neutral full-rank setup constraints, and provide one
inverse-setup child per group so projected X/Y is not replaced by camera depth.
"""

from __future__ import annotations

import argparse
import json
from math import isfinite
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
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.two_axis_scale_profile import (  # noqa: E402
    TwoAxisScaleRigProfile,
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
_NEUTRAL_TOLERANCE = 1.0e-6


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


def _transform_constraints_by_name(document: object) -> dict[str, dict[str, object]]:
    _assert(isinstance(document, dict), "serialized Spine document must be a mapping")
    raw_constraints = document.get("transform", [])
    _assert(
        isinstance(raw_constraints, list),
        "serialized transform constraints must be a list",
    )
    result: dict[str, dict[str, object]] = {}
    for index, item in enumerate(raw_constraints):
        _assert(isinstance(item, dict), f"transform[{index}] must be a mapping")
        name = item.get("name")
        _assert(
            isinstance(name, str) and bool(name.strip()),
            f"transform[{index}].name must be non-empty",
        )
        _assert(name not in result, f"duplicate serialized transform name: {name}")
        result[name] = item
    return result


def _serialized_bones_by_name(document: object) -> dict[str, dict[str, object]]:
    _assert(isinstance(document, dict), "serialized Spine document must be a mapping")
    raw_bones = document.get("bones", [])
    _assert(isinstance(raw_bones, list), "serialized bones must be a list")
    result: dict[str, dict[str, object]] = {}
    for index, item in enumerate(raw_bones):
        _assert(isinstance(item, dict), f"bones[{index}] must be a mapping")
        name = item.get("name")
        _assert(
            isinstance(name, str) and bool(name.strip()),
            f"bones[{index}].name must be non-empty",
        )
        _assert(name not in result, f"duplicate serialized bone name: {name}")
        result[name] = item
    return result


def _finite_number(
    mapping: dict[str, object],
    key: str,
    *,
    default: float,
    path: str,
) -> float:
    raw = mapping.get(key, default)
    _assert(
        isinstance(raw, (int, float)) and not isinstance(raw, bool),
        f"{path}.{key} must be numeric: {raw!r}",
    )
    value = float(raw)
    _assert(isfinite(value), f"{path}.{key} must be finite: {value}")
    return value


def _assert_serialized_active_camera_normal_setup(
    document: object,
    prepared,
) -> tuple[int, int]:
    """Require neutral constraints and one exact inverse child per depth group."""

    constraints = _transform_constraints_by_name(document)
    bones = _serialized_bones_by_name(document)
    profile = prepared.rig.profile
    prefix = prepared.prefix
    depth_constraint_name = (
        profile.scale_depth_constraint(prefix)
        if isinstance(profile, TwoAxisScaleRigProfile)
        else profile.scale_constraint(prefix)
    )
    names = (
        profile.rotation_x_constraint(prefix),
        profile.rotation_y_constraint(prefix),
        depth_constraint_name,
    )
    for name in names:
        _assert(name in constraints, f"missing Active Camera Normal constraint: {name}")

    checked = 0
    for role, constraint in (
        ("rotation_x", constraints[names[0]]),
        ("rotation_y", constraints[names[1]]),
    ):
        path = f"transform[{constraint['name']}]"
        rotation = _finite_number(
            constraint,
            "rotation",
            default=0.0,
            path=path,
        )
        mix_rotate = _finite_number(
            constraint,
            "mixRotate",
            default=1.0,
            path=path,
        )
        _assert(
            abs(rotation) <= _NEUTRAL_TOLERANCE,
            f"Active Camera Normal {role} retained setup rotation: {rotation}",
        )
        _assert(
            abs(mix_rotate - 1.0) <= _NEUTRAL_TOLERANCE,
            f"Active Camera Normal {role} disabled live rotation: {mix_rotate}",
        )
        checked += 1

    depth_scale = constraints[names[2]]
    depth_path = f"transform[{depth_scale['name']}]"
    depth_x = _finite_number(depth_scale, "x", default=0.0, path=depth_path)
    depth_scale_x = _finite_number(
        depth_scale,
        "scaleX",
        default=0.0,
        path=depth_path,
    )
    _assert(
        abs(depth_x) <= _NEUTRAL_TOLERANCE,
        f"Active Camera Normal depth setup translation is not neutral: {depth_x}",
    )
    _assert(
        abs(depth_scale_x) <= _NEUTRAL_TOLERANCE,
        f"Active Camera Normal depth setup is singular: {depth_scale_x}",
    )
    checked += 1

    compensation_count = 0
    for group in prepared.rig.info.z_groups:
        compensation_name = profile.z_camera_setup_bone(prefix, group.index)
        _assert(
            compensation_name in bones,
            f"missing Active Camera inverse setup bone: {compensation_name}",
        )
        compensation = bones[compensation_name]
        _assert(
            compensation.get("parent") == group.bone_name,
            f"inverse setup bone has wrong parent: {compensation_name}",
        )
        compensation_y = _finite_number(
            compensation,
            "y",
            default=0.0,
            path=f"bones[{compensation_name}]",
        )
        _assert(
            abs(compensation_y + float(group.y_offset_pixels))
            <= _NEUTRAL_TOLERANCE,
            "inverse setup does not cancel depth translation: "
            f"group={group.index}, depth={group.y_offset_pixels}, "
            f"inverse={compensation_y}",
        )
        compensation_count += 1

    _assert(checked == 3, f"unexpected Active Camera setup check count: {checked}")
    _assert(
        compensation_count == len(prepared.rig.info.z_groups),
        "inverse setup bone count differs from depth-group count",
    )
    return checked, compensation_count


def _assert_prepared_depth_groups(label: str, prepared) -> int:
    """Require one complete self-consistent ordinary Normal depth pipeline."""

    _assert(
        isinstance(label, str) and bool(label.strip()),
        "depth-group validation label must be non-empty",
    )
    plan = prepared.z_groups
    rig = prepared.rig
    plan_group_count = len(plan.groups)
    rig_group_count = len(rig.info.z_groups)
    vertex_count = len(prepared.source_snapshot.vertices)
    binding_count = len(plan.source_bindings)

    _assert(
        plan_group_count > 1,
        f"{label} collapsed the volumetric real coin to one depth group: "
        f"groups={plan_group_count}",
    )
    _assert(
        rig_group_count == plan_group_count,
        f"{label} rig lost prepared depth groups: "
        f"plan={plan_group_count}, rig={rig_group_count}",
    )
    _assert(
        tuple(rig.request.z_groups) == tuple(plan.groups),
        f"{label} rig request changed the prepared depth-group plan",
    )
    _assert(
        binding_count == vertex_count,
        f"{label} depth bindings do not cover every projected vertex: "
        f"bindings={binding_count}, vertices={vertex_count}",
    )

    expected_group_indices = set(
        range(plan.z_index_base, plan.z_index_base + plan_group_count)
    )
    bound_group_indices = {
        binding.z_group_index for binding in plan.source_bindings
    }
    _assert(
        bound_group_indices == expected_group_indices,
        f"{label} has unused or missing prepared depth groups: "
        f"expected={tuple(sorted(expected_group_indices))}, "
        f"bound={tuple(sorted(bound_group_indices))}",
    )
    return plan_group_count


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

        _assert(axis_result.success, f"real coin +Z Normal export failed: {axis_result.issues}")
        _assert(
            camera_result.success,
            f"real coin Active Camera Normal export failed: {camera_result.issues}",
        )
        _assert(axis_elapsed <= _MAX_EXPORT_SECONDS, f"+Z Normal export exceeded limit: {axis_elapsed:.3f}s")
        _assert(camera_elapsed <= _MAX_EXPORT_SECONDS, f"Active Camera export exceeded limit: {camera_elapsed:.3f}s")

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
            f"axis={axis_uv_streams}, camera={camera_uv_streams}",
        )
        neutral_constraint_count, inverse_setup_count = (
            _assert_serialized_active_camera_normal_setup(
                camera_document,
                camera_prepared,
            )
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
                f"{label} export lost projection-independent material geometry",
            )
            _assert(
                "projection_crop_width" not in statistics,
                f"{label} Normal export silently became Camera Projection",
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
        _assert(
            axis_result.statistics.get("bake_strategy_ids")
            == camera_result.statistics.get("bake_strategy_ids"),
            "Normal projection changed material bake strategy",
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
            "depth_setup=neutral+inverse material_geometry=projection-independent",
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
