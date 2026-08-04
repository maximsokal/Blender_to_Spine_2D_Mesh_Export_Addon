"""Blender 5.2 multi-object and rollback acceptance for Depth parallax reserve.

Two folded objects are exported through the public standalone multi-object route. Each
object owns one FRONT and one face-isolated reserve texture/attachment, while one outer
AtomicFileTransaction owns the final JSON and all four PNG files.

A second run injects a public-progress failpoint immediately before staging object two by
removing the active Scene camera. Object one has already rendered into staged reservations;
the second runtime validation must fail and the transaction must leave no files behind.
No exporter internals are mocked.
"""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile

import bpy


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIRECTORY.parents[1]
for path in (SCRIPT_DIRECTORY, REPOSITORY_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from Blender_to_Spine2D_Mesh_Exporter.application import (  # noqa: E402
    A1ExportProgressUpdate,
    A1MultiObjectExportSettings,
    A1MultiObjectMode,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter import (  # noqa: E402
    A1MultiObjectSource,
    export_a1_multi_object,
    prepare_a1_multi_object,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.a1_object_preparation import (  # noqa: E402
    PreparedDepthA1Object,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine import MeshAttachment  # noqa: E402
from run_bake_integration import (  # noqa: E402
    PNG_SIGNATURE,
    _activate_only,
    _assert,
    _capture_context,
    _capture_scene_bake_state,
    _clear_scene,
    _create_sentinel,
    _material_fingerprint,
    _temporary_datablock_names,
)
from run_camera_projection_integration import (  # noqa: E402
    _configure_scene,
    _create_camera,
    _purge_orphan_scene_data,
    _read_image,
    _scene_render_fingerprint,
    _visible_and_transparent_counts,
)
import run_depth_parallax_integration as smoke  # noqa: E402


_LEFT_PREFIX = "DepthParallaxLeft"
_RIGHT_PREFIX = "DepthParallaxRight"
_MULTI_STEM = "DepthParallaxMulti"
_LEFT_COMPONENT = "left"
_RIGHT_COMPONENT = "right"


def _single_settings(output_directory: Path, prefix: str):
    base = smoke._settings(output_directory)
    return replace(
        base,
        prefix=prefix,
        output_stem=prefix,
        json_output_stem=prefix,
        export=replace(base.export, output_directory=output_directory),
    )


def _prepare_scene(output_directory: Path):
    _clear_scene()
    _purge_orphan_scene_data()
    _configure_scene()
    bpy.context.scene.cycles.samples = 1
    camera = _create_camera(name="DepthParallaxMultiCamera")

    left, left_front, left_reserve = smoke._create_folded_surface(
        f"{_LEFT_PREFIX}_Source"
    )
    right, right_front, right_reserve = smoke._create_folded_surface(
        f"{_RIGHT_PREFIX}_Source"
    )
    left.location = (-0.92, 0.0, 0.0)
    right.location = (0.92, 0.0, 0.0)

    sentinel = _create_sentinel()
    sentinel.location = (8.0, 0.0, 0.0)
    _activate_only(sentinel)
    left.select_set(False)
    right.select_set(False)
    bpy.context.scene.frame_set(1)
    bpy.context.view_layer.update()

    sources = (
        A1MultiObjectSource(
            source_object=left,
            component_id=_LEFT_COMPONENT,
            settings=_single_settings(output_directory, _LEFT_PREFIX),
        ),
        A1MultiObjectSource(
            source_object=right,
            component_id=_RIGHT_COMPONENT,
            settings=_single_settings(output_directory, _RIGHT_PREFIX),
        ),
    )
    settings = A1MultiObjectExportSettings(
        output_directory=output_directory,
        output_stem=_MULTI_STEM,
        mode=A1MultiObjectMode.STANDALONE,
    )
    materials = (left_front, left_reserve, right_front, right_reserve)
    return sources, settings, camera, materials


def _prepared_slot_triangles(item: PreparedDepthA1Object) -> dict[str, tuple[int, ...]]:
    result: dict[str, tuple[int, ...]] = {}
    for component in item.document_assembly.document_build.components:
        attachment = component.attachment
        _assert(
            isinstance(attachment, MeshAttachment),
            f"component {component.request.slot_name} is not a MeshAttachment",
        )
        result[component.request.slot_name] = tuple(
            int(value) for value in attachment.triangles
        )
    return result


def _expected_object_outputs(item: PreparedDepthA1Object) -> tuple[Path, Path]:
    _assert(len(item.bake_plan.frame_tasks) == 1, "FRONT must be static")
    _assert(len(item.reserve_bake_plans) == 1, "expected one reserve plan")
    _assert(
        len(item.reserve_bake_plans[0].frame_tasks) == 1,
        "reserve must be static",
    )
    return (
        item.bake_plan.frame_tasks[0].output_path,
        item.reserve_bake_plans[0].frame_tasks[0].output_path,
    )


def _assert_prepared_multi(prepared: object) -> tuple[tuple[str, str], ...]:
    _assert(len(prepared.objects) == 2, "expected two prepared objects")
    _assert(
        all(isinstance(item, PreparedDepthA1Object) for item in prepared.objects),
        "multi-object preparation lost PreparedDepthA1Object",
    )
    _assert(
        len(prepared.texture_output_paths) == 4,
        f"expected four realized texture paths: {prepared.texture_output_paths}",
    )

    slot_pairs = []
    expected_paths = []
    for source, item in zip(prepared.sources, prepared.objects, strict=True):
        package = item.depth_parallax_package
        _assert(package.front_face_indices == (0, 1), "front faces changed")
        _assert(package.reserve_face_indices == (2, 3), "reserve faces changed")
        _assert(len(package.reserve_surfaces) == 1, "expected one reserve surface")
        reserve_plan = item.reserve_bake_plans[0]
        _assert(
            reserve_plan.source_face_indices == (2, 3),
            "reserve plan lost source-face ownership",
        )
        reserve_slot = f"{item.prefix}_Parallax_{reserve_plan.view_id}"
        front_slot = f"{item.prefix}_Segment_0"
        projection_order = tuple(
            projection.request.slot_name
            for projection in item.document_assembly.projections
        )
        _assert(
            projection_order == (reserve_slot, front_slot),
            f"projection order changed for {source.component_id}: {projection_order}",
        )
        slot_pairs.append((reserve_slot, front_slot))
        expected_paths.extend(_expected_object_outputs(item))
        _assert(
            int(item.statistics.get("depth_parallax_attachment_count", -1)) == 2,
            f"prepared attachment count differs for {source.component_id}",
        )

    _assert(
        tuple(path.resolve(strict=False) for path in prepared.texture_output_paths)
        == tuple(path.resolve(strict=False) for path in expected_paths),
        "realized texture output order differs from object FRONT/reserve plans",
    )
    return tuple(slot_pairs)


def _assert_png(path: Path, label: str) -> tuple[int, int]:
    _assert(path.read_bytes().startswith(PNG_SIGNATURE), f"invalid PNG for {label}: {path}")
    size, pixels = _read_image(path)
    _assert(
        1 <= size[0] <= smoke._TEXTURE_SIZE
        and 1 <= size[1] <= smoke._TEXTURE_SIZE,
        f"invalid crop size for {label}: {size}",
    )
    visible, transparent = _visible_and_transparent_counts(pixels)
    _assert(visible > 100, f"too few visible pixels for {label}: {visible}")
    _assert(transparent > 0, f"no transparent padding for {label}")
    return size


def _serialized_slot_names(document: dict[str, object]) -> tuple[str, ...]:
    slots = document.get("slots")
    _assert(isinstance(slots, list), "serialized slots must be an array")
    return tuple(
        str(slot.get("name"))
        for slot in slots
        if isinstance(slot, dict)
        and str(slot.get("name", "")).startswith("DepthParallax")
    )


def _run_success(output_directory: Path) -> None:
    output_directory.mkdir(parents=True, exist_ok=False)
    sources, settings, camera, materials = _prepare_scene(output_directory)

    context_before = _capture_context()
    bake_before = _capture_scene_bake_state()
    render_before = _scene_render_fingerprint()
    camera_before = smoke._camera_fingerprint(camera)
    material_before = tuple(_material_fingerprint(item) for item in materials)
    temporary_before = _temporary_datablock_names()

    prepared = prepare_a1_multi_object(
        sources,
        settings,
        context=bpy.context,
        scene=bpy.context.scene,
    )
    slot_pairs = _assert_prepared_multi(prepared)
    _assert(
        not tuple(path for path in output_directory.rglob("*") if path.is_file()),
        "preparation wrote output files",
    )
    _assert(_capture_context() == context_before, "prepare changed Blender context")
    _assert(_capture_scene_bake_state() == bake_before, "prepare changed bake state")
    _assert(_scene_render_fingerprint() == render_before, "prepare changed render state")
    _assert(smoke._camera_fingerprint(camera) == camera_before, "prepare changed camera")
    _assert(_temporary_datablock_names() == temporary_before, "prepare leaked datablocks")

    result = export_a1_multi_object(
        sources,
        settings,
        context=bpy.context,
        scene=bpy.context.scene,
    )
    _assert(result.success, f"multi-object parallax export failed: {result.issues}")

    expected_texture_paths = tuple(
        path
        for item in prepared.objects
        for path in _expected_object_outputs(item)
    )
    expected_paths = (prepared.json_path, *expected_texture_paths)
    _assert(
        tuple(path.resolve(strict=False) for path in result.output_files)
        == tuple(path.resolve(strict=False) for path in expected_paths),
        f"multi-object output order differs: {result.output_files}",
    )
    _assert(len(result.output_files) == 5, "expected one JSON plus four PNG files")

    size_by_path: dict[Path, tuple[int, int]] = {}
    for index, path in enumerate(result.output_files[1:]):
        size_by_path[path.resolve(strict=False)] = _assert_png(path, f"texture[{index}]")

    document = json.loads(result.output_files[0].read_text(encoding="utf-8"))
    _assert(isinstance(document, dict), "serialized multi-object document must be object")
    slot_names = _serialized_slot_names(document)
    expected_slots = tuple(name for pair in slot_pairs for name in pair)
    _assert(
        slot_names == expected_slots,
        f"serialized object/reserve slot order differs: {slot_names}",
    )

    for item, (reserve_slot, front_slot) in zip(
        prepared.objects,
        slot_pairs,
        strict=True,
    ):
        triangles = _prepared_slot_triangles(item)
        front_path, reserve_path = _expected_object_outputs(item)
        smoke._assert_serialized_attachment(
            document,
            reserve_slot,
            size_by_path[reserve_path.resolve(strict=False)],
            triangles[reserve_slot],
        )
        smoke._assert_serialized_attachment(
            document,
            front_slot,
            size_by_path[front_path.resolve(strict=False)],
            triangles[front_slot],
        )

    expected_statistics = {
        "texture_output_count": 4,
        "output_file_count": 5,
        f"component.{_LEFT_COMPONENT}.depth_parallax_cropped_view_count": 2,
        f"component.{_RIGHT_COMPONENT}.depth_parallax_cropped_view_count": 2,
        f"component.{_LEFT_COMPONENT}.parallax_texture_output_count": 2,
        f"component.{_RIGHT_COMPONENT}.parallax_texture_output_count": 2,
    }
    for key, expected in expected_statistics.items():
        _assert(
            int(result.statistics.get(key, -1)) == expected,
            f"multi-object statistic {key!r} differs: {result.statistics.get(key)!r}",
        )

    _assert(_capture_context() == context_before, "export changed Blender context")
    _assert(_capture_scene_bake_state() == bake_before, "export changed bake state")
    _assert(_scene_render_fingerprint() == render_before, "export changed render state")
    _assert(smoke._camera_fingerprint(camera) == camera_before, "export changed camera")
    _assert(
        tuple(_material_fingerprint(item) for item in materials) == material_before,
        "export changed source materials",
    )
    _assert(_temporary_datablock_names() == temporary_before, "export leaked datablocks")


def _run_rollback(output_directory: Path) -> None:
    output_directory.mkdir(parents=True, exist_ok=False)
    sources, settings, camera, materials = _prepare_scene(output_directory)

    context_before = _capture_context()
    bake_before = _capture_scene_bake_state()
    render_before = _scene_render_fingerprint()
    camera_before = smoke._camera_fingerprint(camera)
    material_before = tuple(_material_fingerprint(item) for item in materials)
    temporary_before = _temporary_datablock_names()
    failpoint_triggered = False

    def fail_before_second_object(update: A1ExportProgressUpdate) -> None:
        nonlocal failpoint_triggered
        if not isinstance(update, A1ExportProgressUpdate):
            raise TypeError("progress callback received an invalid update")
        if (
            not failpoint_triggered
            and update.stage == "STAGE_OUTPUTS"
            and update.object_id == _RIGHT_COMPONENT
            and "Staging textures for" in update.message
        ):
            bpy.context.scene.camera = None
            failpoint_triggered = True

    try:
        result = export_a1_multi_object(
            sources,
            settings,
            context=bpy.context,
            scene=bpy.context.scene,
            progress_callback=fail_before_second_object,
        )
    finally:
        bpy.context.scene.camera = camera
        bpy.context.view_layer.update()

    _assert(failpoint_triggered, "rollback failpoint did not reach object two staging")
    _assert(not result.success, "rollback case unexpectedly succeeded")
    remaining_files = tuple(
        path for path in output_directory.rglob("*") if path.is_file()
    )
    _assert(
        not remaining_files,
        f"atomic rollback left output or staging files: {remaining_files}",
    )
    _assert(_capture_context() == context_before, "rollback changed Blender context")
    _assert(_capture_scene_bake_state() == bake_before, "rollback changed bake state")
    _assert(_scene_render_fingerprint() == render_before, "rollback changed render state")
    _assert(smoke._camera_fingerprint(camera) == camera_before, "rollback changed camera")
    _assert(
        tuple(_material_fingerprint(item) for item in materials) == material_before,
        "rollback changed source materials",
    )
    _assert(_temporary_datablock_names() == temporary_before, "rollback leaked datablocks")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="spine2d-depth-parallax-multi-") as directory:
        output_root = Path(directory)
        _run_success(output_root / "success")
        _run_rollback(output_root / "rollback")
    print(
        "[DEPTH-PARALLAX-MULTI] PASS objects=2 textures=4 attachments=4 "
        "transaction=shared rollback=no-files"
    )


if __name__ == "__main__":
    main()
