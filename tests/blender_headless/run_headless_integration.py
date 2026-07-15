"""Real Blender 4.4 integration checks for the rewritten adapters.

This script is executed by Blender itself in background mode. It intentionally
creates all fixtures through the Blender data API so the repository does not need
binary .blend files for the foundational adapter tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import traceback
from unittest import mock

import bpy


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from Blender_to_Spine2D_Mesh_Exporter.blender_adapter import (  # noqa: E402
    EvaluatedMeshReadError,
    UvUnwrapError,
    read_evaluated_mesh_snapshot,
    read_source_mesh_snapshot,
    unwrap_snapshot_uv,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter import uv_unwrap as uv_module  # noqa: E402
from Blender_to_Spine2D_Mesh_Exporter.domain.geometry import ModifierLineagePolicy  # noqa: E402
from Blender_to_Spine2D_Mesh_Exporter.domain.uv import UvUnwrapSettings  # noqa: E402


TEMPORARY_PREFIX = "__Spine2D"


@dataclass(frozen=True)
class ContextSnapshot:
    active_object_name: str | None
    selected_object_names: tuple[str, ...]
    mode: str


@dataclass(frozen=True)
class DataSnapshot:
    object_names: tuple[str, ...]
    mesh_names: tuple[str, ...]
    collection_names: tuple[str, ...]
    material_names: tuple[str, ...]
    image_names: tuple[str, ...]


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _clear_scene() -> None:
    if bpy.context.object is not None and bpy.context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    for obj in tuple(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for collection in tuple(bpy.data.collections):
        if collection is not bpy.context.scene.collection:
            bpy.data.collections.remove(collection, do_unlink=True)
    for mesh in tuple(bpy.data.meshes):
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)
    for material in tuple(bpy.data.materials):
        if material.users == 0:
            bpy.data.materials.remove(material)
    for image in tuple(bpy.data.images):
        if image.users == 0:
            bpy.data.images.remove(image)


def _create_mesh_object(
    name: str,
    vertices: tuple[tuple[float, float, float], ...],
    faces: tuple[tuple[int, ...], ...],
    *,
    uv_layer_name: str = "UVMap",
):
    mesh = bpy.data.meshes.new(f"{name}_Mesh")
    mesh.from_pydata(vertices, (), faces)
    mesh.update(calc_edges=True)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)

    uv_layer = mesh.uv_layers.new(name=uv_layer_name)
    min_x = min(vertex[0] for vertex in vertices)
    max_x = max(vertex[0] for vertex in vertices)
    min_y = min(vertex[1] for vertex in vertices)
    max_y = max(vertex[1] for vertex in vertices)
    size_x = max(max_x - min_x, 1.0)
    size_y = max(max_y - min_y, 1.0)
    for polygon in mesh.polygons:
        for loop_index in polygon.loop_indices:
            vertex_index = mesh.loops[loop_index].vertex_index
            x_value, y_value, _ = vertices[vertex_index]
            uv_layer.data[loop_index].uv = (
                (x_value - min_x) / size_x,
                (y_value - min_y) / size_y,
            )
    mesh.uv_layers.active = uv_layer
    return obj


def _create_quad(name: str = "SourceQuad"):
    return _create_mesh_object(
        name,
        (
            (-1.0, -1.0, 0.0),
            (1.0, -1.0, 0.0),
            (1.0, 1.0, 0.0),
            (-1.0, 1.0, 0.0),
        ),
        ((0, 1, 2, 3),),
    )


def _create_offset_triangle(name: str = "MirrorTriangle"):
    return _create_mesh_object(
        name,
        (
            (1.0, -0.75, 0.0),
            (2.0, -0.75, 0.0),
            (1.5, 0.75, 0.0),
        ),
        ((0, 1, 2),),
    )


def _capture_context() -> ContextSnapshot:
    active = bpy.context.view_layer.objects.active
    return ContextSnapshot(
        active_object_name=None if active is None else active.name,
        selected_object_names=tuple(sorted(obj.name for obj in bpy.context.selected_objects)),
        mode=str(bpy.context.mode),
    )


def _capture_data() -> DataSnapshot:
    return DataSnapshot(
        object_names=tuple(sorted(item.name for item in bpy.data.objects)),
        mesh_names=tuple(sorted(item.name for item in bpy.data.meshes)),
        collection_names=tuple(sorted(item.name for item in bpy.data.collections)),
        material_names=tuple(sorted(item.name for item in bpy.data.materials)),
        image_names=tuple(sorted(item.name for item in bpy.data.images)),
    )


def _temporary_datablock_names() -> tuple[str, ...]:
    names: list[str] = []
    for collection in (
        bpy.data.objects,
        bpy.data.meshes,
        bpy.data.collections,
        bpy.data.materials,
        bpy.data.images,
    ):
        names.extend(item.name for item in collection if item.name.startswith(TEMPORARY_PREFIX))
    return tuple(sorted(names))


def _activate_only(obj) -> None:
    for candidate in bpy.context.scene.objects:
        candidate.select_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def _remove_modifiers(obj) -> None:
    for modifier in tuple(obj.modifiers):
        obj.modifiers.remove(modifier)


def test_source_reader_preserves_context_and_datablocks() -> None:
    _clear_scene()
    source = _create_quad()
    sentinel = _create_offset_triangle("Sentinel")
    _activate_only(sentinel)
    source.select_set(False)

    context_before = _capture_context()
    data_before = _capture_data()
    snapshot = read_source_mesh_snapshot(source)

    _assert(len(snapshot.vertices) == 4, "source reader returned wrong vertex count")
    _assert(len(snapshot.faces) == 1, "source reader returned wrong face count")
    _assert(snapshot.active_uv_layer == "UVMap", "active UV layer was not captured")
    _assert(_capture_context() == context_before, "source reader changed Blender context")
    _assert(_capture_data() == data_before, "source reader changed Blender datablocks")
    _assert(not _temporary_datablock_names(), "source reader leaked temporary datablocks")


def test_topology_preserving_modifier_strict_lineage() -> None:
    _clear_scene()
    source = _create_quad()
    modifier = source.modifiers.new(name="Smooth", type="SMOOTH")
    modifier.factor = 0.25
    modifier.iterations = 1
    data_before = _capture_data()

    result = read_evaluated_mesh_snapshot(
        source,
        lineage_policy=ModifierLineagePolicy.STRICT_PRESERVE,
    )

    _assert(result.lineage_report.valid, "strict topology-preserving lineage is invalid")
    _assert(len(result.snapshot.vertices) == len(source.data.vertices), "vertex count changed")
    _assert(len(result.snapshot.faces) == len(source.data.polygons), "face count changed")
    _assert(result.modifier_stack == (("Smooth", "SMOOTH"),), "modifier stack mismatch")
    _assert(_capture_data() == data_before, "evaluated reader changed persistent datablocks")
    _assert(not _temporary_datablock_names(), "evaluated reader leaked temporary datablocks")


def test_mirror_modifier_allows_exact_source_duplication() -> None:
    _clear_scene()
    source = _create_offset_triangle()
    modifier = source.modifiers.new(name="Mirror", type="MIRROR")
    modifier.use_axis[0] = True
    modifier.use_clip = False
    modifier.use_mirror_merge = False
    data_before = _capture_data()

    result = read_evaluated_mesh_snapshot(
        source,
        lineage_policy=ModifierLineagePolicy.ALLOW_SOURCE_DUPLICATION,
    )

    _assert(result.lineage_report.valid, "Mirror lineage was not preserved")
    _assert(len(result.snapshot.faces) == 2, "Mirror did not produce two faces")
    _assert(
        result.lineage_report.faces.duplicated_source_indices == (0,),
        "Mirror face duplication was not reported",
    )
    _assert(
        result.lineage_report.corners.duplicated_source_indices == (0, 1, 2),
        "Mirror corner duplication was not reported",
    )
    _assert(_capture_data() == data_before, "Mirror evaluation changed persistent datablocks")
    _assert(not _temporary_datablock_names(), "Mirror evaluation leaked temporary datablocks")


def test_rejected_topology_change_cleans_all_temporary_data() -> None:
    _clear_scene()
    source = _create_quad()
    modifier = source.modifiers.new(name="Solidify", type="SOLIDIFY")
    modifier.thickness = 0.2
    data_before = _capture_data()

    try:
        read_evaluated_mesh_snapshot(
            source,
            lineage_policy=ModifierLineagePolicy.STRICT_PRESERVE,
        )
    except EvaluatedMeshReadError:
        pass
    else:
        raise AssertionError("Solidify unexpectedly passed STRICT_PRESERVE")

    _assert(_capture_data() == data_before, "failed evaluation changed persistent datablocks")
    _assert(not _temporary_datablock_names(), "failed evaluation leaked temporary datablocks")


def test_uv_unwrap_is_global_transaction_and_restores_context() -> None:
    _clear_scene()
    source = _create_quad()
    sentinel = _create_offset_triangle("Sentinel")
    _activate_only(sentinel)
    source.select_set(False)
    snapshot = read_source_mesh_snapshot(source)
    context_before = _capture_context()
    data_before = _capture_data()

    result = unwrap_snapshot_uv(snapshot, UvUnwrapSettings(layer_name="SpineBakeUV"))

    _assert(result.statistics.loop_count == len(snapshot.loops), "UV loop count mismatch")
    _assert(result.statistics.outside_unit_square_count == 0, "packed UV left unit square")
    _assert(result.snapshot.active_uv_layer == "SpineBakeUV", "result UV layer inactive")
    _assert(source.data.uv_layers.get("SpineBakeUV") is None, "source mesh was modified")
    _assert(_capture_context() == context_before, "UV transaction did not restore context")
    _assert(_capture_data() == data_before, "UV transaction changed persistent datablocks")
    _assert(not _temporary_datablock_names(), "UV transaction leaked temporary datablocks")


def test_uv_failure_in_edit_mode_restores_context_and_cleans_data() -> None:
    _clear_scene()
    source = _create_quad()
    sentinel = _create_offset_triangle("Sentinel")
    _activate_only(sentinel)
    snapshot = read_source_mesh_snapshot(source)
    context_before = _capture_context()
    data_before = _capture_data()
    original_call_operator = uv_module._call_operator

    def fail_during_pack(operator, operator_name, arguments):
        if operator_name == "pack_islands":
            raise UvUnwrapError("forced pack failure")
        return original_call_operator(operator, operator_name, arguments)

    with mock.patch.object(uv_module, "_call_operator", side_effect=fail_during_pack):
        try:
            unwrap_snapshot_uv(snapshot, UvUnwrapSettings(layer_name="SpineBakeUV"))
        except UvUnwrapError as exc:
            _assert("forced pack failure" in str(exc), "unexpected forced UV failure")
        else:
            raise AssertionError("forced UV failure did not propagate")

    _assert(_capture_context() == context_before, "failed UV transaction did not restore context")
    _assert(_capture_data() == data_before, "failed UV transaction changed persistent datablocks")
    _assert(not _temporary_datablock_names(), "failed UV transaction leaked temporary data")


def main() -> None:
    tests = (
        test_source_reader_preserves_context_and_datablocks,
        test_topology_preserving_modifier_strict_lineage,
        test_mirror_modifier_allows_exact_source_duplication,
        test_rejected_topology_change_cleans_all_temporary_data,
        test_uv_unwrap_is_global_transaction_and_restores_context,
        test_uv_failure_in_edit_mode_restores_context_and_cleans_data,
    )
    print(f"Blender version: {bpy.app.version_string}")
    for test in tests:
        print(f"[HEADLESS] RUN {test.__name__}")
        test()
        print(f"[HEADLESS] PASS {test.__name__}")
    print(f"[HEADLESS] PASS {len(tests)} integration tests")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
