"""Real Cycles and complete A1 export checks executed by Blender 5.2.

All fixtures are created at runtime. The suite verifies the standalone bake execution
path, then the complete single-object service including one atomic commit shared by
the baked PNG and the final Spine JSON.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys
import tempfile
import traceback
from unittest import mock

import bpy


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from Blender_to_Spine2D_Mesh_Exporter.application import (  # noqa: E402
    A1SingleObjectExportSettings,
    A1SingleObjectStage,
    A1SourceGeometryMode,
    ExportSettings,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter import (  # noqa: E402
    BakeExecutionError,
    analyse_object_materials,
    execute_bake_plan,
    export_a1_single_object,
    read_source_mesh_snapshot,
    unwrap_snapshot_uv,
)
import Blender_to_Spine2D_Mesh_Exporter.blender_adapter.semantic_bake_execution as bake_module  # noqa: E402
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import (  # noqa: E402
    BakeExecutionSettings,
    BakeMode,
    BakeSettings,
    TextureFormat,
    build_bake_plan,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.uv import UvUnwrapSettings  # noqa: E402


TEMPORARY_PREFIX = "__Spine2D"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


@dataclass(frozen=True)
class ContextSnapshot:
    active_object_name: str | None
    selected_object_names: tuple[str, ...]
    mode: str


@dataclass(frozen=True)
class SceneBakeSnapshot:
    frame_current: int
    render_engine: str
    file_format: str
    color_mode: str
    bake_margin: int
    bake_use_clear: bool
    bake_selected_to_active: bool
    bake_use_cage: bool
    cage_extrusion: float
    cycles_bake_type: str
    cycles_samples: int


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


def _configure_cycles_scene() -> None:
    """Configure the factory-startup Scene for the explicit Cycles test contract."""

    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 1
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "Medium High Contrast"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0


def _create_mesh_object(
    name: str,
    vertices: tuple[tuple[float, float, float], ...],
    faces: tuple[tuple[int, ...], ...],
):
    mesh = bpy.data.meshes.new(f"{name}_Mesh")
    mesh.from_pydata(vertices, (), faces)
    mesh.update(calc_edges=True)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)

    uv_layer = mesh.uv_layers.new(name="UVMap")
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


def _create_quad(name: str = "BakeSource"):
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


def _create_sentinel():
    return _create_mesh_object(
        "Sentinel",
        ((3.0, 0.0, 0.0), (4.0, 0.0, 0.0), (3.5, 1.0, 0.0)),
        ((0, 1, 2),),
    )


def _create_emission_material(obj):
    material = bpy.data.materials.new(name="SourceEmission")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new(type="ShaderNodeOutputMaterial")
    emission = nodes.new(type="ShaderNodeEmission")
    emission.inputs["Color"].default_value = (0.8, 0.15, 0.05, 1.0)
    emission.inputs["Strength"].default_value = 1.0
    material.node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    obj.data.materials.append(material)
    return material


def _activate_only(obj) -> None:
    for candidate in bpy.context.scene.objects:
        candidate.select_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def _capture_context() -> ContextSnapshot:
    active = bpy.context.view_layer.objects.active
    return ContextSnapshot(
        active_object_name=None if active is None else active.name,
        selected_object_names=tuple(sorted(obj.name for obj in bpy.context.selected_objects)),
        mode=str(bpy.context.mode),
    )


def _capture_scene_bake_state() -> SceneBakeSnapshot:
    scene = bpy.context.scene
    return SceneBakeSnapshot(
        frame_current=int(scene.frame_current),
        render_engine=str(scene.render.engine),
        file_format=str(scene.render.image_settings.file_format),
        color_mode=str(scene.render.image_settings.color_mode),
        bake_margin=int(scene.render.bake.margin),
        bake_use_clear=bool(scene.render.bake.use_clear),
        bake_selected_to_active=bool(scene.render.bake.use_selected_to_active),
        bake_use_cage=bool(scene.render.bake.use_cage),
        cage_extrusion=float(scene.render.bake.cage_extrusion),
        cycles_bake_type=str(scene.cycles.bake_type),
        cycles_samples=int(scene.cycles.samples),
    )


def _material_fingerprint(material) -> tuple[object, ...]:
    nodes = tuple(
        sorted(
            (
                node.name,
                node.bl_idname,
                bool(node.select),
            )
            for node in material.node_tree.nodes
        )
    )
    links = tuple(
        sorted(
            (
                link.from_node.name,
                link.from_socket.name,
                link.to_node.name,
                link.to_socket.name,
            )
            for link in material.node_tree.links
        )
    )
    return (material.name, bool(material.use_nodes), nodes, links)


def _temporary_datablock_names() -> tuple[str, ...]:
    names: list[str] = []
    for collection in (
        bpy.data.objects,
        bpy.data.meshes,
        bpy.data.collections,
        bpy.data.materials,
        bpy.data.images,
    ):
        names.extend(
            item.name for item in collection if item.name.startswith(TEMPORARY_PREFIX)
        )
    return tuple(sorted(names))


def _build_fixture(output_directory: Path):
    source = _create_quad()
    source_material = _create_emission_material(source)
    sentinel = _create_sentinel()
    _activate_only(sentinel)
    source.select_set(False)

    source_snapshot = read_source_mesh_snapshot(source)
    unwrap_result = unwrap_snapshot_uv(
        source_snapshot,
        UvUnwrapSettings(layer_name="SpineBakeUV"),
    )
    analysis = analyse_object_materials(
        source,
        render_target="CYCLES",
        source_object_id=source_snapshot.source_object_id,
    )
    plan = build_bake_plan(
        analysis,
        BakeSettings(
            width=16,
            height=16,
            output_directory=output_directory,
            output_stem="HeadlessEmission",
            uv_layer_name="SpineBakeUV",
            texture_format=TextureFormat.PNG,
            margin_pixels=1,
            diffuse_mode=BakeMode.EMIT,
            procedural_mode=BakeMode.EMIT,
        ),
    )
    return source, source_material, sentinel, unwrap_result.snapshot, plan


def _build_service_settings(output_directory: Path) -> A1SingleObjectExportSettings:
    return A1SingleObjectExportSettings(
        export=ExportSettings(
            texture_width=32,
            texture_height=32,
            output_directory=output_directory,
            images_relative_path="images",
            bake_margin=1,
        ),
        prefix="EndToEnd",
        output_stem="EndToEnd",
        source_geometry_mode=A1SourceGeometryMode.EVALUATED,
        uv=UvUnwrapSettings(layer_name="SpineBakeUV"),
        diffuse_mode=BakeMode.EMIT,
        procedural_mode=BakeMode.EMIT,
        bake_execution=BakeExecutionSettings(samples=1),
    )


def test_real_cycles_emit_bake_commits_png_and_restores_state() -> None:
    _clear_scene()
    _configure_cycles_scene()
    with tempfile.TemporaryDirectory(prefix="spine2d-bake-success-") as directory:
        output_directory = Path(directory)
        source, material, sentinel, target_snapshot, plan = _build_fixture(
            output_directory
        )
        _activate_only(sentinel)
        source.select_set(False)
        context_before = _capture_context()
        scene_before = _capture_scene_bake_state()
        material_before = _material_fingerprint(material)

        result = execute_bake_plan(
            source,
            target_snapshot,
            plan,
            BakeExecutionSettings(samples=1),
        )

        _assert(len(result.artifacts) == 1, "expected exactly one bake artifact")
        output_path = result.artifacts[0].output_path
        _assert(output_path.is_file(), "bake output PNG does not exist")
        _assert(output_path.stat().st_size > 8, "bake output PNG is empty")
        _assert(output_path.read_bytes()[:8] == PNG_SIGNATURE, "bake output is not PNG")
        _assert(_capture_context() == context_before, "successful bake changed context")
        _assert(
            _capture_scene_bake_state() == scene_before,
            "successful bake changed scene",
        )
        _assert(
            _material_fingerprint(material) == material_before,
            "source material mutated",
        )
        _assert(not _temporary_datablock_names(), "successful bake leaked temporary data")


def test_forced_bake_failure_rolls_back_file_and_restores_state() -> None:
    _clear_scene()
    _configure_cycles_scene()
    with tempfile.TemporaryDirectory(prefix="spine2d-bake-failure-") as directory:
        output_directory = Path(directory)
        source, material, sentinel, target_snapshot, plan = _build_fixture(
            output_directory
        )
        final_path = plan.representative_task.output_path
        final_path.parent.mkdir(parents=True, exist_ok=True)
        previous_content = b"previous-production-output"
        final_path.write_bytes(previous_content)
        _activate_only(sentinel)
        source.select_set(False)
        context_before = _capture_context()
        scene_before = _capture_scene_bake_state()
        material_before = _material_fingerprint(material)

        with mock.patch.object(
            bake_module,
            "_call_bake_operator",
            side_effect=BakeExecutionError("forced Cycles failure"),
        ):
            try:
                execute_bake_plan(
                    source,
                    target_snapshot,
                    plan,
                    BakeExecutionSettings(samples=1),
                )
            except BakeExecutionError as exc:
                _assert(
                    "forced Cycles failure" in str(exc),
                    "primary bake error was hidden",
                )
            else:
                raise AssertionError("forced bake failure did not propagate")

        _assert(
            final_path.read_bytes() == previous_content,
            "existing output was corrupted",
        )
        _assert(
            tuple(sorted(path.name for path in output_directory.iterdir()))
            == (final_path.name,),
            "rollback left staged or backup files",
        )
        _assert(_capture_context() == context_before, "failed bake changed context")
        _assert(_capture_scene_bake_state() == scene_before, "failed bake changed scene")
        _assert(
            _material_fingerprint(material) == material_before,
            "failed bake mutated material",
        )
        _assert(not _temporary_datablock_names(), "failed bake leaked temporary data")


def test_complete_a1_service_commits_valid_png_and_spine_json() -> None:
    _clear_scene()
    _configure_cycles_scene()
    with tempfile.TemporaryDirectory(prefix="spine2d-service-success-") as directory:
        output_directory = Path(directory)
        source = _create_quad("ServiceSource")
        material = _create_emission_material(source)
        sentinel = _create_sentinel()
        _activate_only(sentinel)
        source.select_set(False)
        context_before = _capture_context()
        scene_before = _capture_scene_bake_state()
        material_before = _material_fingerprint(material)

        result = export_a1_single_object(
            source,
            _build_service_settings(output_directory),
        )

        _assert(result.success, f"single-object service failed: {result.issues}")
        expected_json = output_directory / "EndToEnd.json"
        expected_png = output_directory / "images" / "EndToEnd_Baked.png"
        _assert(
            result.output_files == (expected_json.resolve(), expected_png.resolve()),
            f"unexpected service outputs: {result.output_files}",
        )
        _assert(expected_png.read_bytes()[:8] == PNG_SIGNATURE, "service PNG invalid")
        document = json.loads(expected_json.read_text(encoding="utf-8"))
        bone_names = tuple(item["name"] for item in document["bones"])
        _assert(bone_names[0] == "root", "A1 root bone missing")
        _assert("EndToEnd_main" in bone_names, "A1 main bone missing")
        _assert("EndToEnd_rotate_X" in bone_names, "A1 rotation bone missing")
        _assert(len(document["slots"]) == 1, "expected one region slot")
        slot_name = document["slots"][0]["name"]
        _assert(slot_name == "EndToEnd_Segment_0", "unexpected segment slot name")
        attachment = document["skins"][0]["attachments"][slot_name][slot_name]
        _assert(attachment["type"] == "mesh", "attachment is not a mesh")
        _assert(
            attachment["path"] == "images/EndToEnd_Baked",
            "attachment path does not match committed PNG",
        )
        _assert(len(attachment["uvs"]) == 8, "quad attachment UV count invalid")
        _assert(len(attachment["triangles"]) == 6, "quad triangulation invalid")
        _assert(_capture_context() == context_before, "service changed context")
        _assert(_capture_scene_bake_state() == scene_before, "service changed scene")
        _assert(
            _material_fingerprint(material) == material_before,
            "service mutated source material",
        )
        _assert(not _temporary_datablock_names(), "service leaked temporary data")


def test_complete_a1_service_rolls_back_png_and_json_together() -> None:
    _clear_scene()
    _configure_cycles_scene()
    with tempfile.TemporaryDirectory(prefix="spine2d-service-failure-") as directory:
        output_directory = Path(directory)
        source = _create_quad("ServiceSource")
        material = _create_emission_material(source)
        sentinel = _create_sentinel()
        settings = _build_service_settings(output_directory)
        final_json = output_directory / "EndToEnd.json"
        final_png = output_directory / "images" / "EndToEnd_Baked.png"
        final_json.parent.mkdir(parents=True, exist_ok=True)
        final_png.parent.mkdir(parents=True, exist_ok=True)
        old_json = b"old-json-output"
        old_png = b"old-png-output"
        final_json.write_bytes(old_json)
        final_png.write_bytes(old_png)
        _activate_only(sentinel)
        source.select_set(False)
        context_before = _capture_context()
        scene_before = _capture_scene_bake_state()
        material_before = _material_fingerprint(material)

        with mock.patch.object(
            bake_module,
            "_call_bake_operator",
            side_effect=BakeExecutionError("forced service bake failure"),
        ):
            result = export_a1_single_object(source, settings)

        _assert(not result.success, "forced service failure returned success")
        _assert(len(result.issues) == 1, "failure should contain one primary issue")
        issue = result.issues[0]
        _assert(
            issue.stage == A1SingleObjectStage.STAGE_OUTPUTS.value,
            f"unexpected failure stage: {issue.stage}",
        )
        _assert(
            issue.code == A1SingleObjectStage.STAGE_OUTPUTS.error_code,
            f"unexpected failure code: {issue.code}",
        )
        _assert(final_json.read_bytes() == old_json, "existing JSON was corrupted")
        _assert(final_png.read_bytes() == old_png, "existing PNG was corrupted")
        leftovers = tuple(
            sorted(
                str(path.relative_to(output_directory))
                for path in output_directory.rglob("*")
                if path.is_file()
            )
        )
        _assert(
            leftovers == ("EndToEnd.json", "images/EndToEnd_Baked.png"),
            f"joint rollback left staged or backup files: {leftovers}",
        )
        _assert(_capture_context() == context_before, "failed service changed context")
        _assert(_capture_scene_bake_state() == scene_before, "failed service changed scene")
        _assert(
            _material_fingerprint(material) == material_before,
            "failed service mutated material",
        )
        _assert(not _temporary_datablock_names(), "failed service leaked temporary data")


def main() -> None:
    tests = (
        test_real_cycles_emit_bake_commits_png_and_restores_state,
        test_forced_bake_failure_rolls_back_file_and_restores_state,
        test_complete_a1_service_commits_valid_png_and_spine_json,
        test_complete_a1_service_rolls_back_png_and_json_together,
    )
    print(f"Blender version: {bpy.app.version_string}")
    for test in tests:
        print(f"[BAKE] RUN {test.__name__}")
        test()
        print(f"[BAKE] PASS {test.__name__}")
    print(f"[BAKE] PASS {len(tests)} integration tests")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
