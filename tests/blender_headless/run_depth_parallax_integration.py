"""Real Blender 5.2 smoke acceptance for Depth parallax reserve 0.90.0.

The fixture contains a front quad and a forty-five-degree flap folded behind its right
edge. The active camera sees only the front surface. A fifty-degree Parallax Horizon
Angle must retain the hidden flap, render it from one fitted virtual camera, and export
one reserve mesh attachment below the established front attachment without mutating the
source Scene, camera, materials, selection, or temporary Blender datablocks.
"""

from __future__ import annotations

import json
from math import radians
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
    A1SingleObjectExportSettings,
    A1SourceGeometryMode,
    ExportSettings,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter import (  # noqa: E402
    export_a1_single_object,
    prepare_a1_object,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.a1_object_preparation import (  # noqa: E402
    PreparedDepthA1Object,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import (  # noqa: E402
    A1TextureExportMode,
    BakeExecutionSettings,
    CameraProjectionPlan,
    DepthCameraProjectionSettings,
    DepthParallaxSettings,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.projection import (  # noqa: E402
    A1ProjectionDirection,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine import (  # noqa: E402
    A1RigProfile,
    MeshAttachment,
    SpineJsonTarget,
    decode_weighted_vertices,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.uv import (  # noqa: E402
    UvUnwrapSettings,
)
from run_bake_integration import (  # noqa: E402
    PNG_SIGNATURE,
    _activate_only,
    _assert,
    _capture_context,
    _capture_scene_bake_state,
    _clear_scene,
    _create_mesh_object,
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


_TEXTURE_SIZE = 128
_MAX_DEPTH_POINTS = 24
_HORIZON_ANGLE_RADIANS = radians(50.0)
_TARGET = SpineJsonTarget.SPINE_4_2
_PREFIX = "DepthParallaxSmoke"


def _create_emission_material(
    name: str,
    color: tuple[float, float, float, float],
):
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new(type="ShaderNodeOutputMaterial")
    emission = nodes.new(type="ShaderNodeEmission")
    emission.inputs["Color"].default_value = color
    emission.inputs["Strength"].default_value = 1.8
    material.node_tree.links.new(
        emission.outputs["Emission"],
        output.inputs["Surface"],
    )
    return material


def _create_folded_surface(name: str):
    """Create two front triangles and two fully occluded forty-five-degree triangles."""

    source = _create_mesh_object(
        name,
        (
            (-0.90, -0.90, 0.00),
            (0.90, -0.90, 0.00),
            (0.90, 0.90, 0.00),
            (-0.90, 0.90, 0.00),
            (0.25, -0.90, -0.65),
            (0.25, 0.90, -0.65),
        ),
        (
            (0, 1, 2),
            (0, 2, 3),
            (1, 4, 5),
            (1, 5, 2),
        ),
    )
    front_material = _create_emission_material(
        f"{name}_FrontMaterial",
        (1.0, 0.02, 0.01, 1.0),
    )
    reserve_material = _create_emission_material(
        f"{name}_ReserveMaterial",
        (0.01, 1.0, 0.04, 1.0),
    )
    source.data.materials.append(front_material)
    source.data.materials.append(reserve_material)
    for polygon in source.data.polygons:
        polygon.material_index = 0 if polygon.index < 2 else 1
    return source, front_material, reserve_material


def _settings(output_directory: Path) -> A1SingleObjectExportSettings:
    return A1SingleObjectExportSettings(
        export=ExportSettings(
            texture_width=_TEXTURE_SIZE,
            texture_height=_TEXTURE_SIZE,
            output_directory=output_directory,
            images_relative_path="images",
            spine_version=_TARGET.exact_version,
            rig_profile=A1RigProfile.TWO_AXIS_ROTATION_SCALE.value,
            bake_margin=1,
        ),
        prefix=_PREFIX,
        output_stem=_PREFIX,
        json_output_stem=_PREFIX,
        source_geometry_mode=A1SourceGeometryMode.EVALUATED,
        projection_direction=A1ProjectionDirection.ACTIVE_CAMERA,
        uv=UvUnwrapSettings(layer_name="SpineBakeUV"),
        bake_execution=BakeExecutionSettings(
            samples=1,
            texture_export_mode=A1TextureExportMode.DEPTH_CAMERA_PROJECTION,
            depth_projection=DepthCameraProjectionSettings(
                smoothing=0.0,
                edge_threshold_fraction=1.0,
                mesh_error_pixels=4.0,
                max_points=_MAX_DEPTH_POINTS,
            ),
            depth_parallax=DepthParallaxSettings(
                horizon_angle_radians=_HORIZON_ANGLE_RADIANS,
            ),
        ),
    )


def _camera_fingerprint(camera: object) -> tuple[object, ...]:
    return (
        tuple(float(value) for row in camera.matrix_world for value in row),
        str(camera.data.type),
        float(camera.data.lens),
        float(camera.data.ortho_scale),
        float(camera.data.clip_start),
        float(camera.data.clip_end),
        bpy.context.scene.camera.name if bpy.context.scene.camera is not None else None,
    )


def _prepare_scene() -> tuple[object, object, object, object]:
    _clear_scene()
    _purge_orphan_scene_data()
    _configure_scene()
    bpy.context.scene.cycles.samples = 1
    camera = _create_camera(name="DepthParallaxCamera")
    source, front_material, reserve_material = _create_folded_surface(
        f"{_PREFIX}_Source"
    )
    sentinel = _create_sentinel()
    sentinel.location = (8.0, 0.0, 0.0)
    _activate_only(sentinel)
    source.select_set(False)
    bpy.context.scene.frame_set(1)
    bpy.context.view_layer.update()
    return source, front_material, reserve_material, camera


def _single_influence_bones(attachment: MeshAttachment) -> set[int]:
    weighted = decode_weighted_vertices(attachment.vertices)
    _assert(
        len(weighted) == len(attachment.uvs) // 2,
        "weighted vertex count differs from UV count",
    )
    _assert(
        all(len(vertex.influences) == 1 for vertex in weighted),
        "every parallax point must use exactly one generated vertex bone",
    )
    return {vertex.influences[0].bone_index for vertex in weighted}


def _assert_prepared(prepared: object) -> tuple[str, str, dict[str, tuple[int, ...]]]:
    _assert(
        isinstance(prepared, PreparedDepthA1Object),
        f"Depth parallax did not return PreparedDepthA1Object: {type(prepared)!r}",
    )
    package = prepared.depth_parallax_package
    _assert(
        abs(package.horizon_angle_radians - _HORIZON_ANGLE_RADIANS) <= 1.0e-12,
        "prepared horizon angle changed",
    )
    _assert(
        package.front_face_indices == (0, 1),
        f"unexpected front source faces: {package.front_face_indices}",
    )
    _assert(
        package.reserve_face_indices == (2, 3),
        f"hidden folded faces were not retained: {package.reserve_face_indices}",
    )
    _assert(len(package.reserve_surfaces) == 1, "expected one reserve view surface")
    surface = package.reserve_surfaces[0]
    _assert(
        surface.source_face_indices == (2, 3),
        f"reserve view owns wrong faces: {surface.source_face_indices}",
    )
    _assert(len(package.union_snapshot.vertices) == 6, "union point count changed")
    _assert(len(package.union_snapshot.faces) == 4, "union face count changed")
    _assert(len(package.front_snapshot.faces) == 2, "front subset face count changed")
    _assert(len(surface.snapshot.faces) == 2, "reserve subset face count changed")
    _assert(
        prepared.source_snapshot == package.union_snapshot,
        "prepared source is not the shared parallax union",
    )

    _assert(len(prepared.reserve_bake_plans) == 1, "expected one reserve camera plan")
    reserve_plan = prepared.reserve_bake_plans[0]
    _assert(isinstance(prepared.bake_plan, CameraProjectionPlan), "front plan is not camera")
    _assert(isinstance(reserve_plan, CameraProjectionPlan), "reserve plan is not camera")
    _assert(prepared.bake_plan.view_id == "FRONT", "front plan lost FRONT view id")
    _assert(reserve_plan.virtual_view, "reserve plan is not marked virtual")
    _assert(
        reserve_plan.view_id == surface.view.view_id.value,
        "reserve plan and reserve surface view ids differ",
    )
    _assert(
        reserve_plan.camera_world_matrix_override is not None,
        "reserve plan has no virtual camera matrix",
    )
    _assert(
        0.0 < reserve_plan.lens_scale <= 1.0,
        f"invalid fitted reserve lens scale: {reserve_plan.lens_scale}",
    )

    assembly = prepared.document_assembly
    _assert(len(assembly.projections) == 2, "expected reserve plus front projection")
    _assert(len(assembly.document_build.components) == 2, "expected two components")
    reserve_slot = f"{_PREFIX}_Parallax_{reserve_plan.view_id}"
    front_slot = f"{_PREFIX}_Segment_0"
    slot_order = tuple(
        projection.request.slot_name for projection in assembly.projections
    )
    _assert(
        slot_order == (reserve_slot, front_slot),
        f"reserve/front projection order changed: {slot_order}",
    )

    attachments: dict[str, MeshAttachment] = {}
    triangles: dict[str, tuple[int, ...]] = {}
    for component in assembly.document_build.components:
        attachment = component.attachment
        _assert(
            isinstance(attachment, MeshAttachment),
            f"component {component.request.slot_name} is not a mesh attachment",
        )
        attachments[component.request.slot_name] = attachment
        triangles[component.request.slot_name] = tuple(
            int(value) for value in attachment.triangles
        )

    _assert(set(attachments) == {reserve_slot, front_slot}, "component slots differ")
    reserve_bones = _single_influence_bones(attachments[reserve_slot])
    front_bones = _single_influence_bones(attachments[front_slot])
    _assert(
        len(reserve_bones & front_bones) >= 2,
        "front and reserve attachments do not share hinge vertex bones",
    )

    expected_statistics = {
        "depth_parallax_enabled": 1,
        "depth_parallax_reserve_source_face_count": 2,
        "depth_parallax_reserve_attachment_count": 1,
        "depth_parallax_attachment_count": 2,
        "depth_camera_single_attachment": 0,
        "depth_camera_expected_attachment_count": 2,
    }
    for key, expected in expected_statistics.items():
        _assert(
            int(prepared.statistics.get(key, -1)) == expected,
            f"prepared statistic {key!r} differs: {prepared.statistics.get(key)!r}",
        )
    return reserve_slot, front_slot, triangles


def _dominant_color_counts(pixels: tuple[float, ...]) -> tuple[int, int]:
    red = green = 0
    for offset in range(0, len(pixels), 4):
        r, g, b, alpha = pixels[offset : offset + 4]
        if alpha <= 0.08:
            continue
        if r > g * 1.35 and r > b * 1.35:
            red += 1
        if g > r * 1.35 and g > b * 1.35:
            green += 1
    return red, green


def _attachment_group(
    document: dict[str, object],
    slot_name: str,
) -> dict[str, object]:
    skins = document.get("skins")
    _assert(isinstance(skins, list), "serialized skins must be an array")
    matches = []
    for skin in skins:
        if not isinstance(skin, dict):
            continue
        groups = skin.get("attachments")
        if not isinstance(groups, dict):
            continue
        group = groups.get(slot_name)
        if isinstance(group, dict):
            matches.append(group)
    _assert(len(matches) == 1, f"expected one attachment group for {slot_name}")
    return matches[0]


def _assert_serialized_attachment(
    document: dict[str, object],
    slot_name: str,
    image_size: tuple[int, int],
    expected_triangles: tuple[int, ...],
) -> None:
    group = _attachment_group(document, slot_name)
    _assert(slot_name in group, f"setup attachment {slot_name} is missing")
    _assert(len(group) == 1, f"unexpected attachment variants for {slot_name}")
    attachment = group[slot_name]
    _assert(isinstance(attachment, dict), f"attachment {slot_name} is not an object")
    _assert(attachment.get("type") == "mesh", f"attachment {slot_name} is not mesh")
    uvs = attachment.get("uvs")
    triangles = attachment.get("triangles")
    vertices = attachment.get("vertices")
    _assert(isinstance(uvs, list) and len(uvs) >= 6, f"invalid UVs for {slot_name}")
    _assert(len(uvs) % 2 == 0, f"odd UV stream for {slot_name}")
    _assert(
        all(0.0 <= float(value) <= 1.0 for value in uvs),
        f"cropped UV outside 0..1 for {slot_name}",
    )
    _assert(isinstance(triangles, list), f"triangles missing for {slot_name}")
    _assert(
        tuple(int(value) for value in triangles) == expected_triangles,
        f"crop changed triangles for {slot_name}",
    )
    vertex_count = len(uvs) // 2
    _assert(
        all(0 <= int(index) < vertex_count for index in triangles),
        f"triangles reference missing vertices for {slot_name}",
    )
    _assert(
        isinstance(vertices, list) and len(vertices) > len(uvs),
        f"attachment {slot_name} is not weighted",
    )
    _assert(
        int(round(float(attachment.get("width", 0.0)))) == image_size[0],
        f"attachment width differs from PNG for {slot_name}",
    )
    _assert(
        int(round(float(attachment.get("height", 0.0)))) == image_size[1],
        f"attachment height differs from PNG for {slot_name}",
    )


def _run_smoke(output_root: Path) -> None:
    output_directory = output_root / "positive-perspective"
    output_directory.mkdir(parents=True, exist_ok=False)
    source, front_material, reserve_material, camera = _prepare_scene()
    settings = _settings(output_directory)

    context_before = _capture_context()
    bake_before = _capture_scene_bake_state()
    render_before = _scene_render_fingerprint()
    camera_before = _camera_fingerprint(camera)
    front_material_before = _material_fingerprint(front_material)
    reserve_material_before = _material_fingerprint(reserve_material)
    temporary_before = _temporary_datablock_names()

    prepared = prepare_a1_object(
        source,
        settings,
        context=bpy.context,
        scene=bpy.context.scene,
    )
    reserve_slot, front_slot, prepared_triangles = _assert_prepared(prepared)
    _assert(_capture_context() == context_before, "prepare changed Blender context")
    _assert(_capture_scene_bake_state() == bake_before, "prepare changed bake state")
    _assert(_scene_render_fingerprint() == render_before, "prepare changed render state")
    _assert(_camera_fingerprint(camera) == camera_before, "prepare changed camera")
    _assert(_temporary_datablock_names() == temporary_before, "prepare leaked datablocks")

    result = export_a1_single_object(
        source,
        settings,
        context=bpy.context,
        scene=bpy.context.scene,
    )
    _assert(result.success, f"Depth parallax export failed: {result.issues}")

    expected_paths = (
        prepared.output_paths.json_path,
        prepared.bake_plan.frame_tasks[0].output_path,
        prepared.reserve_bake_plans[0].frame_tasks[0].output_path,
    )
    _assert(
        tuple(path.resolve(strict=False) for path in result.output_files)
        == tuple(path.resolve(strict=False) for path in expected_paths),
        f"unexpected output order: {result.output_files}",
    )
    json_path, front_path, reserve_path = result.output_files
    _assert(json_path.suffix.lower() == ".json", "JSON output must be first")
    for path in (front_path, reserve_path):
        _assert(path.read_bytes().startswith(PNG_SIGNATURE), f"invalid PNG: {path}")

    front_size, front_pixels = _read_image(front_path)
    reserve_size, reserve_pixels = _read_image(reserve_path)
    for label, size, pixels in (
        ("front", front_size, front_pixels),
        ("reserve", reserve_size, reserve_pixels),
    ):
        _assert(
            1 <= size[0] <= _TEXTURE_SIZE and 1 <= size[1] <= _TEXTURE_SIZE,
            f"invalid {label} crop size: {size}",
        )
        visible, transparent = _visible_and_transparent_counts(pixels)
        _assert(visible > 100, f"{label} view has too few visible pixels: {visible}")
        _assert(transparent > 0, f"{label} crop has no transparent padding")

    front_red, front_green = _dominant_color_counts(front_pixels)
    reserve_red, reserve_green = _dominant_color_counts(reserve_pixels)
    _assert(front_red > 100, f"front texture lost front material: {front_red}")
    _assert(
        front_green < max(8, front_red // 50),
        f"hidden flap leaked materially into front texture: green={front_green}",
    )
    _assert(
        reserve_green > 20,
        f"reserve texture does not reveal folded green surface: {reserve_green}",
    )
    _assert(
        front_size != reserve_size or front_pixels != reserve_pixels,
        "front and reserve camera renders are identical",
    )
    _assert(reserve_red > 0, "reserve render unexpectedly lost the front surface entirely")

    document = json.loads(json_path.read_text(encoding="utf-8"))
    _assert(isinstance(document, dict), "serialized Spine document must be an object")
    skeleton = document.get("skeleton")
    _assert(isinstance(skeleton, dict), "skeleton metadata missing")
    _assert(skeleton.get("spine") == _TARGET.exact_version, "wrong Spine target")
    slots = document.get("slots")
    _assert(isinstance(slots, list), "serialized slots must be an array")
    visual_slots = tuple(
        str(slot.get("name"))
        for slot in slots
        if isinstance(slot, dict)
        and str(slot.get("name", "")).startswith(_PREFIX)
    )
    _assert(
        visual_slots == (reserve_slot, front_slot),
        f"serialized reserve/front draw order changed: {visual_slots}",
    )
    _assert_serialized_attachment(
        document,
        reserve_slot,
        reserve_size,
        prepared_triangles[reserve_slot],
    )
    _assert_serialized_attachment(
        document,
        front_slot,
        front_size,
        prepared_triangles[front_slot],
    )

    expected_statistics = {
        "depth_parallax_cropped_view_count": 2,
        "parallax_texture_view_count": 2,
        "parallax_texture_output_count": 2,
        "output_file_count": 3,
    }
    for key, expected in expected_statistics.items():
        _assert(
            int(result.statistics.get(key, -1)) == expected,
            f"result statistic {key!r} differs: {result.statistics.get(key)!r}",
        )

    _assert(_capture_context() == context_before, "export changed Blender context")
    _assert(_capture_scene_bake_state() == bake_before, "export changed bake state")
    _assert(_scene_render_fingerprint() == render_before, "export changed render state")
    _assert(_camera_fingerprint(camera) == camera_before, "export changed camera")
    _assert(
        _material_fingerprint(front_material) == front_material_before,
        "export changed front material",
    )
    _assert(
        _material_fingerprint(reserve_material) == reserve_material_before,
        "export changed reserve material",
    )
    _assert(_temporary_datablock_names() == temporary_before, "export leaked datablocks")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="spine2d-depth-parallax-") as directory:
        _run_smoke(Path(directory))
    print(
        "[DEPTH-PARALLAX] PASS cases=1 target=4.2 camera=PERSP "
        "angle=50deg attachments=2 textures=2 camera_zero=shared"
    )


if __name__ == "__main__":
    main()
