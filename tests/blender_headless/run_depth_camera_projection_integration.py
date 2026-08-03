"""Real Blender 5.2 acceptance for the Depth Camera Projection public mode.

The runner proves that camera zero is global, generated depth is positive camera
distance, one complete visible surface becomes one compensated weighted attachment,
and crop finalization preserves that attachment for every supported Spine target.
"""

from __future__ import annotations

from dataclasses import dataclass
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
    A1SingleObjectExportSettings,
    A1SourceGeometryMode,
    ExportSettings,
    attachment_setup_positions,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter import (  # noqa: E402
    export_a1_single_object,
    prepare_a1_object,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import (  # noqa: E402
    A1TextureExportMode,
    BakeExecutionSettings,
    CameraProjectionPlan,
    DepthCameraProjectionSettings,
    TextureSequenceTiming,
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
from run_all_spine_versions_integration import (  # noqa: E402
    _assert_bone_schema,
    _assert_constraint_schema,
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
)


_TEXTURE_SIZE = 96
_SEQUENCE_START = 1
_SEQUENCE_COUNT = 2
_MAX_DEPTH_POINTS = 36


@dataclass(frozen=True, slots=True)
class _Case:
    target: SpineJsonTarget
    camera_type: str
    sequence_count: int = 0

    @property
    def key(self) -> str:
        suffix = "Sequence" if self.sequence_count else "Static"
        return f"{self.target.value}_{self.camera_type}_{suffix}"


_TARGETS = (
    SpineJsonTarget.SPINE_3_8,
    SpineJsonTarget.SPINE_4_0,
    SpineJsonTarget.SPINE_4_1,
    SpineJsonTarget.SPINE_4_2,
    SpineJsonTarget.SPINE_4_3,
)


def _cases() -> tuple[_Case, ...]:
    cases = tuple(_Case(target, "PERSP") for target in _TARGETS) + (
        _Case(SpineJsonTarget.SPINE_4_2, "ORTHO"),
        _Case(SpineJsonTarget.SPINE_4_2, "PERSP", _SEQUENCE_COUNT),
    )
    _assert(len(cases) == 7, f"Depth matrix must contain seven cases: {cases}")
    return cases


def _create_relief_surface(name: str):
    """Create one visible sloped disk whose camera distance is intentionally non-flat."""

    return _create_mesh_object(
        name,
        (
            (-1.15, -0.90, -0.65),
            (1.05, -0.90, -0.20),
            (1.10, 0.90, 0.85),
            (-1.05, 0.90, 0.15),
            (0.00, 0.00, 1.10),
        ),
        (
            (0, 1, 4),
            (1, 2, 4),
            (2, 3, 4),
            (3, 0, 4),
        ),
    )


def _create_animated_emission_material(name: str):
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new(type="ShaderNodeOutputMaterial")
    emission = nodes.new(type="ShaderNodeEmission")
    emission.inputs["Strength"].default_value = 1.8
    material.node_tree.links.new(
        emission.outputs["Emission"],
        output.inputs["Surface"],
    )
    color = emission.inputs["Color"]
    color.default_value = (0.08, 0.35, 0.95, 1.0)
    color.keyframe_insert(data_path="default_value", frame=1)
    color.default_value = (1.0, 0.12, 0.04, 1.0)
    color.keyframe_insert(data_path="default_value", frame=2)
    return material


def _configure_camera(camera_type: str) -> object:
    camera = _create_camera(name=f"Depth{camera_type}Camera")
    camera.data.type = camera_type
    if camera_type == "ORTHO":
        camera.data.ortho_scale = 4.2
    elif camera_type != "PERSP":
        raise ValueError(f"Unsupported camera_type: {camera_type}")
    return camera


def _settings(directory: Path, case: _Case) -> A1SingleObjectExportSettings:
    return A1SingleObjectExportSettings(
        export=ExportSettings(
            texture_width=_TEXTURE_SIZE,
            texture_height=_TEXTURE_SIZE,
            output_directory=directory,
            images_relative_path="images",
            spine_version=case.target.exact_version,
            rig_profile=A1RigProfile.TWO_AXIS_ROTATION_SCALE.value,
            bake_margin=1,
            sequence_start_frame=_SEQUENCE_START,
            sequence_frame_count=case.sequence_count,
            sequence_timing=TextureSequenceTiming(
                scene_fps=24,
                scene_fps_base=1.0,
            ),
        ),
        prefix=case.key,
        output_stem=case.key,
        json_output_stem=case.key,
        source_geometry_mode=A1SourceGeometryMode.EVALUATED,
        projection_direction=A1ProjectionDirection.ACTIVE_CAMERA,
        uv=UvUnwrapSettings(layer_name="SpineBakeUV"),
        bake_execution=BakeExecutionSettings(
            samples=1,
            texture_export_mode=A1TextureExportMode.DEPTH_CAMERA_PROJECTION,
            depth_projection=DepthCameraProjectionSettings(
                smoothing=0.25,
                edge_threshold_fraction=0.0,
                mesh_error_pixels=10.0,
                max_points=_MAX_DEPTH_POINTS,
            ),
        ),
    )


def _typed_attachment(prepared: object) -> MeshAttachment:
    components = prepared.document_assembly.document_build.components
    _assert(len(components) == 1, f"expected one relief component: {components}")
    attachment = components[0].attachment
    _assert(isinstance(attachment, MeshAttachment), "typed relief attachment is not mesh")
    return attachment


def _assert_compensated_setup(prepared: object) -> None:
    projection = prepared.document_assembly.projections[0]
    setup_positions = attachment_setup_positions(
        projection.request.vertices,
        prepared.rig,
    )
    vertex_map = prepared.source_snapshot.vertex_by_id()
    scale = float(prepared.rig.info.uniform_scale)
    for setup_position, key in zip(
        setup_positions,
        projection.ordered_vertex_keys,
        strict=True,
    ):
        source_vertex = vertex_map[key.vertex_id]
        expected = (
            float(source_vertex.position[0]) * scale,
            -float(source_vertex.position[1]) * scale,
        )
        _assert(
            abs(setup_position[0] - expected[0]) <= 1.0e-6
            and abs(setup_position[1] - expected[1]) <= 1.0e-6,
            f"parent depth compensation failed: {setup_position} != {expected}",
        )


def _assert_prepared_relief(prepared: object) -> tuple[int, ...]:
    _assert(
        isinstance(prepared.bake_plan, CameraProjectionPlan),
        "Depth mode did not retain CameraProjectionPlan",
    )
    _assert(
        prepared.settings.bake_execution.texture_export_mode
        is A1TextureExportMode.DEPTH_CAMERA_PROJECTION,
        "prepared mode changed",
    )
    snapshot = prepared.source_snapshot
    _assert(3 <= len(snapshot.vertices) <= _MAX_DEPTH_POINTS, "depth point budget failed")
    distances = tuple(float(vertex.position[2]) for vertex in snapshot.vertices)
    _assert(
        all(distance > 0.0 for distance in distances),
        f"camera-distance depth is invalid: {distances}",
    )
    _assert(
        max(distances) - min(distances) > 0.15,
        f"relief remained flat: {distances}",
    )
    _assert(len(set(distances)) > 2, f"relief has too few depth layers: {distances}")
    _assert(
        len(prepared.rig.info.z_groups) > 2,
        "Depth mode did not build multiple Normal-style Z groups",
    )
    offsets = tuple(float(group.y_offset_pixels) for group in prepared.rig.info.z_groups)
    _assert(
        offsets and all(offset > 0.0 for offset in offsets),
        f"depth groups are not positive distances from camera zero: {offsets}",
    )
    _assert(max(offsets) > min(offsets), f"depth relief offsets remained flat: {offsets}")
    for statistic in (
        "depth_camera_vertex_rig",
        "depth_camera_global_camera_zero",
        "depth_camera_absolute_distance_retained",
        "depth_camera_parent_y_compensated",
        "depth_camera_single_attachment",
    ):
        _assert(
            int(prepared.statistics.get(statistic, 0)) == 1,
            f"missing corrected Depth statistic: {statistic}",
        )
    _assert(
        int(prepared.statistics.get("depth_projection_point_count", 0))
        == len(snapshot.vertices),
        "depth point statistic mismatch",
    )
    _assert(
        len(prepared.document_assembly.projections) == 1,
        "Depth surface was split into multiple projections",
    )
    _assert_compensated_setup(prepared)

    attachment = _typed_attachment(prepared)
    weighted = decode_weighted_vertices(attachment.vertices)
    _assert(len(weighted) == len(attachment.uvs) // 2, "weighted vertex count mismatch")
    _assert(
        all(len(vertex.influences) == 1 for vertex in weighted),
        "each retained depth point must use one generated vertex bone",
    )
    bone_indices = tuple(vertex.influences[0].bone_index for vertex in weighted)
    _assert(
        len(set(bone_indices)) == len(weighted),
        "retained depth points do not own unique generated vertex bones",
    )
    return tuple(int(value) for value in attachment.triangles)


def _json_array(document: dict[str, object], key: str) -> list[object]:
    value = document.get(key)
    _assert(isinstance(value, list), f"{key} must be an array")
    return value


def _slot_attachment_group(
    document: dict[str, object],
    prefix: str,
) -> tuple[dict[str, object], dict[str, object]]:
    slots = tuple(item for item in _json_array(document, "slots") if isinstance(item, dict))
    visual = tuple(
        slot for slot in slots if slot.get("name") == f"{prefix}_Segment_0"
    )
    _assert(len(visual) == 1, f"expected exactly one visual slot: {visual}")
    _assert(
        not any(
            isinstance(slot.get("name"), str)
            and str(slot.get("name")).startswith(f"{prefix}_Segment_1")
            for slot in slots
        ),
        "Depth document contains an unexpected Segment_1 slot",
    )
    slot = visual[0]
    slot_name = str(slot["name"])
    groups = []
    for skin in _json_array(document, "skins"):
        if not isinstance(skin, dict):
            continue
        attachments = skin.get("attachments")
        if isinstance(attachments, dict) and isinstance(attachments.get(slot_name), dict):
            groups.append(attachments[slot_name])
    _assert(len(groups) == 1, f"expected one attachment group for {slot_name}: {groups}")
    return slot, groups[0]


def _assert_json_relief(
    document: dict[str, object],
    case: _Case,
    image_size: tuple[int, int],
    prepared_triangles: tuple[int, ...],
) -> None:
    skeleton = document.get("skeleton")
    _assert(isinstance(skeleton, dict), "skeleton metadata missing")
    _assert(
        skeleton.get("spine") == case.target.exact_version,
        f"wrong Spine version: {skeleton}",
    )
    _assert_bone_schema(document, case.target)
    _assert_constraint_schema(document, case.target)

    slot, attachments = _slot_attachment_group(document, case.key)
    setup_name = str(slot["attachment"])
    _assert(setup_name in attachments, "setup attachment missing")
    _assert(len(attachments) == 1, f"unexpected attachment count: {attachments}")
    attachment = attachments[setup_name]
    _assert(isinstance(attachment, dict), "setup attachment is not object")
    _assert(attachment.get("type") == "mesh", "setup attachment is not mesh")

    uvs = attachment.get("uvs")
    triangles = attachment.get("triangles")
    vertices = attachment.get("vertices")
    _assert(isinstance(uvs, list) and len(uvs) >= 6 and len(uvs) % 2 == 0, "invalid UVs")
    _assert(all(0.0 <= float(value) <= 1.0 for value in uvs), "cropped UV outside 0..1")
    _assert(isinstance(triangles, list), "triangles missing")
    _assert(
        tuple(int(value) for value in triangles) == prepared_triangles,
        "crop changed triangles",
    )
    vertex_count = len(uvs) // 2
    _assert(
        all(0 <= int(index) < vertex_count for index in triangles),
        "triangles reference missing point",
    )
    _assert(isinstance(vertices, list) and len(vertices) > len(uvs), "attachment is not weighted")
    _assert(
        int(round(float(attachment.get("width", 0.0)))) == image_size[0],
        "attachment width does not match crop",
    )
    _assert(
        int(round(float(attachment.get("height", 0.0)))) == image_size[1],
        "attachment height does not match crop",
    )

    bone_names = {
        str(bone.get("name"))
        for bone in _json_array(document, "bones")
        if isinstance(bone, dict)
    }
    _assert(f"{case.key}_main" in bone_names, "main control missing")
    generated = tuple(
        name
        for name in bone_names
        if name.startswith(f"{case.key}_Segment_0_vertex_")
    )
    _assert(
        len(generated) >= vertex_count,
        f"generated relief vertex bones missing: {generated}",
    )
    _assert(
        not any(name.startswith(f"{case.key}_Segment_1") for name in bone_names),
        "Depth document generated Segment_1 bones",
    )

    if case.sequence_count:
        sequence = attachment.get("sequence")
        _assert(isinstance(sequence, dict), "native sequence metadata missing")
        _assert(sequence.get("count") == case.sequence_count, f"sequence count mismatch: {sequence}")
        animations = document.get("animations")
        _assert(isinstance(animations, dict), "animations missing")
        _assert("sequence" in json.dumps(animations), "sequence timeline missing")


def _prepare_case_scene(case: _Case) -> tuple[object, object, object]:
    _clear_scene()
    _purge_orphan_scene_data()
    _configure_scene()
    bpy.context.scene.cycles.samples = 1
    camera = _configure_camera(case.camera_type)
    source = _create_relief_surface(f"{case.key}_Source")
    source.location = (0.35, -0.20, -0.75)
    material = _create_animated_emission_material(f"{case.key}_Material")
    source.data.materials.append(material)
    sentinel = _create_sentinel()
    sentinel.location = (8.0, 0.0, 0.0)
    _activate_only(sentinel)
    source.select_set(False)
    bpy.context.scene.frame_set(_SEQUENCE_START)
    bpy.context.view_layer.update()
    return source, material, camera


def _run_case(output_root: Path, case: _Case) -> None:
    directory = output_root / case.key
    directory.mkdir(parents=True, exist_ok=False)
    source, material, camera = _prepare_case_scene(case)
    settings = _settings(directory, case)

    context_before = _capture_context()
    bake_before = _capture_scene_bake_state()
    render_before = _scene_render_fingerprint()
    material_before = _material_fingerprint(material)
    camera_matrix_before = tuple(float(value) for row in camera.matrix_world for value in row)
    temporary_before = _temporary_datablock_names()

    prepared = prepare_a1_object(
        source,
        settings,
        context=bpy.context,
        scene=bpy.context.scene,
    )
    prepared_triangles = _assert_prepared_relief(prepared)
    _assert(_capture_context() == context_before, "prepare changed context")
    _assert(_capture_scene_bake_state() == bake_before, "prepare changed bake state")
    _assert(_scene_render_fingerprint() == render_before, "prepare changed render state")
    _assert(_temporary_datablock_names() == temporary_before, "prepare leaked datablocks")

    result = export_a1_single_object(
        source,
        settings,
        context=bpy.context,
        scene=bpy.context.scene,
    )
    _assert(result.success, f"Depth export failed for {case.key}: {result.issues}")
    expected_file_count = 1 + max(1, case.sequence_count)
    _assert(
        len(result.output_files) == expected_file_count,
        f"unexpected output count for {case.key}: {result.output_files}",
    )
    json_path = result.output_files[0]
    image_paths = tuple(result.output_files[1:])
    _assert(json_path.suffix.lower() == ".json", "JSON output must be first")
    _assert(
        all(path.read_bytes().startswith(PNG_SIGNATURE) for path in image_paths),
        f"invalid PNG for {case.key}",
    )
    image_data = tuple(_read_image(path) for path in image_paths)
    _assert(
        len({size for size, _pixels in image_data}) == 1,
        f"sequence crop changed between frames: {[size for size, _ in image_data]}",
    )
    image_size = image_data[0][0]
    _assert(
        1 <= image_size[0] <= _TEXTURE_SIZE
        and 1 <= image_size[1] <= _TEXTURE_SIZE,
        f"invalid crop size: {image_size}",
    )

    document = json.loads(json_path.read_text(encoding="utf-8"))
    _assert(isinstance(document, dict), "serialized Spine document must be object")
    _assert_json_relief(document, case, image_size, prepared_triangles)

    _assert(_capture_context() == context_before, "export changed context")
    _assert(_capture_scene_bake_state() == bake_before, "export changed bake state")
    _assert(_scene_render_fingerprint() == render_before, "export changed render state")
    _assert(_material_fingerprint(material) == material_before, "export changed material")
    _assert(
        tuple(float(value) for row in camera.matrix_world for value in row)
        == camera_matrix_before,
        "export changed camera matrix",
    )
    _assert(_temporary_datablock_names() == temporary_before, "export leaked datablocks")


def main() -> None:
    started = time_started = __import__("time").perf_counter()
    with tempfile.TemporaryDirectory(prefix="spine2d-depth-camera-") as directory:
        output_root = Path(directory)
        for case in _cases():
            _run_case(output_root, case)
    elapsed = __import__("time").perf_counter() - time_started
    print(
        "[DEPTH-CAMERA] PASS cases=7 targets=3.8,4.0,4.1,4.2,4.3 "
        f"camera_zero=shared one_attachment=true elapsed={elapsed:.2f}s"
    )
    _assert(started > 0.0, "timer did not initialize")


if __name__ == "__main__":
    main()
