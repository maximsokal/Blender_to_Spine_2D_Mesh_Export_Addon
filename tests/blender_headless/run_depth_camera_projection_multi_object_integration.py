"""Real Blender 5.2 multi-object acceptance for Depth Camera Projection.

One object owns a two-frame native Spine 4.2 material sequence while its sibling remains
static. Source-object and active-camera animation are intentionally present and must not
move the rendered Depth silhouette, crop, relief geometry, or camera-space placement.
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
    A1MultiObjectExportSettings,
    A1MultiObjectMode,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter import (  # noqa: E402
    A1MultiObjectSource,
    export_a1_multi_object,
    prepare_a1_multi_object,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine import (  # noqa: E402
    SpineJsonTarget,
)
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
    _aim_at,
    _configure_scene,
    _create_camera,
    _purge_orphan_scene_data,
    _read_image,
    _scene_render_fingerprint,
)
from run_depth_camera_projection_integration import (  # noqa: E402
    _Case,
    _create_animated_emission_material,
    _create_relief_surface,
    _settings as _single_settings,
)


_TARGET = SpineJsonTarget.SPINE_4_2
_SEQUENCE_COUNT = 2


def _static_emission_material(name: str):
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new(type="ShaderNodeOutputMaterial")
    emission = nodes.new(type="ShaderNodeEmission")
    emission.inputs["Strength"].default_value = 1.8
    emission.inputs["Color"].default_value = (0.08, 0.95, 0.22, 1.0)
    material.node_tree.links.new(
        emission.outputs["Emission"],
        output.inputs["Surface"],
    )
    return material


def _keyframe_sequence_source(source: object) -> None:
    scene = bpy.context.scene
    scene.frame_set(1)
    source.location = (-1.10, 0.0, 0.0)
    source.rotation_euler = (0.0, 0.0, 0.0)
    source.keyframe_insert(data_path="location", frame=1)
    source.keyframe_insert(data_path="rotation_euler", frame=1)

    scene.frame_set(2)
    source.location = (2.35, 0.65, -1.25)
    source.rotation_euler = (0.55, -0.30, 0.85)
    source.keyframe_insert(data_path="location", frame=2)
    source.keyframe_insert(data_path="rotation_euler", frame=2)
    scene.frame_set(1)
    bpy.context.view_layer.update()


def _keyframe_active_camera(camera: object) -> None:
    scene = bpy.context.scene
    scene.frame_set(1)
    camera.location = (0.0, 0.0, 5.5)
    _aim_at(camera, camera.location.__class__((0.0, 0.0, 0.0)))
    camera.data.lens = 52.0
    camera.keyframe_insert(data_path="location", frame=1)
    camera.keyframe_insert(data_path="rotation_euler", frame=1)
    camera.data.keyframe_insert(data_path="lens", frame=1)

    scene.frame_set(2)
    camera.location = (2.2, -1.1, 7.0)
    _aim_at(camera, camera.location.__class__((0.4, 0.2, -0.5)))
    camera.data.lens = 73.0
    camera.keyframe_insert(data_path="location", frame=2)
    camera.keyframe_insert(data_path="rotation_euler", frame=2)
    camera.data.keyframe_insert(data_path="lens", frame=2)
    scene.frame_set(1)
    bpy.context.view_layer.update()


def _sources(output_directory: Path) -> tuple[A1MultiObjectSource, ...]:
    sequence_object = _create_relief_surface("DepthSequenceA_Source")
    _keyframe_sequence_source(sequence_object)
    sequence_material = _create_animated_emission_material(
        "DepthSequenceA_Material"
    )
    sequence_object.data.materials.append(sequence_material)

    static_object = _create_relief_surface("DepthStaticB_Source")
    static_object.location = (1.10, 0.0, -3.0)
    static_object.scale = (0.82, 0.82, 0.82)
    static_material = _static_emission_material("DepthStaticB_Material")
    static_object.data.materials.append(static_material)

    sequence_settings = replace(
        _single_settings(
            output_directory,
            _Case(_TARGET, "PERSP", _SEQUENCE_COUNT),
        ),
        prefix="DepthSequenceA",
        output_stem="DepthSequenceA",
        json_output_stem=None,
    )
    static_settings = replace(
        _single_settings(
            output_directory,
            _Case(_TARGET, "PERSP", 0),
        ),
        prefix="DepthStaticB",
        output_stem="DepthStaticB",
        json_output_stem=None,
    )
    return (
        A1MultiObjectSource(
            source_object=sequence_object,
            component_id="sequence_depth",
            animation_namespace="sequence_depth",
            settings=sequence_settings,
        ),
        A1MultiObjectSource(
            source_object=static_object,
            component_id="static_depth",
            animation_namespace="static_depth",
            settings=static_settings,
        ),
    )


def _attachment_groups(document: dict[str, object]) -> dict[str, dict[str, object]]:
    resolved: dict[str, dict[str, object]] = {}
    skins = document.get("skins")
    _assert(isinstance(skins, list), "skins must be array")
    for skin in skins:
        if not isinstance(skin, dict):
            continue
        attachments = skin.get("attachments")
        if not isinstance(attachments, dict):
            continue
        for slot_name, group in attachments.items():
            if isinstance(slot_name, str) and isinstance(group, dict):
                resolved[slot_name] = group
    return resolved


def _one_group(
    groups: dict[str, dict[str, object]],
    prefix: str,
) -> tuple[str, dict[str, object]]:
    matches = tuple(
        (slot_name, group)
        for slot_name, group in groups.items()
        if slot_name.startswith(f"{prefix}_Segment_")
    )
    _assert(len(matches) == 1, f"expected one {prefix} attachment group: {matches}")
    _assert(
        matches[0][0] == f"{prefix}_Segment_0",
        f"Depth object must serialize only Segment_0: {matches[0][0]}",
    )
    return matches[0]


def _assert_weighted_depth_attachment(
    attachment: dict[str, object],
    prefix: str,
) -> None:
    _assert(attachment.get("type") == "mesh", f"{prefix} attachment is not mesh")
    uvs = attachment.get("uvs")
    vertices = attachment.get("vertices")
    triangles = attachment.get("triangles")
    _assert(
        isinstance(uvs, list) and len(uvs) >= 6 and len(uvs) % 2 == 0,
        f"{prefix} invalid UV",
    )
    _assert(
        all(0.0 <= float(value) <= 1.0 for value in uvs),
        f"{prefix} UV outside crop",
    )
    _assert(
        isinstance(vertices, list) and len(vertices) > len(uvs),
        f"{prefix} mesh is flat/unweighted",
    )
    _assert(
        isinstance(triangles, list) and len(triangles) >= 3,
        f"{prefix} triangles missing",
    )


def _alpha_mask(pixels: tuple[float, ...]) -> tuple[bool, ...]:
    return tuple(pixels[index + 3] > 0.05 for index in range(0, len(pixels), 4))


def _visible_rgb_signature(pixels: tuple[float, ...]) -> tuple[int, ...]:
    values: list[int] = []
    for index in range(0, len(pixels), 4):
        if pixels[index + 3] <= 0.05:
            continue
        values.extend(
            int(round(max(0.0, min(1.0, pixels[index + channel])) * 255.0))
            for channel in range(3)
        )
    return tuple(values)


def _main_position(document: dict[str, object], prefix: str) -> tuple[float, float]:
    bones = document.get("bones")
    _assert(isinstance(bones, list), "bones must be an array")
    for bone in bones:
        if isinstance(bone, dict) and bone.get("name") == f"{prefix}_main":
            return float(bone.get("x", 0.0)), float(bone.get("y", 0.0))
    raise AssertionError(f"missing {prefix}_main")


def main() -> None:
    _clear_scene()
    _purge_orphan_scene_data()
    _configure_scene()
    bpy.context.scene.cycles.samples = 1
    camera = _create_camera(name="DepthMultiCamera")
    camera.data.type = "PERSP"
    _keyframe_active_camera(camera)
    sentinel = _create_sentinel()
    sentinel.location = (8.0, 0.0, 0.0)
    _activate_only(sentinel)
    bpy.context.scene.frame_set(1)
    bpy.context.view_layer.update()

    with tempfile.TemporaryDirectory(prefix="spine2d-depth-multi-") as temp_directory:
        output_directory = Path(temp_directory)
        sources = _sources(output_directory)
        settings = A1MultiObjectExportSettings(
            output_directory=output_directory,
            output_stem="DepthCameraProjectionMixedTiming",
            mode=A1MultiObjectMode.STANDALONE,
        )

        context_before = _capture_context()
        bake_before = _capture_scene_bake_state()
        render_before = _scene_render_fingerprint()
        temporary_before = _temporary_datablock_names()
        material_before = tuple(
            _material_fingerprint(source.source_object.data.materials[0])
            for source in sources
        )

        prepared = prepare_a1_multi_object(
            sources,
            settings,
            context=bpy.context,
            scene=bpy.context.scene,
        )
        _assert(len(prepared.objects) == 2, "prepared depth object count mismatch")
        ranges: dict[str, tuple[float, float]] = {}
        main_positions: dict[str, tuple[float, float]] = {}
        for item in prepared.objects:
            _assert(
                int(item.statistics.get("depth_projection_point_count", 0)) >= 3,
                f"{item.object_id} depth surface missing",
            )
            offsets = tuple(float(group.y_offset_pixels) for group in item.rig.info.z_groups)
            _assert(
                offsets and all(offset > 0.0 for offset in offsets),
                f"{item.object_id} must retain positive distance from camera zero: {offsets}",
            )
            _assert(
                len(item.document_assembly.document_build.components) == 1,
                f"{item.object_id} was split into multiple attachments",
            )
            ranges[item.object_id] = (min(offsets), max(offsets))
            request_main = item.rig.request.main_position_pixels
            _assert(request_main is not None, f"{item.object_id} main position missing")
            main_positions[item.object_id] = tuple(float(value) for value in request_main)

        sequence_range = ranges["DepthSequenceA_Source"]
        static_range = ranges["DepthStaticB_Source"]
        _assert(
            sequence_range[1] < static_range[0],
            f"objects lost shared camera-zero depth ordering: {ranges}",
        )
        _assert(
            main_positions["DepthSequenceA_Source"]
            != main_positions["DepthStaticB_Source"],
            f"projected object origins collapsed: {main_positions}",
        )

        _assert(_capture_context() == context_before, "preparation changed context")
        _assert(
            _capture_scene_bake_state() == bake_before,
            "preparation changed bake state",
        )
        _assert(
            _scene_render_fingerprint() == render_before,
            "preparation changed render state",
        )
        _assert(
            _temporary_datablock_names() == temporary_before,
            "preparation leaked datablocks",
        )

        result = export_a1_multi_object(
            sources,
            settings,
            context=bpy.context,
            scene=bpy.context.scene,
        )
        _assert(result.success, f"multi-object depth export failed: {result.issues}")
        _assert(
            len(result.output_files) == 4,
            f"expected JSON + 3 PNG: {result.output_files}",
        )
        json_path = result.output_files[0]
        image_paths = tuple(result.output_files[1:])
        _assert(json_path.suffix.lower() == ".json", "JSON output must be first")
        _assert(
            all(path.read_bytes().startswith(PNG_SIGNATURE) for path in image_paths),
            "invalid PNG",
        )
        _assert(
            len({path.name for path in image_paths}) == 3,
            "texture output names collided",
        )

        sequence_paths = tuple(
            sorted(path for path in image_paths if "DepthSequenceA" in path.name)
        )
        static_paths = tuple(
            path for path in image_paths if "DepthStaticB" in path.name
        )
        _assert(
            len(sequence_paths) == 2,
            f"sequence output count mismatch: {sequence_paths}",
        )
        _assert(
            len(static_paths) == 1,
            f"static output count mismatch: {static_paths}",
        )

        sequence_images = tuple(_read_image(path) for path in sequence_paths)
        _assert(
            sequence_images[0][0] == sequence_images[1][0],
            f"animated object/camera changed sequence crop: {sequence_images[0][0]} "
            f"!= {sequence_images[1][0]}",
        )
        _assert(
            _alpha_mask(sequence_images[0][1]) == _alpha_mask(sequence_images[1][1]),
            "animated object/camera changed Depth sequence silhouette",
        )
        _assert(
            _visible_rgb_signature(sequence_images[0][1])
            != _visible_rgb_signature(sequence_images[1][1]),
            "animated material did not change visible sequence RGB",
        )
        _assert(
            sequence_paths[0].read_bytes() != sequence_paths[1].read_bytes(),
            "material sequence PNGs are byte-identical",
        )
        for path in image_paths:
            (width, height), _pixels = _read_image(path)
            _assert(
                1 <= width <= 96 and 1 <= height <= 96,
                f"invalid crop for {path}: {(width, height)}",
            )

        document = json.loads(json_path.read_text(encoding="utf-8"))
        _assert(isinstance(document, dict), "combined JSON must be object")
        skeleton = document.get("skeleton")
        _assert(isinstance(skeleton, dict), "skeleton metadata missing")
        _assert(
            skeleton.get("spine") == _TARGET.exact_version,
            "wrong Spine target",
        )
        _assert(
            "all_objects" not in json.dumps(document.get("bones", [])),
            "standalone output gained connected wrapper",
        )
        _assert(
            _main_position(document, "DepthSequenceA")
            != _main_position(document, "DepthStaticB"),
            "serialized main bones lost projected relative placement",
        )

        groups = _attachment_groups(document)
        sequence_slot, sequence_group = _one_group(groups, "DepthSequenceA")
        static_slot, static_group = _one_group(groups, "DepthStaticB")
        _assert(
            len(sequence_group) == 1,
            f"native sequence should use one attachment: {sequence_group}",
        )
        _assert(
            len(static_group) == 1,
            f"static object should use one attachment: {static_group}",
        )
        sequence_attachment = next(iter(sequence_group.values()))
        static_attachment = next(iter(static_group.values()))
        _assert(isinstance(sequence_attachment, dict), "sequence attachment invalid")
        _assert(isinstance(static_attachment, dict), "static attachment invalid")
        _assert_weighted_depth_attachment(sequence_attachment, "DepthSequenceA")
        _assert_weighted_depth_attachment(static_attachment, "DepthStaticB")
        sequence = sequence_attachment.get("sequence")
        _assert(
            isinstance(sequence, dict),
            "sequence owner lost native sequence metadata",
        )
        _assert(
            sequence.get("count") == _SEQUENCE_COUNT,
            f"sequence count mismatch: {sequence}",
        )
        _assert(
            "sequence" not in static_attachment,
            "static sibling inherited sequence metadata",
        )

        animations_text = json.dumps(document.get("animations", {}), sort_keys=True)
        _assert(sequence_slot in animations_text, "sequence slot timeline missing")
        _assert(
            static_slot not in animations_text,
            "static sibling inherited sequence timeline",
        )

        bone_names = {
            str(item.get("name"))
            for item in document.get("bones", [])
            if isinstance(item, dict)
        }
        for prefix in ("DepthSequenceA", "DepthStaticB"):
            _assert(f"{prefix}_main" in bone_names, f"{prefix} main missing")
            _assert(
                any(
                    name.startswith(f"{prefix}_Segment_0_vertex_")
                    for name in bone_names
                ),
                f"{prefix} generated relief bones missing",
            )
            _assert(
                not any(name.startswith(f"{prefix}_Segment_1") for name in bone_names),
                f"{prefix} generated an unexpected second segment",
            )

        _assert(_capture_context() == context_before, "export changed context")
        _assert(
            _capture_scene_bake_state() == bake_before,
            "export changed bake state",
        )
        _assert(
            _scene_render_fingerprint() == render_before,
            "export changed render state",
        )
        _assert(
            _temporary_datablock_names() == temporary_before,
            "export leaked datablocks",
        )
        _assert(
            tuple(
                _material_fingerprint(source.source_object.data.materials[0])
                for source in sources
            )
            == material_before,
            "export changed source materials",
        )

    print(
        "[DEPTH-CAMERA-MULTI] PASS target=4.2 objects=2 "
        "camera_zero=shared attachments=1+1 material_sequence=2 static=1"
    )


if __name__ == "__main__":
    main()
