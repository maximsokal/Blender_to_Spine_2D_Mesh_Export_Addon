"""Real Blender 5.2 multi-object acceptance for Depth Camera Projection.

One object owns a two-frame native Spine 4.2 sequence while its sibling remains static.
Both objects retain independent optimized depth-relief meshes and generated vertex rigs
inside one public STANDALONE multi-object JSON/texture transaction.
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


def _sources(output_directory: Path) -> tuple[A1MultiObjectSource, ...]:
    sequence_object = _create_relief_surface("DepthSequenceA_Source")
    sequence_object.location.x = -1.10
    sequence_material = _create_animated_emission_material(
        "DepthSequenceA_Material"
    )
    sequence_object.data.materials.append(sequence_material)

    static_object = _create_relief_surface("DepthStaticB_Source")
    static_object.location.x = 1.10
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


def main() -> None:
    _clear_scene()
    _purge_orphan_scene_data()
    _configure_scene()
    bpy.context.scene.cycles.samples = 1
    camera = _create_camera(name="DepthMultiCamera")
    camera.data.type = "PERSP"
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
        for item in prepared.objects:
            _assert(
                int(item.statistics.get("depth_projection_point_count", 0)) >= 3,
                f"{item.object_id} depth surface missing",
            )
            offsets = tuple(group.y_offset_pixels for group in item.rig.info.z_groups)
            _assert(
                offsets and min(offsets) == 0.0,
                f"{item.object_id} depth base mismatch",
            )
            _assert(
                all(offset >= 0.0 for offset in offsets),
                f"{item.object_id} depth points extend away",
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
            path for path in image_paths if "DepthSequenceA" in path.name
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
        sequence_payloads = {path.read_bytes() for path in sequence_paths}
        _assert(len(sequence_payloads) == 2, "sequence PNGs are identical")
        _assert(
            static_paths[0].read_bytes() not in sequence_payloads,
            "static PNG collided with sequence",
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
                    name.startswith(f"{prefix}_Segment_") and "_vertex_" in name
                    for name in bone_names
                ),
                f"{prefix} generated relief bones missing",
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
        "sequence=2 static=1 png=3"
    )


if __name__ == "__main__":
    main()
