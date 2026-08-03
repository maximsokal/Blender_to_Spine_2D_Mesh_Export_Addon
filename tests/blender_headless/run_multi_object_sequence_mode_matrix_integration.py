"""Real Blender 5.2 multi-object sequence matrix for Spine 3.8 through 4.3.

The runner exercises only the public ``export_a1_multi_object`` service.  Every case
exports two animated mesh objects for two timeline frames at a 128x128 source texture
size.  Both production texture modes are covered:

* Normal / UV Segments uses Cycles object baking into the generated Spine UV layout;
* Camera Projection uses the active Camera and the production render/crop pipeline.

The matrix intentionally validates files and serialized runtime data rather than merely
checking planner objects.  Spine 3.8/4.0 must use attachment swaps, while Spine 4.1+
must use native sequence metadata and sequence timelines.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys
import tempfile
import time
import traceback
from typing import Iterable, Mapping

import bpy


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIRECTORY.parents[1]
for path in (SCRIPT_DIRECTORY, REPOSITORY_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from Blender_to_Spine2D_Mesh_Exporter.application import (  # noqa: E402
    A1MultiObjectExportSettings,
    A1MultiObjectMode,
    A1SingleObjectExportSettings,
    A1SourceGeometryMode,
    ExportSettings,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter import (  # noqa: E402
    A1MultiObjectSource,
    export_a1_multi_object,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import (  # noqa: E402
    A1TextureExportMode,
    BakeExecutionSettings,
    TextureSequenceTiming,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine import (  # noqa: E402
    A1RigProfile,
    SpineJsonTarget,
    SpineTextureAnimationEncoding,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.uv import UvUnwrapSettings  # noqa: E402
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


_TEXTURE_SIZE = 128
_SEQUENCE_START_FRAME = 1
_SEQUENCE_FRAME_COUNT = 2
_ANALYSIS_FRAME = 19
_OBJECT_COUNT = 2
_FLOAT32_EPSILON = 2.0 ** -23
_MATRIX_COMPARE_ULPS = 8.0

_TARGETS = (
    SpineJsonTarget.SPINE_3_8,
    SpineJsonTarget.SPINE_4_0,
    SpineJsonTarget.SPINE_4_1,
    SpineJsonTarget.SPINE_4_2,
    SpineJsonTarget.SPINE_4_3,
)
_TEXTURE_MODES = (
    A1TextureExportMode.NORMAL_UV_SEGMENTS,
    A1TextureExportMode.CAMERA_PROJECTION,
)
_TARGET_TOKENS = {
    SpineJsonTarget.SPINE_3_8: "Spine38",
    SpineJsonTarget.SPINE_4_0: "Spine40",
    SpineJsonTarget.SPINE_4_1: "Spine41",
    SpineJsonTarget.SPINE_4_2: "Spine42",
    SpineJsonTarget.SPINE_4_3: "Spine43",
}
_MODE_TOKENS = {
    A1TextureExportMode.NORMAL_UV_SEGMENTS: "NormalUv",
    A1TextureExportMode.CAMERA_PROJECTION: "CameraProjection",
}


@dataclass(frozen=True, slots=True)
class _Case:
    target: SpineJsonTarget
    texture_mode: A1TextureExportMode

    @property
    def key(self) -> str:
        return f"{_TARGET_TOKENS[self.target]}_{_MODE_TOKENS[self.texture_mode]}"


@dataclass(frozen=True, slots=True)
class _SourceFixture:
    source: A1MultiObjectSource
    material: object
    expected_uv_scalar_count: int

    @property
    def object(self) -> object:
        return self.source.source_object

    @property
    def prefix(self) -> str:
        return self.source.settings.prefix

    @property
    def output_stem(self) -> str:
        return self.source.settings.output_stem


@dataclass(frozen=True, slots=True)
class _ImageSummary:
    path: Path
    size: tuple[int, int]
    signature: tuple[float, float, float, float]
    content: bytes


def _create_pentagon(name: str):
    return _create_mesh_object(
        name,
        (
            (-0.75, -0.55, 0.0),
            (0.10, -0.80, 0.0),
            (0.80, -0.25, 0.0),
            (0.55, 0.70, 0.0),
            (-0.55, 0.75, 0.0),
        ),
        ((0, 1, 2, 3, 4),),
    )


def _create_quad(name: str):
    return _create_mesh_object(
        name,
        (
            (-0.70, -0.60, 0.0),
            (0.75, -0.50, 0.0),
            (0.60, 0.65, 0.0),
            (-0.65, 0.70, 0.0),
        ),
        ((0, 1, 2, 3),),
    )


def _scaled_math(nodes, name: str, factor: float):
    node = nodes.new(type="ShaderNodeMath")
    node.name = name
    node.operation = "MULTIPLY"
    node.inputs[1].default_value = factor
    return node


def _add_math(nodes, name: str, offset: float | None = None):
    node = nodes.new(type="ShaderNodeMath")
    node.name = name
    node.operation = "ADD"
    if offset is not None:
        node.inputs[1].default_value = offset
    return node


def _variant_colors(variant: int):
    if variant == 0:
        return (
            (0.02, 0.10, 1.00, 1.0),
            (0.05, 1.00, 0.65, 1.0),
            (0.95, 0.03, 0.20, 1.0),
            (1.00, 0.55, 0.02, 1.0),
            (0.25, 0.05, 0.90, 1.0),
            (0.10, 0.80, 1.00, 1.0),
        )
    if variant == 1:
        return (
            (0.75, 0.02, 1.00, 1.0),
            (1.00, 0.85, 0.03, 1.0),
            (0.02, 0.80, 0.95, 1.0),
            (0.95, 0.03, 0.70, 1.0),
            (0.05, 0.95, 0.35, 1.0),
            (0.95, 0.20, 0.04, 1.0),
        )
    raise ValueError(f"Unsupported material variant: {variant}")


def _create_animated_camera_reflection_material(name: str, *, variant: int):
    """Create an animated surface using Camera once and Reflection twice."""

    colors = _variant_colors(variant)
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()

    output = nodes.new(type="ShaderNodeOutputMaterial")
    emission = nodes.new(type="ShaderNodeEmission")
    coordinates = nodes.new(type="ShaderNodeTexCoord")

    camera_xyz = nodes.new(type="ShaderNodeSeparateXYZ")
    camera_xyz.name = "Camera Coordinates"
    reflection_xyz_a = nodes.new(type="ShaderNodeSeparateXYZ")
    reflection_xyz_a.name = "Reflection Coordinates A"
    reflection_xyz_b = nodes.new(type="ShaderNodeSeparateXYZ")
    reflection_xyz_b.name = "Reflection Coordinates B"

    camera_scale = _scaled_math(nodes, "Camera Scale", 0.28)
    reflection_scale_a = _scaled_math(nodes, "Reflection X Scale", 0.22)
    reflection_scale_b = _scaled_math(nodes, "Reflection Z Scale", 0.18)
    camera_plus_reflection = _add_math(nodes, "Camera Plus Reflection")
    reflection_sum = _add_math(nodes, "Reflection Sum")
    centered_factor = _add_math(nodes, "Centered Factor", 0.50)
    centered_factor.use_clamp = True
    ramp = nodes.new(type="ShaderNodeValToRGB")

    links.new(coordinates.outputs["Camera"], camera_xyz.inputs["Vector"])
    links.new(coordinates.outputs["Reflection"], reflection_xyz_a.inputs["Vector"])
    links.new(coordinates.outputs["Reflection"], reflection_xyz_b.inputs["Vector"])
    links.new(camera_xyz.outputs["Z"], camera_scale.inputs[0])
    links.new(reflection_xyz_a.outputs["X"], reflection_scale_a.inputs[0])
    links.new(reflection_xyz_b.outputs["Z"], reflection_scale_b.inputs[0])
    links.new(camera_scale.outputs[0], camera_plus_reflection.inputs[0])
    links.new(reflection_scale_a.outputs[0], camera_plus_reflection.inputs[1])
    links.new(camera_plus_reflection.outputs[0], reflection_sum.inputs[0])
    links.new(reflection_scale_b.outputs[0], reflection_sum.inputs[1])
    links.new(reflection_sum.outputs[0], centered_factor.inputs[0])
    links.new(centered_factor.outputs[0], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], emission.inputs["Color"])
    links.new(emission.outputs["Emission"], output.inputs["Surface"])

    left = ramp.color_ramp.elements[0]
    right = ramp.color_ramp.elements[1]
    frames = (
        (
            1,
            colors[0],
            colors[1],
            0.85 + variant * 0.20,
        ),
        (
            2,
            colors[2],
            colors[3],
            1.65 + variant * 0.25,
        ),
        (
            _ANALYSIS_FRAME,
            colors[4],
            colors[5],
            1.10 + variant * 0.15,
        ),
    )
    for frame, left_color, right_color, strength in frames:
        left.color = left_color
        right.color = right_color
        emission.inputs["Strength"].default_value = strength
        left.keyframe_insert(data_path="color", frame=frame)
        right.keyframe_insert(data_path="color", frame=frame)
        emission.inputs["Strength"].keyframe_insert(
            data_path="default_value",
            frame=frame,
        )
    return material


def _animate_source_transform(source: object, *, variant: int) -> None:
    """Key two bake frames and a distinct frame used for export planning."""

    base_x = -0.85 if variant == 0 else 0.85
    transforms = (
        (
            1,
            (base_x - 0.18, -0.18 + variant * 0.20, variant * 0.18),
            (0.10 + variant * 0.08, -0.22, -0.22 + variant * 0.38),
            (0.80 + variant * 0.08, 1.02 - variant * 0.12, 1.0),
        ),
        (
            2,
            (base_x + 0.22, 0.22 - variant * 0.12, 0.28 - variant * 0.12),
            (0.34, 0.18 - variant * 0.10, 0.38 + variant * 0.28),
            (1.08 - variant * 0.12, 0.76 + variant * 0.18, 1.0),
        ),
        (
            _ANALYSIS_FRAME,
            (base_x, -0.46 + variant * 0.16, -0.18 + variant * 0.36),
            (0.62 - variant * 0.20, -0.58 + variant * 0.24, 0.90 - variant * 0.22),
            (1.22 - variant * 0.14, 0.88 + variant * 0.10, 1.0),
        ),
    )
    for frame, location, rotation, scale in transforms:
        source.location = location
        source.rotation_euler = rotation
        source.scale = scale
        source.keyframe_insert(data_path="location", frame=frame)
        source.keyframe_insert(data_path="rotation_euler", frame=frame)
        source.keyframe_insert(data_path="scale", frame=frame)


def _matrix_tuple(obj: object) -> tuple[float, ...]:
    matrix = obj.matrix_world
    return tuple(
        float(matrix[row][column])
        for row in range(4)
        for column in range(4)
    )


def _matrices_close(
    first: tuple[float, ...],
    second: tuple[float, ...],
) -> bool:
    if len(first) != 16 or len(second) != 16:
        return False
    tolerance_scale = _FLOAT32_EPSILON * _MATRIX_COMPARE_ULPS
    return all(
        abs(left - right)
        <= tolerance_scale * max(1.0, abs(left), abs(right)) + tolerance_scale
        for left, right in zip(first, second, strict=True)
    )


def _frame_matrices(
    scene: object,
    obj: object,
) -> tuple[tuple[float, ...], ...]:
    original_frame = int(scene.frame_current)
    try:
        matrices = []
        for frame in range(
            _SEQUENCE_START_FRAME,
            _SEQUENCE_START_FRAME + _SEQUENCE_FRAME_COUNT,
        ):
            scene.frame_set(frame)
            bpy.context.view_layer.update()
            matrices.append(_matrix_tuple(obj))
        return tuple(matrices)
    finally:
        scene.frame_set(original_frame)
        bpy.context.view_layer.update()


def _object_settings(
    output_directory: Path,
    case: _Case,
    *,
    prefix: str,
) -> A1SingleObjectExportSettings:
    return A1SingleObjectExportSettings(
        export=ExportSettings(
            texture_width=_TEXTURE_SIZE,
            texture_height=_TEXTURE_SIZE,
            output_directory=output_directory,
            images_relative_path="images",
            spine_version=case.target.exact_version,
            rig_profile=A1RigProfile.TWO_AXIS_ROTATION_SCALE.value,
            bake_margin=1,
            sequence_start_frame=_SEQUENCE_START_FRAME,
            sequence_frame_count=_SEQUENCE_FRAME_COUNT,
            sequence_timing=TextureSequenceTiming(
                scene_fps=24,
                scene_fps_base=1.0,
            ),
        ),
        prefix=prefix,
        output_stem=prefix,
        json_output_stem=prefix,
        source_geometry_mode=A1SourceGeometryMode.EVALUATED,
        uv=UvUnwrapSettings(layer_name="SpineBakeUV"),
        bake_execution=BakeExecutionSettings(
            samples=1,
            texture_export_mode=case.texture_mode,
        ),
    )


def _build_sources(
    output_directory: Path,
    case: _Case,
) -> tuple[_SourceFixture, ...]:
    geometries = (
        (_create_pentagon, 10),
        (_create_quad, 8),
    )
    fixtures: list[_SourceFixture] = []
    for index, (builder, uv_scalar_count) in enumerate(geometries):
        suffix = chr(ord("A") + index)
        prefix = f"{case.key}Object{suffix}"
        obj = builder(f"{prefix}_Source")
        material = _create_animated_camera_reflection_material(
            f"{prefix}_Material",
            variant=index,
        )
        obj.data.materials.append(material)
        _animate_source_transform(obj, variant=index)
        fixtures.append(
            _SourceFixture(
                source=A1MultiObjectSource(
                    source_object=obj,
                    component_id=f"{case.key.casefold()}_component_{index + 1}",
                    animation_namespace=f"object_{index + 1}",
                    settings=_object_settings(
                        output_directory,
                        case,
                        prefix=prefix,
                    ),
                ),
                material=material,
                expected_uv_scalar_count=uv_scalar_count,
            )
        )
    _assert(len(fixtures) == _OBJECT_COUNT, "multi fixture object count changed")
    return tuple(fixtures)


def _multi_settings(
    output_directory: Path,
    case: _Case,
) -> A1MultiObjectExportSettings:
    return A1MultiObjectExportSettings(
        output_directory=output_directory,
        output_stem=f"{case.key}Multi",
        mode=A1MultiObjectMode.STANDALONE,
        namespace_animations=True,
    )


def _prepare_scene() -> object:
    _clear_scene()
    _purge_orphan_scene_data()
    _configure_scene()
    _create_camera()
    sentinel = _create_sentinel()
    sentinel.location = (12.0, 12.0, 12.0)
    sentinel.hide_render = True
    _activate_only(sentinel)
    return sentinel


def _image_signature(pixels: tuple[float, ...]) -> tuple[float, float, float, float]:
    red = green = blue = alpha_sum = 0.0
    visible = 0
    for offset in range(0, len(pixels), 4):
        alpha = float(pixels[offset + 3])
        if alpha <= 0.01:
            continue
        red += float(pixels[offset]) * alpha
        green += float(pixels[offset + 1]) * alpha
        blue += float(pixels[offset + 2]) * alpha
        alpha_sum += alpha
        visible += 1
    _assert(visible > 0 and alpha_sum > 0.0, "rendered sequence frame is empty")
    return (
        red / alpha_sum,
        green / alpha_sum,
        blue / alpha_sum,
        float(visible),
    )


def _read_image_summary(path: Path) -> _ImageSummary:
    _assert(path.is_file(), f"missing PNG output: {path}")
    content = path.read_bytes()
    _assert(content[:8] == PNG_SIGNATURE, f"invalid PNG signature: {path}")
    size, pixels = _read_image(path)
    _assert(
        len(pixels) == size[0] * size[1] * 4,
        f"invalid RGBA buffer length for {path}: {len(pixels)}",
    )
    return _ImageSummary(
        path=path,
        size=size,
        signature=_image_signature(pixels),
        content=content,
    )


def _assert_frame_difference(
    first: _ImageSummary,
    second: _ImageSummary,
    *,
    label: str,
) -> None:
    _assert(first.content != second.content, f"{label} frame PNG bytes are identical")
    delta = sum(
        abs(left - right)
        for left, right in zip(
            first.signature[:3],
            second.signature[:3],
            strict=True,
        )
    )
    visible_delta = abs(first.signature[3] - second.signature[3])
    _assert(
        delta > 0.05 or visible_delta > 4.0,
        f"{label} frames are not visually distinct; rgb_delta={delta}, "
        f"visible_delta={visible_delta}",
    )


def _texture_paths_by_source(
    output_files: tuple[Path, ...],
    fixtures: tuple[_SourceFixture, ...],
) -> dict[str, tuple[Path, ...]]:
    png_paths = tuple(path for path in output_files if path.suffix.casefold() == ".png")
    expected_count = _OBJECT_COUNT * _SEQUENCE_FRAME_COUNT
    _assert(
        len(png_paths) == expected_count,
        f"expected {expected_count} PNGs, received {len(png_paths)}: {png_paths}",
    )
    grouped: dict[str, tuple[Path, ...]] = {}
    for fixture in fixtures:
        prefix = f"{fixture.output_stem}_Baked_"
        matches = tuple(sorted(path for path in png_paths if path.name.startswith(prefix)))
        _assert(
            len(matches) == _SEQUENCE_FRAME_COUNT,
            f"{fixture.output_stem} expected {_SEQUENCE_FRAME_COUNT} frames, got {matches}",
        )
        grouped[fixture.prefix] = matches
    flattened = tuple(path for paths in grouped.values() for path in paths)
    _assert(
        len(flattened) == len(set(flattened)) == expected_count,
        "multi-object sequence paths overlap or were not assigned to one source",
    )
    return grouped


def _validate_texture_outputs(
    paths_by_prefix: Mapping[str, tuple[Path, ...]],
    case: _Case,
) -> dict[str, tuple[int, int]]:
    dimensions: dict[str, tuple[int, int]] = {}
    all_contents: list[bytes] = []
    for prefix, paths in paths_by_prefix.items():
        summaries = tuple(_read_image_summary(path) for path in paths)
        _assert_frame_difference(
            summaries[0],
            summaries[1],
            label=f"{case.key}/{prefix}",
        )
        sizes = tuple(summary.size for summary in summaries)
        if case.texture_mode is A1TextureExportMode.NORMAL_UV_SEGMENTS:
            _assert(
                sizes == ((_TEXTURE_SIZE, _TEXTURE_SIZE),) * _SEQUENCE_FRAME_COUNT,
                f"Normal UV output size changed for {prefix}: {sizes}",
            )
        else:
            _assert(
                len(set(sizes)) == 1,
                f"Camera Projection sequence crop differs by frame for {prefix}: {sizes}",
            )
            width, height = sizes[0]
            _assert(
                1 <= width <= _TEXTURE_SIZE and 1 <= height <= _TEXTURE_SIZE,
                f"Camera Projection crop outside 1..{_TEXTURE_SIZE}: {sizes[0]}",
            )
        dimensions[prefix] = sizes[0]
        all_contents.extend(summary.content for summary in summaries)

    _assert(
        len(all_contents) == len({content for content in all_contents}),
        "two objects or sequence frames produced duplicate PNG payloads",
    )
    return dimensions


def _json_array(document: Mapping[str, object], field_name: str) -> list[object]:
    value = document.get(field_name, [])
    _assert(isinstance(value, list), f"{field_name} must be a JSON array")
    return value


def _visual_slots(
    document: Mapping[str, object],
    fixtures: tuple[_SourceFixture, ...],
) -> tuple[dict[str, object], ...]:
    prefixes = tuple(fixture.prefix for fixture in fixtures)
    result = tuple(
        slot
        for slot in _json_array(document, "slots")
        if isinstance(slot, dict)
        and isinstance(slot.get("name"), str)
        and isinstance(slot.get("attachment"), str)
        and str(slot["name"]).startswith(prefixes)
    )
    _assert(
        len(result) == _OBJECT_COUNT,
        f"expected {_OBJECT_COUNT} visual slots, got {tuple(slot.get('name') for slot in result)}",
    )
    return result


def _skin_slot_attachments(
    document: Mapping[str, object],
    slot_name: str,
) -> dict[str, object]:
    matches: list[dict[str, object]] = []
    for skin in _json_array(document, "skins"):
        if not isinstance(skin, dict):
            continue
        attachments = skin.get("attachments")
        if not isinstance(attachments, dict):
            continue
        slot_attachments = attachments.get(slot_name)
        if isinstance(slot_attachments, dict):
            matches.append(slot_attachments)
    _assert(
        len(matches) == 1,
        f"expected one skin attachment map for slot {slot_name!r}, got {len(matches)}",
    )
    return matches[0]


def _contains_key(value: object, key: str) -> bool:
    if isinstance(value, dict):
        return key in value or any(_contains_key(child, key) for child in value.values())
    if isinstance(value, list):
        return any(_contains_key(child, key) for child in value)
    return False


def _collect_lists_for_key(value: object, key: str) -> tuple[list[object], ...]:
    result: list[list[object]] = []

    def visit(current: object) -> None:
        if isinstance(current, dict):
            candidate = current.get(key)
            if isinstance(candidate, list):
                result.append(candidate)
            for child in current.values():
                visit(child)
        elif isinstance(current, list):
            for child in current:
                visit(child)

    visit(value)
    return tuple(result)


def _fixture_for_slot(
    slot_name: str,
    fixtures: tuple[_SourceFixture, ...],
) -> _SourceFixture:
    matches = tuple(fixture for fixture in fixtures if slot_name.startswith(fixture.prefix))
    _assert(len(matches) == 1, f"slot {slot_name!r} does not map to one source prefix")
    return matches[0]


def _assert_normal_mesh_attachment(
    attachment: Mapping[str, object],
    fixture: _SourceFixture,
    *,
    label: str,
) -> None:
    _assert(attachment.get("type") == "mesh", f"{label} is not a mesh attachment")
    uvs = attachment.get("uvs")
    _assert(isinstance(uvs, list), f"{label}.uvs must be a list")
    _assert(
        len(uvs) == fixture.expected_uv_scalar_count,
        f"{label} changed source topology: expected {fixture.expected_uv_scalar_count} "
        f"UV scalars, got {len(uvs)}",
    )


def _assert_projection_mesh_attachment(
    attachment: Mapping[str, object],
    image_size: tuple[int, int],
    *,
    label: str,
) -> None:
    _assert(attachment.get("type") == "mesh", f"{label} is not a mesh attachment")
    uvs = attachment.get("uvs")
    triangles = attachment.get("triangles")
    _assert(isinstance(uvs, list) and len(uvs) >= 6, f"{label}.uvs is degenerate")
    _assert(len(uvs) % 2 == 0, f"{label}.uvs does not contain coordinate pairs")
    _assert(
        all(0.0 <= float(value) <= 1.0 for value in uvs),
        f"{label}.uvs contains values outside 0..1",
    )
    _assert(isinstance(triangles, list) and len(triangles) >= 3, f"{label} has no triangles")
    vertex_count = len(uvs) // 2
    _assert(
        all(0 <= int(index) < vertex_count for index in triangles),
        f"{label}.triangles references a missing vertex",
    )
    hull = int(attachment.get("hull", 0))
    _assert(3 <= hull <= vertex_count, f"{label}.hull is invalid: {hull}")
    _assert(
        int(round(float(attachment.get("width", 0.0)))) == image_size[0],
        f"{label}.width does not match cropped PNG width {image_size[0]}",
    )
    _assert(
        int(round(float(attachment.get("height", 0.0)))) == image_size[1],
        f"{label}.height does not match cropped PNG height {image_size[1]}",
    )


def _assert_sequence_encoding(
    document: Mapping[str, object],
    case: _Case,
    fixtures: tuple[_SourceFixture, ...],
    image_dimensions: Mapping[str, tuple[int, int]],
) -> None:
    slots = _visual_slots(document, fixtures)
    native = (
        case.target.texture_animation_encoding
        is SpineTextureAnimationEncoding.NATIVE_SEQUENCE
    )

    for slot in slots:
        slot_name = str(slot["name"])
        setup_attachment = str(slot["attachment"])
        fixture = _fixture_for_slot(slot_name, fixtures)
        attachments = _skin_slot_attachments(document, slot_name)
        expected_attachment_count = 1 if native else _SEQUENCE_FRAME_COUNT
        _assert(
            len(attachments) == expected_attachment_count,
            f"{case.key}/{slot_name} expected {expected_attachment_count} attachments, "
            f"got {tuple(attachments)}",
        )
        _assert(
            setup_attachment in attachments,
            f"setup attachment {setup_attachment!r} missing for slot {slot_name!r}",
        )

        for attachment_name, raw_attachment in attachments.items():
            _assert(
                isinstance(raw_attachment, dict),
                f"{case.key}/{slot_name}/{attachment_name} must be an object",
            )
            if case.texture_mode is A1TextureExportMode.NORMAL_UV_SEGMENTS:
                _assert_normal_mesh_attachment(
                    raw_attachment,
                    fixture,
                    label=f"{case.key}/{slot_name}/{attachment_name}",
                )
            else:
                _assert_projection_mesh_attachment(
                    raw_attachment,
                    image_dimensions[fixture.prefix],
                    label=f"{case.key}/{slot_name}/{attachment_name}",
                )

        if native:
            sequence = attachments[setup_attachment].get("sequence")
            _assert(
                isinstance(sequence, dict),
                f"native sequence metadata missing for {slot_name!r}",
            )
            _assert(
                sequence.get("count") == _SEQUENCE_FRAME_COUNT,
                f"native sequence count is wrong for {slot_name!r}: {sequence}",
            )

    animations = document.get("animations", {})
    _assert(isinstance(animations, dict), "animations must be a JSON object")
    if native:
        sequence_timelines = _collect_lists_for_key(animations, "sequence")
        _assert(
            len(sequence_timelines) == _OBJECT_COUNT,
            f"expected {_OBJECT_COUNT} native sequence timelines, got {len(sequence_timelines)}",
        )
        for timeline in sequence_timelines:
            _assert(len(timeline) == 2, f"native Loop timeline must have two keys: {timeline}")
            _assert(
                isinstance(timeline[0], dict) and timeline[0].get("mode") == "loop",
                f"native sequence is not Loop: {timeline}",
            )
        return

    _assert(
        not _contains_key(document, "sequence"),
        f"native sequence metadata leaked into {case.target.label}",
    )
    attachment_timelines = _collect_lists_for_key(animations, "attachment")
    _assert(
        len(attachment_timelines) == _OBJECT_COUNT,
        f"expected {_OBJECT_COUNT} legacy attachment timelines, "
        f"got {len(attachment_timelines)}",
    )
    all_attachment_names = {
        attachment_name
        for slot in slots
        for attachment_name in _skin_slot_attachments(document, str(slot["name"]))
    }
    for timeline in attachment_timelines:
        _assert(
            len(timeline) == _SEQUENCE_FRAME_COUNT + 1,
            f"legacy Loop timeline must contain two frames plus wrap: {timeline}",
        )
        names = tuple(
            item.get("name") if isinstance(item, dict) else None
            for item in timeline
        )
        _assert(
            len(set(names[:-1])) == _SEQUENCE_FRAME_COUNT,
            f"legacy frame attachment names are not unique: {names}",
        )
        _assert(names[-1] == names[0], f"legacy Loop does not wrap: {names}")
        _assert(
            all(name in all_attachment_names for name in names if isinstance(name, str)),
            f"legacy timeline references an unknown attachment: {names}",
        )


def _assert_document(
    document: dict[str, object],
    case: _Case,
    fixtures: tuple[_SourceFixture, ...],
    image_dimensions: Mapping[str, tuple[int, int]],
) -> None:
    skeleton = document.get("skeleton")
    _assert(isinstance(skeleton, dict), "skeleton metadata is missing")
    _assert(
        skeleton.get("spine") == case.target.exact_version,
        f"{case.key} version mismatch: {skeleton.get('spine')!r}",
    )
    _assert_bone_schema(document, case.target)
    _assert_constraint_schema(document, case.target)

    bone_names = {
        str(item.get("name"))
        for item in _json_array(document, "bones")
        if isinstance(item, dict)
    }
    for fixture in fixtures:
        _assert(
            f"{fixture.prefix}_main" in bone_names,
            f"{case.key} lost main bone for {fixture.prefix}",
        )
    _assert(
        not any(name.startswith("all_objects") for name in bone_names),
        f"{case.key} standalone multi export unexpectedly created connected wrapper",
    )
    _assert_sequence_encoding(
        document,
        case,
        fixtures,
        image_dimensions,
    )


def _assert_state_restored(
    *,
    context_before: object,
    scene_before: object,
    render_before: object,
    fixtures: tuple[_SourceFixture, ...],
    material_fingerprints: tuple[object, ...],
    analysis_matrices: tuple[tuple[float, ...], ...],
) -> None:
    _assert(_capture_context() == context_before, "multi export changed Blender context")
    _assert(
        _capture_scene_bake_state() == scene_before,
        "multi export changed Scene bake settings",
    )
    _assert(
        _scene_render_fingerprint() == render_before,
        "multi export changed Camera Projection render state",
    )
    _assert(
        tuple(_material_fingerprint(item.material) for item in fixtures)
        == material_fingerprints,
        "multi export mutated source materials",
    )
    bpy.context.view_layer.update()
    for fixture, expected_matrix in zip(fixtures, analysis_matrices, strict=True):
        actual_matrix = _matrix_tuple(fixture.object)
        _assert(
            _matrices_close(actual_matrix, expected_matrix),
            f"multi export did not restore matrix_world for {fixture.object.name}",
        )
    _assert(
        not _temporary_datablock_names(),
        "multi export leaked temporary Blender datablocks",
    )


def _run_case(output_root: Path, case: _Case) -> None:
    case_directory = (output_root / case.key).resolve(strict=False)
    case_directory.mkdir(parents=True, exist_ok=False)
    _prepare_scene()
    fixtures = _build_sources(case_directory, case)
    sentinel = _create_sentinel()
    sentinel.name = f"{case.key}_Sentinel"
    sentinel.location = (12.0, 12.0, 12.0)
    sentinel.hide_render = True
    _activate_only(sentinel)
    for fixture in fixtures:
        fixture.object.select_set(False)

    scene = bpy.context.scene
    scene.frame_set(_ANALYSIS_FRAME)
    bpy.context.view_layer.update()
    analysis_matrices = tuple(_matrix_tuple(fixture.object) for fixture in fixtures)
    for fixture, analysis_matrix in zip(fixtures, analysis_matrices, strict=True):
        frame_matrices = _frame_matrices(scene, fixture.object)
        _assert(
            not _matrices_close(frame_matrices[0], frame_matrices[1]),
            f"{case.key}/{fixture.prefix} bake frames have identical matrix_world",
        )
        _assert(
            all(not _matrices_close(matrix, analysis_matrix) for matrix in frame_matrices),
            f"{case.key}/{fixture.prefix} analysis matrix matches a bake frame",
        )

    context_before = _capture_context()
    scene_before = _capture_scene_bake_state()
    render_before = _scene_render_fingerprint()
    material_fingerprints = tuple(
        _material_fingerprint(fixture.material) for fixture in fixtures
    )

    result = export_a1_multi_object(
        tuple(fixture.source for fixture in fixtures),
        _multi_settings(case_directory, case),
        context=bpy.context,
        scene=scene,
    )
    _assert(result.success, f"{case.key} export failed: {result.issues}")
    _assert(
        len(result.output_files) == 1 + _OBJECT_COUNT * _SEQUENCE_FRAME_COUNT,
        f"{case.key} output count is wrong: {result.output_files}",
    )
    json_path = result.output_files[0]
    _assert(json_path.suffix.casefold() == ".json", f"JSON is not first: {json_path}")
    _assert(json_path.is_file(), f"missing JSON output: {json_path}")

    paths_by_prefix = _texture_paths_by_source(result.output_files, fixtures)
    image_dimensions = _validate_texture_outputs(paths_by_prefix, case)
    document = json.loads(json_path.read_text(encoding="utf-8"))
    _assert(isinstance(document, dict), f"{case.key} JSON root must be an object")
    _assert_document(document, case, fixtures, image_dimensions)
    _assert(
        result.statistics.get("object_count") == _OBJECT_COUNT,
        f"{case.key} statistics lost object count: {result.statistics}",
    )
    _assert_state_restored(
        context_before=context_before,
        scene_before=scene_before,
        render_before=render_before,
        fixtures=fixtures,
        material_fingerprints=material_fingerprints,
        analysis_matrices=analysis_matrices,
    )


def _cases() -> tuple[_Case, ...]:
    cases = tuple(
        _Case(target=target, texture_mode=texture_mode)
        for target in _TARGETS
        for texture_mode in _TEXTURE_MODES
    )
    _assert(len(cases) == 10, f"multi-object sequence matrix must contain 10 cases: {cases}")
    return cases


def main() -> None:
    failures: list[tuple[str, str]] = []
    cases = _cases()
    started = time.perf_counter()
    print(f"Blender version: {bpy.app.version_string}")
    print(
        "[MULTI-SEQUENCE-MATRIX] "
        f"cases={len(cases)} objects={_OBJECT_COUNT} frames={_SEQUENCE_FRAME_COUNT} "
        f"texture={_TEXTURE_SIZE}x{_TEXTURE_SIZE}"
    )

    with tempfile.TemporaryDirectory(prefix="spine2d-multi-sequence-matrix-") as directory:
        output_root = Path(directory)
        for case_index, case in enumerate(cases, start=1):
            case_started = time.perf_counter()
            print(
                f"[MULTI-SEQUENCE-MATRIX] RUN {case_index}/{len(cases)} "
                f"{case.key}"
            )
            try:
                _run_case(output_root, case)
            except Exception:
                failures.append((case.key, traceback.format_exc()))
                print(f"[MULTI-SEQUENCE-MATRIX] FAIL {case.key}")
            else:
                elapsed = time.perf_counter() - case_started
                print(f"[MULTI-SEQUENCE-MATRIX] PASS {case.key} ({elapsed:.2f}s)")
            finally:
                _clear_scene()
                _purge_orphan_scene_data()

    if failures:
        for case_key, details in failures:
            print(f"\n--- {case_key} ---\n{details}")
        raise SystemExit(1)

    elapsed = time.perf_counter() - started
    print(
        f"[MULTI-SEQUENCE-MATRIX] PASS {len(cases)} cases "
        f"({elapsed:.2f}s total)"
    )


if __name__ == "__main__":
    main()
