"""Real Blender 5.2 multi-object sequence matrix for Spine 3.8 through 4.3.

Every case calls only the public ``export_a1_multi_object`` service and exports two
animated mesh objects for two frames.  Texture size is fixed at 128x128 and Cycles uses
one sample so the full matrix remains practical for local validation.

Matrix:
    5 Spine targets x 2 texture modes x 2 objects x 2 frames.

Spine 3.8/4.0 must serialize attachment swaps.  Spine 4.1/4.2/4.3 must serialize
native sequence metadata and Loop timelines.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys
import tempfile
import time
import traceback
from typing import Mapping

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
from run_normal_uv_camera_context_sequence_integration import (  # noqa: E402
    _create_animated_camera_reflection_material,
    _matrices_equal_at_float32_precision,
    _matrix_tuple,
)


_TEXTURE_SIZE = 128
_SEQUENCE_START_FRAME = 1
_SEQUENCE_FRAME_COUNT = 2
_ANALYSIS_FRAME = 19
_OBJECT_COUNT = 2

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
class _Fixture:
    source: A1MultiObjectSource
    material: object
    expected_normal_uv_scalars: int

    @property
    def obj(self) -> object:
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
    rgb: tuple[float, float, float]
    visible_pixels: int
    file_bytes: bytes


def _cases() -> tuple[_Case, ...]:
    result = tuple(
        _Case(target=target, texture_mode=texture_mode)
        for target in _TARGETS
        for texture_mode in _TEXTURE_MODES
    )
    _assert(len(result) == 10, f"matrix must contain 10 cases, got {result}")
    return result


def _create_pentagon(name: str):
    return _create_mesh_object(
        name,
        (
            (-0.72, -0.55, 0.0),
            (0.10, -0.80, 0.0),
            (0.82, -0.22, 0.0),
            (0.52, 0.72, 0.0),
            (-0.58, 0.74, 0.0),
        ),
        ((0, 1, 2, 3, 4),),
    )


def _create_quad(name: str):
    return _create_mesh_object(
        name,
        (
            (-0.68, -0.60, 0.0),
            (0.76, -0.48, 0.0),
            (0.58, 0.68, 0.0),
            (-0.64, 0.72, 0.0),
        ),
        ((0, 1, 2, 3),),
    )


def _animate_transform(obj: object, *, variant: int) -> None:
    """Create two distinct bake matrices and one distinct planning matrix."""

    base_x = -0.82 if variant == 0 else 0.82
    transforms = (
        (
            1,
            (base_x - 0.16, -0.16 + variant * 0.18, variant * 0.16),
            (0.12 + variant * 0.08, -0.22, -0.20 + variant * 0.36),
            (0.80 + variant * 0.08, 1.02 - variant * 0.12, 1.0),
        ),
        (
            2,
            (base_x + 0.22, 0.24 - variant * 0.12, 0.26 - variant * 0.10),
            (0.34, 0.16 - variant * 0.08, 0.38 + variant * 0.24),
            (1.08 - variant * 0.12, 0.76 + variant * 0.18, 1.0),
        ),
        (
            _ANALYSIS_FRAME,
            (base_x, -0.44 + variant * 0.16, -0.16 + variant * 0.32),
            (0.62 - variant * 0.18, -0.56 + variant * 0.22, 0.88 - variant * 0.20),
            (1.20 - variant * 0.12, 0.88 + variant * 0.10, 1.0),
        ),
    )
    for frame, location, rotation, scale in transforms:
        obj.location = location
        obj.rotation_euler = rotation
        obj.scale = scale
        obj.keyframe_insert(data_path="location", frame=frame)
        obj.keyframe_insert(data_path="rotation_euler", frame=frame)
        obj.keyframe_insert(data_path="scale", frame=frame)


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


def _build_fixtures(
    output_directory: Path,
    case: _Case,
) -> tuple[_Fixture, ...]:
    geometry = (
        (_create_pentagon, 10),
        (_create_quad, 8),
    )
    fixtures: list[_Fixture] = []
    for index, (builder, uv_scalar_count) in enumerate(geometry):
        suffix = chr(ord("A") + index)
        prefix = f"{case.key}Object{suffix}"
        obj = builder(f"{prefix}_Source")
        material = _create_animated_camera_reflection_material(
            f"{prefix}_Material"
        )
        obj.data.materials.append(material)
        _animate_transform(obj, variant=index)
        fixtures.append(
            _Fixture(
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
                expected_normal_uv_scalars=uv_scalar_count,
            )
        )
    _assert(len(fixtures) == _OBJECT_COUNT, "fixture object count changed")
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
    sentinel.name = "MultiSequenceMatrixSentinel"
    sentinel.location = (12.0, 12.0, 12.0)
    sentinel.hide_render = True
    _activate_only(sentinel)
    return sentinel


def _frame_matrices(scene: object, obj: object) -> tuple[tuple[float, ...], ...]:
    original_frame = int(scene.frame_current)
    try:
        result = []
        for frame in range(
            _SEQUENCE_START_FRAME,
            _SEQUENCE_START_FRAME + _SEQUENCE_FRAME_COUNT,
        ):
            scene.frame_set(frame)
            bpy.context.view_layer.update()
            result.append(_matrix_tuple(obj))
        return tuple(result)
    finally:
        scene.frame_set(original_frame)
        bpy.context.view_layer.update()


def _image_summary(path: Path) -> _ImageSummary:
    _assert(path.is_file(), f"missing PNG: {path}")
    file_bytes = path.read_bytes()
    _assert(file_bytes[:8] == PNG_SIGNATURE, f"invalid PNG signature: {path}")
    size, pixels = _read_image(path)
    _assert(
        len(pixels) == size[0] * size[1] * 4,
        f"invalid RGBA buffer size for {path}",
    )

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
    _assert(visible > 0 and alpha_sum > 0.0, f"empty PNG: {path}")
    return _ImageSummary(
        path=path,
        size=size,
        rgb=(red / alpha_sum, green / alpha_sum, blue / alpha_sum),
        visible_pixels=visible,
        file_bytes=file_bytes,
    )


def _assert_distinct_frames(
    first: _ImageSummary,
    second: _ImageSummary,
    *,
    label: str,
) -> None:
    _assert(first.file_bytes != second.file_bytes, f"{label} PNG bytes are identical")
    rgb_delta = sum(
        abs(left - right)
        for left, right in zip(first.rgb, second.rgb, strict=True)
    )
    visible_delta = abs(first.visible_pixels - second.visible_pixels)
    _assert(
        rgb_delta > 0.05 or visible_delta > 4,
        f"{label} frames are not visually distinct; "
        f"rgb_delta={rgb_delta}, visible_delta={visible_delta}",
    )


def _texture_groups(
    output_files: tuple[Path, ...],
    fixtures: tuple[_Fixture, ...],
) -> dict[str, tuple[Path, ...]]:
    png_paths = tuple(path for path in output_files if path.suffix.casefold() == ".png")
    expected_count = _OBJECT_COUNT * _SEQUENCE_FRAME_COUNT
    _assert(
        len(png_paths) == expected_count,
        f"expected {expected_count} PNGs, got {png_paths}",
    )
    result: dict[str, tuple[Path, ...]] = {}
    for fixture in fixtures:
        filename_prefix = f"{fixture.output_stem}_Baked_"
        matches = tuple(
            sorted(path for path in png_paths if path.name.startswith(filename_prefix))
        )
        _assert(
            len(matches) == _SEQUENCE_FRAME_COUNT,
            f"{fixture.prefix} expected two PNGs, got {matches}",
        )
        result[fixture.prefix] = matches
    flattened = tuple(path for group in result.values() for path in group)
    _assert(
        len(flattened) == len(set(flattened)) == expected_count,
        "PNG paths overlap between multi-object sources",
    )
    return result


def _validate_images(
    groups: Mapping[str, tuple[Path, ...]],
    case: _Case,
) -> dict[str, tuple[int, int]]:
    dimensions: dict[str, tuple[int, int]] = {}
    all_payloads: list[bytes] = []
    for prefix, paths in groups.items():
        frames = tuple(_image_summary(path) for path in paths)
        _assert_distinct_frames(frames[0], frames[1], label=f"{case.key}/{prefix}")
        sizes = tuple(frame.size for frame in frames)
        if case.texture_mode is A1TextureExportMode.NORMAL_UV_SEGMENTS:
            _assert(
                sizes == ((_TEXTURE_SIZE, _TEXTURE_SIZE),) * _SEQUENCE_FRAME_COUNT,
                f"Normal UV size mismatch for {prefix}: {sizes}",
            )
        else:
            _assert(
                len(set(sizes)) == 1,
                f"Camera Projection crop changed between frames for {prefix}: {sizes}",
            )
            width, height = sizes[0]
            _assert(
                1 <= width <= _TEXTURE_SIZE and 1 <= height <= _TEXTURE_SIZE,
                f"Camera Projection crop outside 1..{_TEXTURE_SIZE}: {sizes[0]}",
            )
        dimensions[prefix] = sizes[0]
        all_payloads.extend(frame.file_bytes for frame in frames)

    _assert(
        len(all_payloads) == len({payload for payload in all_payloads}),
        "objects or frames produced duplicate PNG payloads",
    )
    return dimensions


def _json_array(document: Mapping[str, object], field_name: str) -> list[object]:
    value = document.get(field_name, [])
    _assert(isinstance(value, list), f"{field_name} must be a JSON array")
    return value


def _visual_slots(
    document: Mapping[str, object],
    fixtures: tuple[_Fixture, ...],
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
        f"expected two visual slots, got {tuple(item.get('name') for item in result)}",
    )
    return result


def _slot_attachments(
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
        value = attachments.get(slot_name)
        if isinstance(value, dict):
            matches.append(value)
    _assert(len(matches) == 1, f"slot {slot_name!r} has {len(matches)} skin maps")
    return matches[0]


def _fixture_for_slot(slot_name: str, fixtures: tuple[_Fixture, ...]) -> _Fixture:
    matches = tuple(fixture for fixture in fixtures if slot_name.startswith(fixture.prefix))
    _assert(len(matches) == 1, f"slot {slot_name!r} does not map to one source")
    return matches[0]


def _contains_key(value: object, key: str) -> bool:
    if isinstance(value, dict):
        return key in value or any(_contains_key(child, key) for child in value.values())
    if isinstance(value, list):
        return any(_contains_key(child, key) for child in value)
    return False


def _lists_for_key(value: object, key: str) -> tuple[list[object], ...]:
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


def _assert_normal_attachment(
    attachment: Mapping[str, object],
    fixture: _Fixture,
    *,
    label: str,
) -> None:
    _assert(attachment.get("type") == "mesh", f"{label} is not mesh")
    uvs = attachment.get("uvs")
    _assert(isinstance(uvs, list), f"{label}.uvs must be a list")
    _assert(
        len(uvs) == fixture.expected_normal_uv_scalars,
        f"{label} changed Normal UV topology: {len(uvs)}",
    )


def _assert_projection_attachment(
    attachment: Mapping[str, object],
    image_size: tuple[int, int],
    *,
    label: str,
) -> None:
    _assert(attachment.get("type") == "mesh", f"{label} is not mesh")
    uvs = attachment.get("uvs")
    triangles = attachment.get("triangles")
    _assert(isinstance(uvs, list) and len(uvs) >= 6, f"{label}.uvs is degenerate")
    _assert(len(uvs) % 2 == 0, f"{label}.uvs is not paired")
    _assert(
        all(0.0 <= float(value) <= 1.0 for value in uvs),
        f"{label}.uvs outside 0..1",
    )
    vertex_count = len(uvs) // 2
    _assert(
        isinstance(triangles, list) and len(triangles) >= 3,
        f"{label}.triangles is empty",
    )
    _assert(
        all(0 <= int(index) < vertex_count for index in triangles),
        f"{label}.triangles references missing vertices",
    )
    hull = int(attachment.get("hull", 0))
    _assert(3 <= hull <= vertex_count, f"{label}.hull is invalid: {hull}")
    _assert(
        int(round(float(attachment.get("width", 0.0)))) == image_size[0],
        f"{label}.width does not match PNG {image_size[0]}",
    )
    _assert(
        int(round(float(attachment.get("height", 0.0)))) == image_size[1],
        f"{label}.height does not match PNG {image_size[1]}",
    )


def _assert_sequence_encoding(
    document: Mapping[str, object],
    case: _Case,
    fixtures: tuple[_Fixture, ...],
    dimensions: Mapping[str, tuple[int, int]],
) -> None:
    slots = _visual_slots(document, fixtures)
    native = (
        case.target.texture_animation_encoding
        is SpineTextureAnimationEncoding.NATIVE_SEQUENCE
    )

    all_attachment_names: set[str] = set()
    for slot in slots:
        slot_name = str(slot["name"])
        setup_name = str(slot["attachment"])
        fixture = _fixture_for_slot(slot_name, fixtures)
        attachments = _slot_attachments(document, slot_name)
        all_attachment_names.update(attachments)
        expected_count = 1 if native else _SEQUENCE_FRAME_COUNT
        _assert(
            len(attachments) == expected_count,
            f"{case.key}/{slot_name} expected {expected_count} attachments, got {tuple(attachments)}",
        )
        _assert(setup_name in attachments, f"setup attachment missing for {slot_name}")

        for attachment_name, raw_attachment in attachments.items():
            _assert(isinstance(raw_attachment, dict), f"{attachment_name} is not an object")
            label = f"{case.key}/{slot_name}/{attachment_name}"
            if case.texture_mode is A1TextureExportMode.NORMAL_UV_SEGMENTS:
                _assert_normal_attachment(raw_attachment, fixture, label=label)
            else:
                _assert_projection_attachment(
                    raw_attachment,
                    dimensions[fixture.prefix],
                    label=label,
                )

        if native:
            sequence = attachments[setup_name].get("sequence")
            _assert(isinstance(sequence, dict), f"native sequence missing for {slot_name}")
            _assert(
                sequence.get("count") == _SEQUENCE_FRAME_COUNT,
                f"native sequence count is wrong for {slot_name}: {sequence}",
            )

    animations = document.get("animations", {})
    _assert(isinstance(animations, dict), "animations must be an object")
    if native:
        timelines = _lists_for_key(animations, "sequence")
        _assert(len(timelines) == _OBJECT_COUNT, f"expected two native timelines: {timelines}")
        for timeline in timelines:
            _assert(len(timeline) == 2, f"native Loop timeline is invalid: {timeline}")
            _assert(
                isinstance(timeline[0], dict) and timeline[0].get("mode") == "loop",
                f"native timeline is not Loop: {timeline}",
            )
        return

    _assert(
        not _contains_key(document, "sequence"),
        f"native sequence leaked into {case.target.label}",
    )
    timelines = _lists_for_key(animations, "attachment")
    _assert(len(timelines) == _OBJECT_COUNT, f"expected two legacy timelines: {timelines}")
    for timeline in timelines:
        _assert(
            len(timeline) == _SEQUENCE_FRAME_COUNT + 1,
            f"legacy timeline must contain two frames and one wrap key: {timeline}",
        )
        names = tuple(
            item.get("name") if isinstance(item, dict) else None
            for item in timeline
        )
        _assert(len(set(names[:-1])) == 2, f"legacy names are not unique: {names}")
        _assert(names[-1] == names[0], f"legacy Loop does not wrap: {names}")
        _assert(
            all(name in all_attachment_names for name in names if isinstance(name, str)),
            f"legacy timeline references unknown attachments: {names}",
        )


def _assert_document(
    document: dict[str, object],
    case: _Case,
    fixtures: tuple[_Fixture, ...],
    dimensions: Mapping[str, tuple[int, int]],
) -> None:
    skeleton = document.get("skeleton")
    _assert(isinstance(skeleton, dict), "skeleton metadata missing")
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
            f"{case.key} lost {fixture.prefix}_main",
        )
    _assert(
        not any(name.startswith("all_objects") for name in bone_names),
        f"{case.key} standalone export created connected wrapper",
    )
    _assert_sequence_encoding(document, case, fixtures, dimensions)


def _assert_state_restored(
    *,
    context_before: object,
    bake_state_before: object,
    render_state_before: object,
    fixtures: tuple[_Fixture, ...],
    material_fingerprints: tuple[object, ...],
    analysis_matrices: tuple[tuple[float, ...], ...],
) -> None:
    _assert(_capture_context() == context_before, "export changed Blender context")
    _assert(
        _capture_scene_bake_state() == bake_state_before,
        "export changed Scene bake state",
    )
    _assert(
        _scene_render_fingerprint() == render_state_before,
        "export changed render or camera state",
    )
    _assert(
        tuple(_material_fingerprint(fixture.material) for fixture in fixtures)
        == material_fingerprints,
        "export mutated source materials",
    )
    bpy.context.view_layer.update()
    for fixture, expected in zip(fixtures, analysis_matrices, strict=True):
        _assert(
            _matrices_equal_at_float32_precision(_matrix_tuple(fixture.obj), expected),
            f"export did not restore matrix_world for {fixture.obj.name}",
        )
    _assert(not _temporary_datablock_names(), "export leaked temporary datablocks")


def _run_case(output_root: Path, case: _Case) -> None:
    case_directory = output_root / case.key
    case_directory.mkdir(parents=True, exist_ok=False)
    sentinel = _prepare_scene()
    fixtures = _build_fixtures(case_directory, case)
    _activate_only(sentinel)
    for fixture in fixtures:
        fixture.obj.select_set(False)

    scene = bpy.context.scene
    scene.frame_set(_ANALYSIS_FRAME)
    bpy.context.view_layer.update()
    analysis_matrices = tuple(_matrix_tuple(fixture.obj) for fixture in fixtures)
    for fixture, analysis_matrix in zip(fixtures, analysis_matrices, strict=True):
        frame_matrices = _frame_matrices(scene, fixture.obj)
        _assert(
            not _matrices_equal_at_float32_precision(frame_matrices[0], frame_matrices[1]),
            f"{case.key}/{fixture.prefix} bake matrices are identical",
        )
        _assert(
            all(
                not _matrices_equal_at_float32_precision(matrix, analysis_matrix)
                for matrix in frame_matrices
            ),
            f"{case.key}/{fixture.prefix} planning matrix matches a bake frame",
        )

    context_before = _capture_context()
    bake_state_before = _capture_scene_bake_state()
    render_state_before = _scene_render_fingerprint()
    material_fingerprints = tuple(
        _material_fingerprint(fixture.material) for fixture in fixtures
    )

    result = export_a1_multi_object(
        tuple(fixture.source for fixture in fixtures),
        _multi_settings(case_directory, case),
        context=bpy.context,
        scene=scene,
    )
    _assert(result.success, f"{case.key} failed: {result.issues}")
    _assert(
        len(result.output_files) == 1 + _OBJECT_COUNT * _SEQUENCE_FRAME_COUNT,
        f"{case.key} output count is wrong: {result.output_files}",
    )

    json_path = result.output_files[0]
    _assert(json_path.suffix.casefold() == ".json", f"JSON is not first: {json_path}")
    _assert(json_path.is_file(), f"missing JSON: {json_path}")
    groups = _texture_groups(result.output_files, fixtures)
    dimensions = _validate_images(groups, case)
    document = json.loads(json_path.read_text(encoding="utf-8"))
    _assert(isinstance(document, dict), f"{case.key} JSON root is not an object")
    _assert_document(document, case, fixtures, dimensions)
    _assert(
        result.statistics.get("object_count") == _OBJECT_COUNT,
        f"{case.key} statistics lost object_count: {result.statistics}",
    )
    _assert_state_restored(
        context_before=context_before,
        bake_state_before=bake_state_before,
        render_state_before=render_state_before,
        fixtures=fixtures,
        material_fingerprints=material_fingerprints,
        analysis_matrices=analysis_matrices,
    )


def main() -> None:
    cases = _cases()
    failures: list[tuple[str, str]] = []
    started = time.perf_counter()
    print(f"Blender version: {bpy.app.version_string}")
    print(
        "[MULTI-SEQUENCE-MATRIX] "
        f"cases={len(cases)} objects={_OBJECT_COUNT} frames={_SEQUENCE_FRAME_COUNT} "
        f"texture={_TEXTURE_SIZE}x{_TEXTURE_SIZE}"
    )

    with tempfile.TemporaryDirectory(prefix="spine2d-multi-sequence-matrix-") as directory:
        output_root = Path(directory)
        for index, case in enumerate(cases, start=1):
            case_started = time.perf_counter()
            print(f"[MULTI-SEQUENCE-MATRIX] RUN {index}/{len(cases)} {case.key}")
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
    print(f"[MULTI-SEQUENCE-MATRIX] PASS {len(cases)} cases ({elapsed:.2f}s total)")


if __name__ == "__main__":
    main()
