"""Real Blender mixed static/sequence standalone matrix for Spine 3.8 through 4.3.

Every case exports three objects through ``export_a1_multi_object``:

* object A: two-frame texture sequence starting at frame 1;
* objects B and C: one static texture evaluated at the current analysis frame.

Both Normal / UV Segments and Camera Projection are exercised at 128x128 with one
Cycles sample.  The final JSON must animate only object A.  Static objects must not
inherit attachment-swap or native-sequence metadata from the animated source.
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
    _activate_only,
    _assert,
    _capture_context,
    _capture_scene_bake_state,
    _clear_scene,
    _material_fingerprint,
    _temporary_datablock_names,
)
from run_camera_projection_integration import (  # noqa: E402
    _configure_scene,
    _create_camera,
    _purge_orphan_scene_data,
    _scene_render_fingerprint,
)
from run_connected_mixed_sequence_mode_matrix_integration import (  # noqa: E402
    _animate_transform,
    _create_pentagon,
    _create_quad,
    _create_triangle,
    _variant_material,
)
from run_multi_object_sequence_mode_matrix_integration import (  # noqa: E402
    _Fixture,
    _assert_distinct_frames,
    _assert_normal_attachment,
    _assert_projection_attachment,
    _image_summary,
    _json_array,
    _matrices_equal_at_float32_precision,
    _matrix_tuple,
    _prepare_scene,
    _slot_attachments,
)


_TEXTURE_SIZE = 128
_SEQUENCE_START_FRAME = 1
_SEQUENCE_FRAME_COUNT = 2
_ANALYSIS_FRAME = 19
_OBJECT_COUNT = 3
_EXPECTED_TEXTURE_COUNT = 4

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
class _MixedFixture:
    fixture: _Fixture
    sequence_frame_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.fixture, _Fixture):
            raise TypeError("fixture must be _Fixture")
        if self.sequence_frame_count not in {0, _SEQUENCE_FRAME_COUNT}:
            raise ValueError("fixture must be static or use the two-frame sequence")

    @property
    def source(self) -> A1MultiObjectSource:
        return self.fixture.source

    @property
    def obj(self) -> object:
        return self.fixture.obj

    @property
    def prefix(self) -> str:
        return self.fixture.prefix

    @property
    def output_stem(self) -> str:
        return self.fixture.output_stem

    @property
    def texture_count(self) -> int:
        return max(1, self.sequence_frame_count)

    @property
    def is_sequence(self) -> bool:
        return self.sequence_frame_count > 0


@dataclass(frozen=True, slots=True)
class _ImageGroup:
    fixture: _MixedFixture
    paths: tuple[Path, ...]
    size: tuple[int, int]


def _cases() -> tuple[_Case, ...]:
    result = tuple(
        _Case(target=target, texture_mode=texture_mode)
        for target in _TARGETS
        for texture_mode in _TEXTURE_MODES
    )
    _assert(len(result) == 10, f"mixed static/sequence matrix must contain 10 cases")
    return result


def _object_settings(
    output_directory: Path,
    case: _Case,
    *,
    prefix: str,
    sequence_frame_count: int,
) -> A1SingleObjectExportSettings:
    if sequence_frame_count not in {0, _SEQUENCE_FRAME_COUNT}:
        raise ValueError("sequence_frame_count must be zero or two")
    return A1SingleObjectExportSettings(
        export=ExportSettings(
            texture_width=_TEXTURE_SIZE,
            texture_height=_TEXTURE_SIZE,
            output_directory=output_directory,
            images_relative_path="images",
            spine_version=case.target.exact_version,
            rig_profile=A1RigProfile.TWO_AXIS_ROTATION_SCALE.value,
            bake_margin=1,
            sequence_start_frame=(
                _SEQUENCE_START_FRAME if sequence_frame_count else 0
            ),
            sequence_frame_count=sequence_frame_count,
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
) -> tuple[_MixedFixture, ...]:
    geometry = (
        (_create_pentagon, 10, _SEQUENCE_FRAME_COUNT),
        (_create_quad, 8, 0),
        (_create_triangle, 6, 0),
    )
    fixtures: list[_MixedFixture] = []
    for index, (builder, uv_scalar_count, frame_count) in enumerate(geometry):
        suffix = chr(ord("A") + index)
        prefix = f"{case.key}MixedTimingObject{suffix}"
        obj = builder(f"{prefix}_Source")
        material = _variant_material(f"{prefix}_Material", variant=index)
        obj.data.materials.append(material)
        _animate_transform(obj, variant=index)
        fixture = _Fixture(
            source=A1MultiObjectSource(
                source_object=obj,
                component_id=f"{case.key.casefold()}_mixed_timing_{index + 1}",
                animation_namespace=f"object_{index + 1}",
                settings=_object_settings(
                    output_directory,
                    case,
                    prefix=prefix,
                    sequence_frame_count=frame_count,
                ),
            ),
            material=material,
            expected_normal_uv_scalars=uv_scalar_count,
        )
        fixtures.append(
            _MixedFixture(
                fixture=fixture,
                sequence_frame_count=frame_count,
            )
        )
    result = tuple(fixtures)
    _assert(len(result) == _OBJECT_COUNT, "fixture count changed")
    _assert(sum(item.is_sequence for item in result) == 1, "exactly one sequence required")
    _assert(sum(item.texture_count for item in result) == _EXPECTED_TEXTURE_COUNT, "texture count changed")
    return result


def _multi_settings(output_directory: Path, case: _Case) -> A1MultiObjectExportSettings:
    return A1MultiObjectExportSettings(
        output_directory=output_directory,
        output_stem=f"{case.key}MixedTimingMulti",
        mode=A1MultiObjectMode.STANDALONE,
        namespace_animations=True,
    )


def _frame_matrices(scene: object, obj: object) -> tuple[tuple[float, ...], ...]:
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


def _texture_groups(
    output_files: tuple[Path, ...],
    fixtures: tuple[_MixedFixture, ...],
) -> dict[str, tuple[Path, ...]]:
    png_paths = tuple(path for path in output_files if path.suffix.casefold() == ".png")
    _assert(
        len(png_paths) == _EXPECTED_TEXTURE_COUNT,
        f"expected {_EXPECTED_TEXTURE_COUNT} PNGs, got {png_paths}",
    )
    groups: dict[str, tuple[Path, ...]] = {}
    for item in fixtures:
        filename_prefix = f"{item.output_stem}_Baked"
        matches = tuple(
            sorted(path for path in png_paths if path.name.startswith(filename_prefix))
        )
        _assert(
            len(matches) == item.texture_count,
            f"{item.prefix} expected {item.texture_count} PNG(s), got {matches}",
        )
        groups[item.prefix] = matches
    flattened = tuple(path for paths in groups.values() for path in paths)
    _assert(
        len(flattened) == len(set(flattened)) == _EXPECTED_TEXTURE_COUNT,
        "PNG paths overlap between static and sequence objects",
    )
    return groups


def _validate_images(
    groups: Mapping[str, tuple[Path, ...]],
    fixtures: tuple[_MixedFixture, ...],
    case: _Case,
) -> dict[str, _ImageGroup]:
    result: dict[str, _ImageGroup] = {}
    all_payloads: list[bytes] = []
    for item in fixtures:
        paths = groups[item.prefix]
        images = tuple(_image_summary(path) for path in paths)
        if item.is_sequence:
            _assert_distinct_frames(
                images[0],
                images[1],
                label=f"{case.key}/{item.prefix}",
            )
        sizes = tuple(image.size for image in images)
        if case.texture_mode is A1TextureExportMode.NORMAL_UV_SEGMENTS:
            _assert(
                sizes == ((_TEXTURE_SIZE, _TEXTURE_SIZE),) * item.texture_count,
                f"Normal UV size mismatch for {item.prefix}: {sizes}",
            )
        else:
            if item.is_sequence:
                _assert(
                    len(set(sizes)) == 1,
                    f"Camera Projection sequence crop changed: {item.prefix}/{sizes}",
                )
            width, height = sizes[0]
            _assert(
                1 <= width <= _TEXTURE_SIZE and 1 <= height <= _TEXTURE_SIZE,
                f"Camera Projection crop outside 1..{_TEXTURE_SIZE}: {sizes[0]}",
            )
        result[item.prefix] = _ImageGroup(
            fixture=item,
            paths=paths,
            size=sizes[0],
        )
        all_payloads.extend(image.file_bytes for image in images)

    _assert(
        len(all_payloads) == len({payload for payload in all_payloads}),
        "static objects or sequence frames produced duplicate PNG payloads",
    )
    return result


def _visual_slots(
    document: Mapping[str, object],
    fixtures: tuple[_MixedFixture, ...],
) -> tuple[dict[str, object], ...]:
    prefixes = tuple(item.prefix for item in fixtures)
    slots = tuple(
        slot
        for slot in _json_array(document, "slots")
        if isinstance(slot, dict)
        and isinstance(slot.get("name"), str)
        and isinstance(slot.get("attachment"), str)
        and str(slot["name"]).startswith(prefixes)
    )
    _assert(
        len(slots) == _OBJECT_COUNT,
        f"expected {_OBJECT_COUNT} visual slots, got {tuple(item.get('name') for item in slots)}",
    )
    return slots


def _fixture_for_slot(
    slot_name: str,
    fixtures: tuple[_MixedFixture, ...],
) -> _MixedFixture:
    matches = tuple(item for item in fixtures if slot_name.startswith(item.prefix))
    _assert(len(matches) == 1, f"slot {slot_name!r} does not map to one fixture")
    return matches[0]


def _slot_timelines(
    document: Mapping[str, object],
    slot_name: str,
    timeline_key: str,
) -> tuple[list[object], ...]:
    animations = document.get("animations", {})
    _assert(isinstance(animations, dict), "animations must be a JSON object")
    matches: list[list[object]] = []

    def visit(value: object) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key == slot_name and isinstance(child, dict):
                    timeline = child.get(timeline_key)
                    if isinstance(timeline, list):
                        matches.append(timeline)
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(animations)
    return tuple(matches)


def _assert_attachment_geometry(
    attachment: Mapping[str, object],
    item: _MixedFixture,
    image_size: tuple[int, int],
    case: _Case,
    *,
    label: str,
) -> None:
    if case.texture_mode is A1TextureExportMode.NORMAL_UV_SEGMENTS:
        _assert_normal_attachment(attachment, item.fixture, label=label)
    else:
        _assert_projection_attachment(attachment, image_size, label=label)


def _assert_texture_encoding(
    document: Mapping[str, object],
    case: _Case,
    fixtures: tuple[_MixedFixture, ...],
    images: Mapping[str, _ImageGroup],
) -> None:
    native = (
        case.target.texture_animation_encoding
        is SpineTextureAnimationEncoding.NATIVE_SEQUENCE
    )
    for slot in _visual_slots(document, fixtures):
        slot_name = str(slot["name"])
        setup_name = str(slot["attachment"])
        item = _fixture_for_slot(slot_name, fixtures)
        attachments = _slot_attachments(document, slot_name)
        expected_attachment_count = (
            1
            if native or not item.is_sequence
            else _SEQUENCE_FRAME_COUNT
        )
        _assert(
            len(attachments) == expected_attachment_count,
            f"{case.key}/{slot_name} expected {expected_attachment_count} attachment(s), "
            f"got {tuple(attachments)}",
        )
        _assert(setup_name in attachments, f"setup attachment missing for {slot_name}")

        for attachment_name, raw_attachment in attachments.items():
            _assert(isinstance(raw_attachment, dict), f"{attachment_name} is not an object")
            _assert_attachment_geometry(
                raw_attachment,
                item,
                images[item.prefix].size,
                case,
                label=f"{case.key}/{slot_name}/{attachment_name}",
            )

        setup_attachment = attachments[setup_name]
        _assert(isinstance(setup_attachment, dict), f"setup attachment invalid for {slot_name}")
        if native and item.is_sequence:
            sequence = setup_attachment.get("sequence")
            _assert(isinstance(sequence, dict), f"native sequence missing for {slot_name}")
            _assert(
                sequence.get("count") == _SEQUENCE_FRAME_COUNT,
                f"native sequence count is wrong for {slot_name}: {sequence}",
            )
            timelines = _slot_timelines(document, slot_name, "sequence")
            _assert(len(timelines) == 1, f"expected one sequence timeline for {slot_name}")
            timeline = timelines[0]
            _assert(len(timeline) == 2, f"native Loop timeline invalid: {timeline}")
            _assert(
                isinstance(timeline[0], dict) and timeline[0].get("mode") == "loop",
                f"native sequence is not Loop: {timeline}",
            )
        elif native:
            _assert(
                "sequence" not in setup_attachment,
                f"static attachment inherited native sequence metadata: {slot_name}",
            )
            _assert(
                not _slot_timelines(document, slot_name, "sequence"),
                f"static slot inherited native sequence timeline: {slot_name}",
            )
        elif item.is_sequence:
            _assert(
                all(
                    isinstance(value, dict) and "sequence" not in value
                    for value in attachments.values()
                ),
                f"legacy sequence contains native metadata: {slot_name}",
            )
            timelines = _slot_timelines(document, slot_name, "attachment")
            _assert(len(timelines) == 1, f"expected one attachment timeline for {slot_name}")
            timeline = timelines[0]
            _assert(
                len(timeline) == _SEQUENCE_FRAME_COUNT + 1,
                f"legacy Loop timeline must contain frame keys plus wrap: {timeline}",
            )
            names = tuple(
                key.get("name") if isinstance(key, dict) else None
                for key in timeline
            )
            _assert(len(set(names[:-1])) == 2, f"legacy frame names are not unique: {names}")
            _assert(names[-1] == names[0], f"legacy Loop does not wrap: {names}")
            _assert(
                all(name in attachments for name in names if isinstance(name, str)),
                f"legacy timeline references unknown attachments: {names}",
            )
        else:
            _assert(
                all(
                    isinstance(value, dict) and "sequence" not in value
                    for value in attachments.values()
                ),
                f"static legacy attachment contains sequence metadata: {slot_name}",
            )
            _assert(
                not _slot_timelines(document, slot_name, "attachment"),
                f"static slot inherited attachment timeline: {slot_name}",
            )


def _assert_document(
    document: dict[str, object],
    case: _Case,
    fixtures: tuple[_MixedFixture, ...],
    images: Mapping[str, _ImageGroup],
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
    for item in fixtures:
        _assert(
            f"{item.prefix}_main" in bone_names,
            f"{case.key} lost main bone for {item.prefix}",
        )
    _assert(
        not any(name.startswith("all_objects") for name in bone_names),
        f"{case.key} standalone export created connected wrapper",
    )
    _assert_texture_encoding(document, case, fixtures, images)


def _assert_state_restored(
    *,
    context_before: object,
    bake_state_before: object,
    render_state_before: object,
    fixtures: tuple[_MixedFixture, ...],
    material_fingerprints: tuple[object, ...],
    analysis_matrices: tuple[tuple[float, ...], ...],
) -> None:
    _assert(_capture_context() == context_before, "export changed Blender context")
    _assert(
        _capture_scene_bake_state() == bake_state_before,
        "export changed Scene bake settings",
    )
    _assert(
        _scene_render_fingerprint() == render_state_before,
        "export changed render or camera state",
    )
    _assert(
        tuple(_material_fingerprint(item.fixture.material) for item in fixtures)
        == material_fingerprints,
        "export mutated source materials",
    )
    bpy.context.view_layer.update()
    for item, expected_matrix in zip(fixtures, analysis_matrices, strict=True):
        _assert(
            _matrices_equal_at_float32_precision(_matrix_tuple(item.obj), expected_matrix),
            f"export did not restore matrix_world for {item.obj.name}",
        )
    _assert(not _temporary_datablock_names(), "export leaked temporary datablocks")


def _run_case(output_root: Path, case: _Case) -> None:
    case_directory = output_root / case.key
    case_directory.mkdir(parents=True, exist_ok=False)
    sentinel = _prepare_scene()
    fixtures = _build_fixtures(case_directory, case)
    _activate_only(sentinel)
    for item in fixtures:
        item.obj.select_set(False)

    scene = bpy.context.scene
    scene.frame_set(_ANALYSIS_FRAME)
    bpy.context.view_layer.update()
    analysis_matrices = tuple(_matrix_tuple(item.obj) for item in fixtures)
    sequence_item = next(item for item in fixtures if item.is_sequence)
    sequence_matrices = _frame_matrices(scene, sequence_item.obj)
    _assert(
        not _matrices_equal_at_float32_precision(
            sequence_matrices[0], sequence_matrices[1]
        ),
        f"{case.key} sequence object bake matrices are identical",
    )
    _assert(
        all(
            not _matrices_equal_at_float32_precision(matrix, analysis_matrices[0])
            for matrix in sequence_matrices
        ),
        f"{case.key} planning matrix matches a sequence bake frame",
    )

    context_before = _capture_context()
    bake_state_before = _capture_scene_bake_state()
    render_state_before = _scene_render_fingerprint()
    material_fingerprints = tuple(
        _material_fingerprint(item.fixture.material) for item in fixtures
    )

    result = export_a1_multi_object(
        tuple(item.source for item in fixtures),
        _multi_settings(case_directory, case),
        context=bpy.context,
        scene=scene,
    )
    _assert(result.success, f"{case.key} failed: {result.issues}")
    _assert(
        len(result.output_files) == 1 + _EXPECTED_TEXTURE_COUNT,
        f"{case.key} output count is wrong: {result.output_files}",
    )
    json_path = result.output_files[0]
    _assert(json_path.suffix.casefold() == ".json", f"JSON is not first: {json_path}")
    _assert(json_path.is_file(), f"missing JSON: {json_path}")
    groups = _texture_groups(result.output_files, fixtures)
    images = _validate_images(groups, fixtures, case)
    document = json.loads(json_path.read_text(encoding="utf-8"))
    _assert(isinstance(document, dict), f"{case.key} JSON root is not an object")
    _assert_document(document, case, fixtures, images)
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
        "[MULTI-MIXED-TIMING] "
        f"cases={len(cases)} objects={_OBJECT_COUNT} sequence_objects=1 "
        f"static_objects=2 expected_png_per_case={_EXPECTED_TEXTURE_COUNT} "
        f"texture={_TEXTURE_SIZE}x{_TEXTURE_SIZE}"
    )

    with tempfile.TemporaryDirectory(prefix="spine2d-multi-mixed-timing-") as directory:
        output_root = Path(directory)
        for index, case in enumerate(cases, start=1):
            case_started = time.perf_counter()
            print(f"[MULTI-MIXED-TIMING] RUN {index}/{len(cases)} {case.key}")
            try:
                _run_case(output_root, case)
            except Exception:
                failures.append((case.key, traceback.format_exc()))
                print(f"[MULTI-MIXED-TIMING] FAIL {case.key}")
            else:
                elapsed = time.perf_counter() - case_started
                print(f"[MULTI-MIXED-TIMING] PASS {case.key} ({elapsed:.2f}s)")
            finally:
                _clear_scene()
                _purge_orphan_scene_data()

    if failures:
        for case_key, details in failures:
            print(f"\n--- {case_key} ---\n{details}")
        raise SystemExit(1)

    elapsed = time.perf_counter() - started
    print(f"[MULTI-MIXED-TIMING] PASS {len(cases)} cases ({elapsed:.2f}s total)")


if __name__ == "__main__":
    main()
