"""Real Blender mixed static/sequence matrix for connected and mixed Spine 4.2.

The production-supported connected and mixed scopes are validated with both rig
profiles and both texture modes.  Exactly one object owns a two-frame sequence; every
other object is static.  Mixed export is exercised twice so the sequence owner exists
once inside the connected subgroup and once inside the standalone subgroup.
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
    ConnectedCameraRenderPolicy,
    ExportSettings,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter import (  # noqa: E402
    A1MultiObjectSource,
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
)
from run_camera_projection_integration import (  # noqa: E402
    _purge_orphan_scene_data,
    _scene_render_fingerprint,
)
from run_connected_mixed_sequence_mode_matrix_integration import (  # noqa: E402
    _Case as _CompositionCase,
    _CaseOutput,
    _animate_transform,
    _assert_composition,
    _assert_state_restored,
    _create_pentagon,
    _create_quad,
    _create_triangle,
    _execute_public_export,
    _prepare_scene,
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
    _slot_attachments,
)


_TEXTURE_SIZE = 128
_SEQUENCE_START_FRAME = 1
_SEQUENCE_FRAME_COUNT = 2
_ANALYSIS_FRAME = 19
_TARGET = SpineJsonTarget.SPINE_4_2
_SEQUENCE_CONNECTED = "CONNECTED_OBJECT"
_SEQUENCE_STANDALONE = "STANDALONE_OBJECT"

_PROFILES = (
    A1RigProfile.THREE_AXIS_ROTATION,
    A1RigProfile.TWO_AXIS_ROTATION_SCALE,
)
_TEXTURE_MODES = (
    A1TextureExportMode.NORMAL_UV_SEGMENTS,
    A1TextureExportMode.CAMERA_PROJECTION,
)
_PROFILE_TOKENS = {
    A1RigProfile.THREE_AXIS_ROTATION: "ThreeAxis",
    A1RigProfile.TWO_AXIS_ROTATION_SCALE: "TwoAxisScale",
}
_MODE_TOKENS = {
    A1TextureExportMode.NORMAL_UV_SEGMENTS: "NormalUv",
    A1TextureExportMode.CAMERA_PROJECTION: "CameraProjection",
}


@dataclass(frozen=True, slots=True)
class _Case:
    scope: A1MultiObjectMode
    profile: A1RigProfile
    texture_mode: A1TextureExportMode
    sequence_owner: str

    def __post_init__(self) -> None:
        if self.scope not in {A1MultiObjectMode.CONNECTED, A1MultiObjectMode.MIXED}:
            raise ValueError(f"Unsupported scope: {self.scope!r}")
        if self.profile not in _PROFILES:
            raise ValueError(f"Unsupported profile: {self.profile!r}")
        if self.texture_mode not in _TEXTURE_MODES:
            raise ValueError(f"Unsupported texture mode: {self.texture_mode!r}")
        allowed_owners = (
            {_SEQUENCE_CONNECTED}
            if self.scope is A1MultiObjectMode.CONNECTED
            else {_SEQUENCE_CONNECTED, _SEQUENCE_STANDALONE}
        )
        if self.sequence_owner not in allowed_owners:
            raise ValueError(
                f"Unsupported sequence owner {self.sequence_owner!r} for {self.scope.value}"
            )

    @property
    def key(self) -> str:
        owner_token = (
            "SequenceConnected"
            if self.sequence_owner == _SEQUENCE_CONNECTED
            else "SequenceStandalone"
        )
        return "_".join(
            (
                self.scope.value.title(),
                _PROFILE_TOKENS[self.profile],
                _MODE_TOKENS[self.texture_mode],
                owner_token,
            )
        )

    @property
    def object_count(self) -> int:
        return 2 if self.scope is A1MultiObjectMode.CONNECTED else 3

    @property
    def sequence_index(self) -> int:
        return 0 if self.sequence_owner == _SEQUENCE_CONNECTED else 2

    @property
    def expected_texture_count(self) -> int:
        return self.object_count + 1

    @property
    def composition_case(self) -> _CompositionCase:
        return _CompositionCase(
            scope=self.scope,
            profile=self.profile,
            texture_mode=self.texture_mode,
        )


@dataclass(frozen=True, slots=True)
class _MixedFixture:
    fixture: _Fixture
    sequence_frame_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.fixture, _Fixture):
            raise TypeError("fixture must be _Fixture")
        if self.sequence_frame_count not in {0, _SEQUENCE_FRAME_COUNT}:
            raise ValueError("fixture must be static or a two-frame sequence")

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
    def is_sequence(self) -> bool:
        return self.sequence_frame_count > 0

    @property
    def texture_count(self) -> int:
        return max(1, self.sequence_frame_count)


@dataclass(frozen=True, slots=True)
class _CaseFixtures:
    all_items: tuple[_MixedFixture, ...]
    connected_items: tuple[_MixedFixture, ...]
    standalone_items: tuple[_MixedFixture, ...]

    def __post_init__(self) -> None:
        if len(self.connected_items) != 2:
            raise ValueError("connected_items must contain exactly two objects")
        if self.all_items != self.connected_items + self.standalone_items:
            raise ValueError("all_items must preserve connected then standalone order")
        if len(self.standalone_items) not in {0, 1}:
            raise ValueError("standalone_items must contain zero or one object")
        if sum(item.is_sequence for item in self.all_items) != 1:
            raise ValueError("exactly one object must own a sequence")

    @property
    def composition_output(self) -> _CaseOutput:
        return _CaseOutput(
            fixtures=tuple(item.fixture for item in self.all_items),
            connected_fixtures=tuple(item.fixture for item in self.connected_items),
            standalone_fixtures=tuple(item.fixture for item in self.standalone_items),
        )


@dataclass(frozen=True, slots=True)
class _ImageGroup:
    item: _MixedFixture
    paths: tuple[Path, ...]
    size: tuple[int, int]


def _cases() -> tuple[_Case, ...]:
    cases: list[_Case] = []
    for profile in _PROFILES:
        for texture_mode in _TEXTURE_MODES:
            cases.append(
                _Case(
                    scope=A1MultiObjectMode.CONNECTED,
                    profile=profile,
                    texture_mode=texture_mode,
                    sequence_owner=_SEQUENCE_CONNECTED,
                )
            )
            for owner in (_SEQUENCE_CONNECTED, _SEQUENCE_STANDALONE):
                cases.append(
                    _Case(
                        scope=A1MultiObjectMode.MIXED,
                        profile=profile,
                        texture_mode=texture_mode,
                        sequence_owner=owner,
                    )
                )
    result = tuple(cases)
    _assert(len(result) == 12, f"connected/mixed mixed-timing matrix must contain 12 cases")
    return result


def _object_settings(
    output_directory: Path,
    case: _Case,
    *,
    prefix: str,
    sequence_frame_count: int,
) -> A1SingleObjectExportSettings:
    return A1SingleObjectExportSettings(
        export=ExportSettings(
            texture_width=_TEXTURE_SIZE,
            texture_height=_TEXTURE_SIZE,
            output_directory=output_directory,
            images_relative_path="images",
            spine_version=_TARGET.exact_version,
            rig_profile=case.profile.value,
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


def _build_fixtures(output_directory: Path, case: _Case) -> _CaseFixtures:
    geometry = (
        (_create_pentagon, 10),
        (_create_quad, 8),
        (_create_triangle, 6),
    )
    items: list[_MixedFixture] = []
    for index, (builder, uv_scalar_count) in enumerate(geometry[: case.object_count]):
        suffix = chr(ord("A") + index)
        prefix = f"{case.key}Object{suffix}"
        obj = builder(f"{prefix}_Source")
        material = _variant_material(f"{prefix}_Material", variant=index)
        obj.data.materials.append(material)
        _animate_transform(obj, variant=index)
        frame_count = _SEQUENCE_FRAME_COUNT if index == case.sequence_index else 0
        fixture = _Fixture(
            source=A1MultiObjectSource(
                source_object=obj,
                component_id=f"{case.key.casefold()}_component_{index + 1}",
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
        items.append(
            _MixedFixture(
                fixture=fixture,
                sequence_frame_count=frame_count,
            )
        )
    all_items = tuple(items)
    return _CaseFixtures(
        all_items=all_items,
        connected_items=all_items[:2],
        standalone_items=all_items[2:],
    )


def _multi_settings(
    output_directory: Path,
    case: _Case,
    fixtures: _CaseFixtures,
) -> A1MultiObjectExportSettings:
    return A1MultiObjectExportSettings(
        output_directory=output_directory,
        output_stem=f"{case.key}Output",
        mode=case.scope,
        namespace_animations=True,
        anchor_component_id=fixtures.connected_items[0].source.component_id,
        connected_camera_render_policy=ConnectedCameraRenderPolicy.INDIVIDUAL_LAYERS,
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
    case: _Case,
    fixtures: _CaseFixtures,
) -> dict[str, tuple[Path, ...]]:
    png_paths = tuple(path for path in output_files if path.suffix.casefold() == ".png")
    _assert(
        len(png_paths) == case.expected_texture_count,
        f"{case.key} expected {case.expected_texture_count} PNGs, got {png_paths}",
    )
    groups: dict[str, tuple[Path, ...]] = {}
    for item in fixtures.all_items:
        filename_prefix = f"{item.output_stem}_Baked"
        matches = tuple(
            sorted(path for path in png_paths if path.name.startswith(filename_prefix))
        )
        _assert(
            len(matches) == item.texture_count,
            f"{case.key}/{item.prefix} expected {item.texture_count} PNG(s), got {matches}",
        )
        groups[item.prefix] = matches
    flattened = tuple(path for paths in groups.values() for path in paths)
    _assert(
        len(flattened) == len(set(flattened)) == case.expected_texture_count,
        "connected/mixed static and sequence PNG paths overlap",
    )
    return groups


def _validate_images(
    groups: Mapping[str, tuple[Path, ...]],
    case: _Case,
    fixtures: _CaseFixtures,
) -> dict[str, _ImageGroup]:
    result: dict[str, _ImageGroup] = {}
    payloads: list[bytes] = []
    for item in fixtures.all_items:
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
                    f"Camera Projection sequence crop changed for {item.prefix}: {sizes}",
                )
            width, height = sizes[0]
            _assert(
                1 <= width <= _TEXTURE_SIZE and 1 <= height <= _TEXTURE_SIZE,
                f"Camera Projection crop outside 1..{_TEXTURE_SIZE}: {sizes[0]}",
            )
        result[item.prefix] = _ImageGroup(
            item=item,
            paths=paths,
            size=sizes[0],
        )
        payloads.extend(image.file_bytes for image in images)
    _assert(
        len(payloads) == len({payload for payload in payloads}),
        "connected/mixed objects or frames produced duplicate PNG payloads",
    )
    return result


def _visual_slots(
    document: Mapping[str, object],
    fixtures: _CaseFixtures,
) -> tuple[dict[str, object], ...]:
    prefixes = tuple(item.prefix for item in fixtures.all_items)
    slots = tuple(
        slot
        for slot in _json_array(document, "slots")
        if isinstance(slot, dict)
        and isinstance(slot.get("name"), str)
        and isinstance(slot.get("attachment"), str)
        and str(slot["name"]).startswith(prefixes)
    )
    _assert(
        len(slots) == len(fixtures.all_items),
        f"expected {len(fixtures.all_items)} visual slots, got {tuple(item.get('name') for item in slots)}",
    )
    return slots


def _fixture_for_slot(slot_name: str, fixtures: _CaseFixtures) -> _MixedFixture:
    matches = tuple(
        item for item in fixtures.all_items if slot_name.startswith(item.prefix)
    )
    _assert(len(matches) == 1, f"slot {slot_name!r} does not map to one fixture")
    return matches[0]


def _slot_sequence_timelines(
    document: Mapping[str, object],
    slot_name: str,
) -> tuple[list[object], ...]:
    animations = document.get("animations", {})
    _assert(isinstance(animations, dict), "animations must be a JSON object")
    matches: list[list[object]] = []

    def visit(value: object) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key == slot_name and isinstance(child, dict):
                    timeline = child.get("sequence")
                    if isinstance(timeline, list):
                        matches.append(timeline)
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(animations)
    return tuple(matches)


def _assert_native_sequences(
    document: Mapping[str, object],
    case: _Case,
    fixtures: _CaseFixtures,
    images: Mapping[str, _ImageGroup],
) -> None:
    _assert(
        _TARGET.texture_animation_encoding
        is SpineTextureAnimationEncoding.NATIVE_SEQUENCE,
        "Spine 4.2 must use native sequence encoding",
    )
    for slot in _visual_slots(document, fixtures):
        slot_name = str(slot["name"])
        setup_name = str(slot["attachment"])
        item = _fixture_for_slot(slot_name, fixtures)
        attachments = _slot_attachments(document, slot_name)
        _assert(
            tuple(attachments) == (setup_name,),
            f"{case.key}/{slot_name} must contain one attachment: {tuple(attachments)}",
        )
        attachment = attachments[setup_name]
        _assert(isinstance(attachment, dict), f"invalid attachment for {slot_name}")
        label = f"{case.key}/{slot_name}/{setup_name}"
        if case.texture_mode is A1TextureExportMode.NORMAL_UV_SEGMENTS:
            _assert_normal_attachment(attachment, item.fixture, label=label)
        else:
            _assert_projection_attachment(
                attachment,
                images[item.prefix].size,
                label=label,
            )

        timelines = _slot_sequence_timelines(document, slot_name)
        if item.is_sequence:
            sequence = attachment.get("sequence")
            _assert(isinstance(sequence, dict), f"native sequence missing for {slot_name}")
            _assert(
                sequence.get("count") == _SEQUENCE_FRAME_COUNT,
                f"native sequence count is wrong for {slot_name}: {sequence}",
            )
            _assert(len(timelines) == 1, f"expected one sequence timeline for {slot_name}")
            timeline = timelines[0]
            _assert(len(timeline) == 2, f"native Loop timeline invalid: {timeline}")
            _assert(
                isinstance(timeline[0], dict) and timeline[0].get("mode") == "loop",
                f"native sequence is not Loop: {timeline}",
            )
        else:
            _assert(
                "sequence" not in attachment,
                f"static attachment inherited sequence metadata: {slot_name}",
            )
            _assert(
                not timelines,
                f"static slot inherited sequence timeline: {slot_name}",
            )


def _assert_document(
    document: dict[str, object],
    case: _Case,
    fixtures: _CaseFixtures,
    images: Mapping[str, _ImageGroup],
) -> None:
    skeleton = document.get("skeleton")
    _assert(isinstance(skeleton, dict), "skeleton metadata missing")
    _assert(
        skeleton.get("spine") == _TARGET.exact_version,
        f"{case.key} version mismatch: {skeleton.get('spine')!r}",
    )
    _assert_bone_schema(document, _TARGET)
    _assert_constraint_schema(document, _TARGET)
    _assert_composition(document, case.composition_case, fixtures.composition_output)
    _assert_native_sequences(document, case, fixtures, images)


def _run_case(output_root: Path, case: _Case) -> None:
    case_directory = output_root / case.key
    case_directory.mkdir(parents=True, exist_ok=False)
    sentinel = _prepare_scene()
    fixtures = _build_fixtures(case_directory, case)
    _activate_only(sentinel)
    for item in fixtures.all_items:
        item.obj.select_set(False)

    scene = bpy.context.scene
    scene.frame_set(_ANALYSIS_FRAME)
    bpy.context.view_layer.update()
    analysis_matrices = tuple(_matrix_tuple(item.obj) for item in fixtures.all_items)
    sequence_item = next(item for item in fixtures.all_items if item.is_sequence)
    frame_matrices = _frame_matrices(scene, sequence_item.obj)
    _assert(
        not _matrices_equal_at_float32_precision(frame_matrices[0], frame_matrices[1]),
        f"{case.key} sequence object bake matrices are identical",
    )
    sequence_analysis_matrix = analysis_matrices[case.sequence_index]
    _assert(
        all(
            not _matrices_equal_at_float32_precision(matrix, sequence_analysis_matrix)
            for matrix in frame_matrices
        ),
        f"{case.key} planning matrix matches a sequence bake frame",
    )

    context_before = _capture_context()
    bake_state_before = _capture_scene_bake_state()
    render_state_before = _scene_render_fingerprint()
    material_fingerprints = tuple(
        _material_fingerprint(item.fixture.material) for item in fixtures.all_items
    )
    settings = _multi_settings(case_directory, case, fixtures)

    result = _execute_public_export(
        fixtures.composition_output,
        settings,
        context=bpy.context,
        scene=scene,
    )
    _assert(result.success, f"{case.key} failed: {result.issues}")
    expected_output_count = 1 + case.expected_texture_count
    _assert(
        len(result.output_files) == expected_output_count,
        f"{case.key} expected {expected_output_count} outputs: {result.output_files}",
    )
    json_path = result.output_files[0]
    _assert(json_path.suffix.casefold() == ".json", f"JSON is not first: {json_path}")
    _assert(json_path.is_file(), f"missing JSON: {json_path}")

    groups = _texture_groups(result.output_files, case, fixtures)
    images = _validate_images(groups, case, fixtures)
    document = json.loads(json_path.read_text(encoding="utf-8"))
    _assert(isinstance(document, dict), f"{case.key} JSON root is not an object")
    _assert_document(document, case, fixtures, images)
    _assert(
        result.statistics.get("object_count") == case.object_count,
        f"{case.key} statistics lost object_count: {result.statistics}",
    )
    _assert_state_restored(
        context_before=context_before,
        bake_state_before=bake_state_before,
        render_state_before=render_state_before,
        fixtures=tuple(item.fixture for item in fixtures.all_items),
        material_fingerprints=material_fingerprints,
        analysis_matrices=analysis_matrices,
    )


def main() -> None:
    cases = _cases()
    failures: list[tuple[str, str]] = []
    started = time.perf_counter()
    print(f"Blender version: {bpy.app.version_string}")
    print(
        "[CONNECTED-MIXED-MIXED-TIMING] "
        f"cases={len(cases)} target={_TARGET.label} frames={_SEQUENCE_FRAME_COUNT} "
        f"texture={_TEXTURE_SIZE}x{_TEXTURE_SIZE} sequence_objects=1"
    )

    with tempfile.TemporaryDirectory(
        prefix="spine2d-connected-mixed-mixed-timing-"
    ) as directory:
        output_root = Path(directory)
        for index, case in enumerate(cases, start=1):
            case_started = time.perf_counter()
            print(
                f"[CONNECTED-MIXED-MIXED-TIMING] RUN "
                f"{index}/{len(cases)} {case.key}"
            )
            try:
                _run_case(output_root, case)
            except Exception:
                failures.append((case.key, traceback.format_exc()))
                print(f"[CONNECTED-MIXED-MIXED-TIMING] FAIL {case.key}")
            else:
                elapsed = time.perf_counter() - case_started
                print(
                    f"[CONNECTED-MIXED-MIXED-TIMING] "
                    f"PASS {case.key} ({elapsed:.2f}s)"
                )
            finally:
                _clear_scene()
                _purge_orphan_scene_data()

    if failures:
        for case_key, details in failures:
            print(f"\n--- {case_key} ---\n{details}")
        raise SystemExit(1)

    elapsed = time.perf_counter() - started
    print(
        f"[CONNECTED-MIXED-MIXED-TIMING] PASS "
        f"{len(cases)} cases ({elapsed:.2f}s total)"
    )


if __name__ == "__main__":
    main()
