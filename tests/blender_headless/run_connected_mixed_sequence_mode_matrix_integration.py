"""Real Blender 5.2 sequence matrix for connected and mixed Spine 4.2 export.

Only production-supported combinations are exercised:

* scopes: CONNECTED and MIXED;
* profiles: Three-Axis Rotation and Two-Axis Rotation + Scale;
* texture modes: Normal / UV Segments and Camera Projection;
* two timeline frames at 128x128 with one Cycles sample.

Connected cases export two objects. Mixed cases export a two-object connected subgroup
plus one standalone object. Every case calls the public output service, validates the
physical PNGs and final combined JSON, and proves Blender state restoration.
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
    export_a1_mixed_object,
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
    _create_mesh_object,
    _material_fingerprint,
    _temporary_datablock_names,
)
from run_camera_projection_integration import (  # noqa: E402
    _configure_scene,
    _create_camera,
    _purge_orphan_scene_data,
    _scene_render_fingerprint,
)
from run_multi_object_sequence_mode_matrix_integration import (  # noqa: E402
    _Fixture,
    _assert_distinct_frames,
    _assert_normal_attachment,
    _assert_projection_attachment,
    _contains_key,
    _image_summary,
    _json_array,
    _lists_for_key,
    _matrices_equal_at_float32_precision,
    _matrix_tuple,
    _slot_attachments,
)
from run_normal_uv_camera_context_sequence_integration import (  # noqa: E402
    _create_animated_camera_reflection_material,
)


_TEXTURE_SIZE = 128
_SEQUENCE_START_FRAME = 1
_SEQUENCE_FRAME_COUNT = 2
_ANALYSIS_FRAME = 19
_TARGET = SpineJsonTarget.SPINE_4_2

_SCOPES = (
    A1MultiObjectMode.CONNECTED,
    A1MultiObjectMode.MIXED,
)
_PROFILES = (
    A1RigProfile.THREE_AXIS_ROTATION,
    A1RigProfile.TWO_AXIS_ROTATION_SCALE,
)
_TEXTURE_MODES = (
    A1TextureExportMode.NORMAL_UV_SEGMENTS,
    A1TextureExportMode.CAMERA_PROJECTION,
)
_SCOPE_TOKENS = {
    A1MultiObjectMode.CONNECTED: "Connected",
    A1MultiObjectMode.MIXED: "Mixed",
}
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

    def __post_init__(self) -> None:
        if self.scope not in _SCOPES:
            raise ValueError(f"Unsupported matrix scope: {self.scope!r}")
        if self.profile not in _PROFILES:
            raise ValueError(f"Unsupported matrix profile: {self.profile!r}")
        if self.texture_mode not in _TEXTURE_MODES:
            raise ValueError(f"Unsupported matrix texture mode: {self.texture_mode!r}")

    @property
    def key(self) -> str:
        return "_".join(
            (
                _SCOPE_TOKENS[self.scope],
                _PROFILE_TOKENS[self.profile],
                _MODE_TOKENS[self.texture_mode],
            )
        )

    @property
    def object_count(self) -> int:
        return 2 if self.scope is A1MultiObjectMode.CONNECTED else 3


@dataclass(frozen=True, slots=True)
class _CaseOutput:
    fixtures: tuple[_Fixture, ...]
    connected_fixtures: tuple[_Fixture, ...]
    standalone_fixtures: tuple[_Fixture, ...]

    def __post_init__(self) -> None:
        if len(self.connected_fixtures) != 2:
            raise ValueError("connected_fixtures must contain exactly two objects")
        if self.fixtures != self.connected_fixtures + self.standalone_fixtures:
            raise ValueError("fixtures must preserve connected then standalone order")
        if len(self.standalone_fixtures) not in {0, 1}:
            raise ValueError("standalone_fixtures must contain zero or one object")


def _cases() -> tuple[_Case, ...]:
    cases = tuple(
        _Case(scope=scope, profile=profile, texture_mode=texture_mode)
        for scope in _SCOPES
        for profile in _PROFILES
        for texture_mode in _TEXTURE_MODES
    )
    _assert(len(cases) == 8, f"connected/mixed matrix must contain 8 cases: {cases}")
    return cases


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


def _create_triangle(name: str):
    return _create_mesh_object(
        name,
        (
            (-0.72, -0.60, 0.0),
            (0.78, -0.42, 0.0),
            (0.04, 0.82, 0.0),
        ),
        ((0, 1, 2),),
    )


def _variant_material(name: str, *, variant: int):
    """Create the audited Camera/Reflection graph with object-unique animation."""

    material = _create_animated_camera_reflection_material(name)
    node_tree = material.node_tree
    if node_tree is None:
        raise RuntimeError(f"Material {name!r} has no node tree")
    emission_nodes = tuple(
        node for node in node_tree.nodes if node.bl_idname == "ShaderNodeEmission"
    )
    ramp_nodes = tuple(
        node for node in node_tree.nodes if node.bl_idname == "ShaderNodeValToRGB"
    )
    if len(emission_nodes) != 1 or len(ramp_nodes) != 1:
        raise RuntimeError(
            f"Material {name!r} fixture shape changed: "
            f"emission={len(emission_nodes)}, ramps={len(ramp_nodes)}"
        )

    emission = emission_nodes[0]
    ramp = ramp_nodes[0].color_ramp
    palettes = (
        (
            (0.02, 0.12, 1.00, 1.0),
            (0.04, 0.95, 0.68, 1.0),
            (0.95, 0.04, 0.18, 1.0),
            (1.00, 0.52, 0.02, 1.0),
            (0.22, 0.04, 0.92, 1.0),
            (0.08, 0.82, 1.00, 1.0),
        ),
        (
            (0.72, 0.02, 1.00, 1.0),
            (1.00, 0.82, 0.04, 1.0),
            (0.02, 0.78, 0.96, 1.0),
            (0.96, 0.04, 0.68, 1.0),
            (0.06, 0.94, 0.34, 1.0),
            (0.94, 0.18, 0.04, 1.0),
        ),
        (
            (0.02, 0.82, 0.26, 1.0),
            (0.10, 0.18, 1.00, 1.0),
            (0.92, 0.70, 0.02, 1.0),
            (0.18, 0.02, 0.88, 1.0),
            (0.96, 0.08, 0.44, 1.0),
            (0.04, 0.88, 0.84, 1.0),
        ),
    )
    if not 0 <= variant < len(palettes):
        raise ValueError(f"Unsupported fixture variant: {variant}")
    palette = palettes[variant]
    for frame, left, right, strength in (
        (1, palette[0], palette[1], 0.80 + variant * 0.25),
        (2, palette[2], palette[3], 1.55 + variant * 0.30),
        (_ANALYSIS_FRAME, palette[4], palette[5], 1.05 + variant * 0.20),
    ):
        ramp.elements[0].color = left
        ramp.elements[1].color = right
        emission.inputs["Strength"].default_value = strength
        ramp.elements[0].keyframe_insert(data_path="color", frame=frame)
        ramp.elements[1].keyframe_insert(data_path="color", frame=frame)
        emission.inputs["Strength"].keyframe_insert(
            data_path="default_value",
            frame=frame,
        )
    return material


def _animate_transform(obj: object, *, variant: int) -> None:
    """Create two bake matrices and a distinct matrix at the planning frame."""

    base_x = (-0.92, 0.10, 1.02)[variant]
    base_y = (-0.28, 0.18, -0.02)[variant]
    transforms = (
        (
            1,
            (base_x - 0.14, base_y - 0.12, 0.10 + variant * 0.22),
            (0.10 + variant * 0.08, -0.22 + variant * 0.06, -0.20 + variant * 0.34),
            (0.82 + variant * 0.05, 1.04 - variant * 0.10, 1.0),
        ),
        (
            2,
            (base_x + 0.20, base_y + 0.24, 0.30 - variant * 0.08),
            (0.34 - variant * 0.04, 0.18 - variant * 0.08, 0.38 + variant * 0.22),
            (1.10 - variant * 0.10, 0.76 + variant * 0.12, 1.0),
        ),
        (
            _ANALYSIS_FRAME,
            (base_x, base_y - 0.42, -0.18 + variant * 0.30),
            (0.62 - variant * 0.16, -0.56 + variant * 0.20, 0.88 - variant * 0.18),
            (1.22 - variant * 0.12, 0.86 + variant * 0.09, 1.0),
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
            spine_version=_TARGET.exact_version,
            rig_profile=case.profile.value,
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


def _build_case_fixtures(
    output_directory: Path,
    case: _Case,
) -> _CaseOutput:
    geometry = (
        (_create_pentagon, 10),
        (_create_quad, 8),
        (_create_triangle, 6),
    )
    fixtures: list[_Fixture] = []
    for index, (builder, uv_scalar_count) in enumerate(geometry[: case.object_count]):
        suffix = chr(ord("A") + index)
        prefix = f"{case.key}Object{suffix}"
        obj = builder(f"{prefix}_Source")
        material = _variant_material(f"{prefix}_Material", variant=index)
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

    fixture_tuple = tuple(fixtures)
    connected = fixture_tuple[:2]
    standalone = fixture_tuple[2:]
    return _CaseOutput(
        fixtures=fixture_tuple,
        connected_fixtures=connected,
        standalone_fixtures=standalone,
    )


def _multi_settings(
    output_directory: Path,
    case: _Case,
    fixtures: _CaseOutput,
) -> A1MultiObjectExportSettings:
    return A1MultiObjectExportSettings(
        output_directory=output_directory,
        output_stem=f"{case.key}Output",
        mode=case.scope,
        namespace_animations=True,
        anchor_component_id=fixtures.connected_fixtures[0].source.component_id,
        connected_camera_render_policy=(
            ConnectedCameraRenderPolicy.INDIVIDUAL_LAYERS
        ),
    )


def _prepare_scene() -> object:
    _clear_scene()
    _purge_orphan_scene_data()
    _configure_scene()
    _create_camera()
    sentinel = bpy.data.objects.new("ConnectedMixedSequenceSentinel", None)
    bpy.context.scene.collection.objects.link(sentinel)
    sentinel.location = (12.0, 12.0, 12.0)
    sentinel.hide_render = True
    _activate_only(sentinel)
    return sentinel


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
    fixtures: tuple[_Fixture, ...],
) -> dict[str, tuple[Path, ...]]:
    png_paths = tuple(path for path in output_files if path.suffix.casefold() == ".png")
    expected_count = len(fixtures) * _SEQUENCE_FRAME_COUNT
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
    flattened = tuple(path for paths in result.values() for path in paths)
    _assert(
        len(flattened) == len(set(flattened)) == expected_count,
        "PNG paths overlap between connected or mixed objects",
    )
    return result


def _validate_images(
    groups: Mapping[str, tuple[Path, ...]],
    case: _Case,
) -> dict[str, tuple[int, int]]:
    dimensions: dict[str, tuple[int, int]] = {}
    payloads: list[bytes] = []
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
                f"Camera Projection crop changed across frames for {prefix}: {sizes}",
            )
            width, height = sizes[0]
            _assert(
                1 <= width <= _TEXTURE_SIZE and 1 <= height <= _TEXTURE_SIZE,
                f"Camera Projection crop outside 1..{_TEXTURE_SIZE}: {sizes[0]}",
            )
        dimensions[prefix] = sizes[0]
        payloads.extend(frame.file_bytes for frame in frames)

    _assert(
        len(payloads) == len({payload for payload in payloads}),
        "connected/mixed objects or frames produced duplicate PNG payloads",
    )
    return dimensions


def _visual_slots(
    document: Mapping[str, object],
    fixtures: tuple[_Fixture, ...],
) -> tuple[dict[str, object], ...]:
    prefixes = tuple(fixture.prefix for fixture in fixtures)
    slots = tuple(
        slot
        for slot in _json_array(document, "slots")
        if isinstance(slot, dict)
        and isinstance(slot.get("name"), str)
        and isinstance(slot.get("attachment"), str)
        and str(slot["name"]).startswith(prefixes)
    )
    _assert(
        len(slots) == len(fixtures),
        f"expected {len(fixtures)} visual slots, got "
        f"{tuple(slot.get('name') for slot in slots)}",
    )
    return slots


def _fixture_for_slot(slot_name: str, fixtures: tuple[_Fixture, ...]) -> _Fixture:
    matches = tuple(fixture for fixture in fixtures if slot_name.startswith(fixture.prefix))
    _assert(len(matches) == 1, f"slot {slot_name!r} does not map to one fixture")
    return matches[0]


def _assert_native_sequences(
    document: Mapping[str, object],
    case: _Case,
    fixtures: tuple[_Fixture, ...],
    dimensions: Mapping[str, tuple[int, int]],
) -> None:
    _assert(
        _TARGET.texture_animation_encoding
        is SpineTextureAnimationEncoding.NATIVE_SEQUENCE,
        "Spine 4.2 matrix must use native sequence encoding",
    )
    slots = _visual_slots(document, fixtures)
    for slot in slots:
        slot_name = str(slot["name"])
        setup_name = str(slot["attachment"])
        fixture = _fixture_for_slot(slot_name, fixtures)
        attachments = _slot_attachments(document, slot_name)
        _assert(
            tuple(attachments) == (setup_name,),
            f"{case.key}/{slot_name} must contain one native attachment: "
            f"{tuple(attachments)}",
        )
        attachment = attachments[setup_name]
        _assert(isinstance(attachment, dict), f"{slot_name} attachment is invalid")
        sequence = attachment.get("sequence")
        _assert(isinstance(sequence, dict), f"native sequence missing for {slot_name}")
        _assert(
            sequence.get("count") == _SEQUENCE_FRAME_COUNT,
            f"native sequence count is wrong for {slot_name}: {sequence}",
        )

        label = f"{case.key}/{slot_name}/{setup_name}"
        if case.texture_mode is A1TextureExportMode.NORMAL_UV_SEGMENTS:
            _assert_normal_attachment(attachment, fixture, label=label)
        else:
            _assert_projection_attachment(
                attachment,
                dimensions[fixture.prefix],
                label=label,
            )

    animations = document.get("animations", {})
    _assert(isinstance(animations, dict), "animations must be a JSON object")
    sequence_timelines = _lists_for_key(animations, "sequence")
    _assert(
        len(sequence_timelines) == len(fixtures),
        f"expected {len(fixtures)} native sequence timelines, "
        f"got {len(sequence_timelines)}",
    )
    for timeline in sequence_timelines:
        _assert(len(timeline) == 2, f"native Loop timeline is invalid: {timeline}")
        _assert(
            isinstance(timeline[0], dict) and timeline[0].get("mode") == "loop",
            f"native sequence timeline is not Loop: {timeline}",
        )
    _assert(_contains_key(document, "sequence"), "document lost native sequence data")


def _assert_composition(
    document: Mapping[str, object],
    case: _Case,
    fixtures: _CaseOutput,
) -> None:
    bones = {
        str(item.get("name")): item
        for item in _json_array(document, "bones")
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }
    _assert("all_objects" in bones, f"{case.key} lost connected wrapper main")
    _assert(
        any(name.startswith("all_objects_layer_") for name in bones),
        f"{case.key} lost connected depth layers",
    )

    for fixture in fixtures.connected_fixtures:
        main_name = f"{fixture.prefix}_main"
        _assert(main_name in bones, f"{case.key} lost connected main {main_name}")
        parent = str(bones[main_name].get("parent", ""))
        _assert(
            parent.startswith("all_objects_layer_"),
            f"{case.key} connected main has wrong parent: {bones[main_name]}",
        )

    if case.scope is A1MultiObjectMode.CONNECTED:
        _assert(
            not fixtures.standalone_fixtures,
            "connected case unexpectedly contains standalone fixtures",
        )
        return

    _assert(
        len(fixtures.standalone_fixtures) == 1,
        "mixed case must contain one standalone fixture",
    )
    standalone = fixtures.standalone_fixtures[0]
    standalone_main = f"{standalone.prefix}_main"
    _assert(standalone_main in bones, f"mixed case lost {standalone_main}")
    standalone_parent = str(bones[standalone_main].get("parent", ""))
    _assert(
        not standalone_parent.startswith("all_objects"),
        f"mixed standalone main entered connected hierarchy: {bones[standalone_main]}",
    )


def _assert_document(
    document: dict[str, object],
    case: _Case,
    fixtures: _CaseOutput,
    dimensions: Mapping[str, tuple[int, int]],
) -> None:
    skeleton = document.get("skeleton")
    _assert(isinstance(skeleton, dict), "skeleton metadata is missing")
    _assert(
        skeleton.get("spine") == _TARGET.exact_version,
        f"{case.key} version mismatch: {skeleton.get('spine')!r}",
    )
    _assert_bone_schema(document, _TARGET)
    _assert_constraint_schema(document, _TARGET)
    _assert_composition(document, case, fixtures)
    _assert_native_sequences(
        document,
        case,
        fixtures.fixtures,
        dimensions,
    )


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
    for fixture, expected_matrix in zip(fixtures, analysis_matrices, strict=True):
        _assert(
            _matrices_equal_at_float32_precision(
                _matrix_tuple(fixture.obj),
                expected_matrix,
            ),
            f"export did not restore matrix_world for {fixture.obj.name}",
        )
    _assert(
        not _temporary_datablock_names(),
        "connected/mixed export leaked temporary Blender datablocks",
    )


def _execute_public_export(
    fixtures: _CaseOutput,
    settings: A1MultiObjectExportSettings,
    *,
    context: object,
    scene: object,
):
    if settings.mode is A1MultiObjectMode.CONNECTED:
        return export_a1_multi_object(
            tuple(fixture.source for fixture in fixtures.connected_fixtures),
            settings,
            context=context,
            scene=scene,
        )
    if settings.mode is A1MultiObjectMode.MIXED:
        return export_a1_mixed_object(
            tuple(fixture.source for fixture in fixtures.connected_fixtures),
            tuple(fixture.source for fixture in fixtures.standalone_fixtures),
            settings,
            context=context,
            scene=scene,
        )
    raise ValueError(f"Unsupported export mode: {settings.mode!r}")


def _run_case(output_root: Path, case: _Case) -> None:
    case_directory = output_root / case.key
    case_directory.mkdir(parents=True, exist_ok=False)
    sentinel = _prepare_scene()
    fixtures = _build_case_fixtures(case_directory, case)
    _activate_only(sentinel)
    for fixture in fixtures.fixtures:
        fixture.obj.select_set(False)

    scene = bpy.context.scene
    scene.frame_set(_ANALYSIS_FRAME)
    bpy.context.view_layer.update()
    analysis_matrices = tuple(
        _matrix_tuple(fixture.obj) for fixture in fixtures.fixtures
    )
    for fixture, analysis_matrix in zip(
        fixtures.fixtures,
        analysis_matrices,
        strict=True,
    ):
        frame_matrices = _frame_matrices(scene, fixture.obj)
        _assert(
            not _matrices_equal_at_float32_precision(
                frame_matrices[0],
                frame_matrices[1],
            ),
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
        _material_fingerprint(fixture.material) for fixture in fixtures.fixtures
    )
    settings = _multi_settings(case_directory, case, fixtures)

    result = _execute_public_export(
        fixtures,
        settings,
        context=bpy.context,
        scene=scene,
    )
    _assert(result.success, f"{case.key} failed: {result.issues}")
    expected_output_count = 1 + case.object_count * _SEQUENCE_FRAME_COUNT
    _assert(
        len(result.output_files) == expected_output_count,
        f"{case.key} expected {expected_output_count} outputs: {result.output_files}",
    )
    json_path = result.output_files[0]
    _assert(json_path.suffix.casefold() == ".json", f"JSON is not first: {json_path}")
    _assert(json_path.is_file(), f"missing JSON: {json_path}")

    groups = _texture_groups(result.output_files, fixtures.fixtures)
    dimensions = _validate_images(groups, case)
    document = json.loads(json_path.read_text(encoding="utf-8"))
    _assert(isinstance(document, dict), f"{case.key} JSON root is not an object")
    _assert_document(document, case, fixtures, dimensions)
    _assert(
        result.statistics.get("object_count") == case.object_count,
        f"{case.key} statistics lost object_count: {result.statistics}",
    )
    _assert_state_restored(
        context_before=context_before,
        bake_state_before=bake_state_before,
        render_state_before=render_state_before,
        fixtures=fixtures.fixtures,
        material_fingerprints=material_fingerprints,
        analysis_matrices=analysis_matrices,
    )


def main() -> None:
    cases = _cases()
    failures: list[tuple[str, str]] = []
    started = time.perf_counter()
    print(f"Blender version: {bpy.app.version_string}")
    print(
        "[CONNECTED-MIXED-SEQUENCE-MATRIX] "
        f"cases={len(cases)} frames={_SEQUENCE_FRAME_COUNT} "
        f"texture={_TEXTURE_SIZE}x{_TEXTURE_SIZE} target={_TARGET.label}"
    )

    with tempfile.TemporaryDirectory(
        prefix="spine2d-connected-mixed-sequence-matrix-"
    ) as directory:
        output_root = Path(directory)
        for index, case in enumerate(cases, start=1):
            case_started = time.perf_counter()
            print(
                f"[CONNECTED-MIXED-SEQUENCE-MATRIX] RUN "
                f"{index}/{len(cases)} {case.key}"
            )
            try:
                _run_case(output_root, case)
            except Exception:
                failures.append((case.key, traceback.format_exc()))
                print(f"[CONNECTED-MIXED-SEQUENCE-MATRIX] FAIL {case.key}")
            else:
                elapsed = time.perf_counter() - case_started
                print(
                    f"[CONNECTED-MIXED-SEQUENCE-MATRIX] "
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
        f"[CONNECTED-MIXED-SEQUENCE-MATRIX] PASS "
        f"{len(cases)} cases ({elapsed:.2f}s total)"
    )


if __name__ == "__main__":
    main()
