"""Contracts for per-object static/sequence multi-export support."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

from Blender_to_Spine2D_Mesh_Exporter.application import (
    A1GeometryPreparationSettings,
    A1MultiObjectMode,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.a1_ui_export_plan import (
    _capture_selected_profiles,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.a1_ui_scene_capture import (
    _SceneExportProfile,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.a1_ui_settings import (
    _build_sources_from_profiles,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import BakeExecutionSettings
from Blender_to_Spine2D_Mesh_Exporter.domain.spine import (
    A1RigProfile,
    SpineJsonTarget,
)


ROOT = Path(__file__).resolve().parents[1]
STANDALONE_RUNNER = (
    ROOT
    / "tests"
    / "blender_headless"
    / "run_multi_object_mixed_static_sequence_matrix_integration.py"
)
CONNECTED_MIXED_RUNNER = (
    ROOT
    / "tests"
    / "blender_headless"
    / "run_connected_mixed_static_sequence_matrix_integration.py"
)


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=path.name)


def _assignment(tree: ast.Module, name: str) -> ast.AST:
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            return node.value
    raise AssertionError(f"Runner does not define {name}")


def _constant(tree: ast.Module, name: str) -> object:
    value = _assignment(tree, name)
    assert isinstance(value, ast.Constant), name
    return value.value


def _attribute_path(node: ast.AST) -> str:
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    return ".".join(reversed(parts))


def _tuple_paths(tree: ast.Module, name: str) -> tuple[str, ...]:
    value = _assignment(tree, name)
    assert isinstance(value, ast.Tuple), name
    return tuple(_attribute_path(item) for item in value.elts)


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    matches = tuple(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    assert len(matches) == 1, name
    return matches[0]


def _call_names(node: ast.AST) -> tuple[str, ...]:
    result = []
    for candidate in ast.walk(node):
        if isinstance(candidate, ast.Call):
            name = _attribute_path(candidate.func)
            if name:
                result.append(name)
    return tuple(result)


def _name_ids(node: ast.AST) -> set[str]:
    return {
        candidate.id
        for candidate in ast.walk(node)
        if isinstance(candidate, ast.Name)
    }


def _string_constants(node: ast.AST) -> tuple[str, ...]:
    return tuple(
        candidate.value
        for candidate in ast.walk(node)
        if isinstance(candidate, ast.Constant) and isinstance(candidate.value, str)
    )


def _scene_profile() -> _SceneExportProfile:
    return _SceneExportProfile(
        output_directory=Path("mixed-static-sequence-ui"),
        images_relative_path="images",
        texture_size=128,
        seam_mode="AUTO",
        angle_limit_degrees=30.0,
        geometry=A1GeometryPreparationSettings(),
        bake_execution=BakeExecutionSettings(),
        include_control_icons=False,
        include_preview_animation=False,
        spine_target=SpineJsonTarget.SPINE_4_2,
        rig_profile=A1RigProfile.TWO_AXIS_ROTATION_SCALE,
    )


def test_selected_ui_profiles_preserve_one_sequence_and_two_static_objects() -> None:
    objects = (
        SimpleNamespace(
            name="Animated",
            spine2d_bake_settings=SimpleNamespace(
                bake_frame_start=1,
                frames_for_render=2,
            ),
        ),
        SimpleNamespace(
            name="StaticA",
            spine2d_bake_settings=SimpleNamespace(
                bake_frame_start=0,
                frames_for_render=0,
            ),
        ),
        SimpleNamespace(
            name="StaticB",
            spine2d_bake_settings=SimpleNamespace(
                bake_frame_start=0,
                frames_for_render=0,
            ),
        ),
    )

    profiles = _capture_selected_profiles(objects)
    sources = _build_sources_from_profiles(profiles, _scene_profile())

    assert tuple(profile.sequence_frame_count for profile in profiles) == (2, 0, 0)
    assert tuple(profile.sequence_start_frame for profile in profiles) == (1, 0, 0)
    assert tuple(source.settings.export.sequence_frame_count for source in sources) == (
        2,
        0,
        0,
    )
    assert tuple(source.settings.export.sequence_start_frame for source in sources) == (
        1,
        0,
        0,
    )
    assert sources[0].settings is not sources[1].settings
    assert sources[1].settings is not sources[2].settings
    assert sources[0].settings is not sources[2].settings


def test_standalone_runner_covers_all_targets_and_both_texture_modes() -> None:
    tree = _tree(STANDALONE_RUNNER)

    assert _tuple_paths(tree, "_TARGETS") == (
        "SpineJsonTarget.SPINE_3_8",
        "SpineJsonTarget.SPINE_4_0",
        "SpineJsonTarget.SPINE_4_1",
        "SpineJsonTarget.SPINE_4_2",
        "SpineJsonTarget.SPINE_4_3",
    )
    assert _tuple_paths(tree, "_TEXTURE_MODES") == (
        "A1TextureExportMode.NORMAL_UV_SEGMENTS",
        "A1TextureExportMode.CAMERA_PROJECTION",
    )
    assert _constant(tree, "_TEXTURE_SIZE") == 128
    assert _constant(tree, "_SEQUENCE_FRAME_COUNT") == 2
    assert _constant(tree, "_OBJECT_COUNT") == 3
    assert _constant(tree, "_EXPECTED_TEXTURE_COUNT") == 4

    run_case_calls = set(_call_names(_function(tree, "_run_case")))
    assert "export_a1_multi_object" in run_case_calls
    assert "json.loads" in run_case_calls
    assert "_assert_document" in run_case_calls
    assert "_assert_state_restored" in run_case_calls


def test_standalone_runner_checks_static_objects_have_no_sequence_data() -> None:
    tree = _tree(STANDALONE_RUNNER)
    function = _function(tree, "_assert_texture_encoding")
    calls = set(_call_names(function))
    strings = set(_string_constants(function))

    assert "_slot_timelines" in calls
    assert "sequence" in strings
    assert "attachment" in strings
    assert "static attachment inherited native sequence metadata: " in strings
    assert "static slot inherited native sequence timeline: " in strings
    assert "static slot inherited attachment timeline: " in strings


def test_connected_mixed_runner_covers_both_profiles_modes_and_sequence_locations() -> None:
    tree = _tree(CONNECTED_MIXED_RUNNER)

    assert _tuple_paths(tree, "_PROFILES") == (
        "A1RigProfile.THREE_AXIS_ROTATION",
        "A1RigProfile.TWO_AXIS_ROTATION_SCALE",
    )
    assert _tuple_paths(tree, "_TEXTURE_MODES") == (
        "A1TextureExportMode.NORMAL_UV_SEGMENTS",
        "A1TextureExportMode.CAMERA_PROJECTION",
    )
    assert _constant(tree, "_TEXTURE_SIZE") == 128
    assert _constant(tree, "_SEQUENCE_FRAME_COUNT") == 2
    assert _constant(tree, "_SEQUENCE_CONNECTED") == "CONNECTED_OBJECT"
    assert _constant(tree, "_SEQUENCE_STANDALONE") == "STANDALONE_OBJECT"

    cases = _function(tree, "_cases")
    assert "_Case" in set(_call_names(cases))
    assert {
        "_PROFILES",
        "_TEXTURE_MODES",
        "_SEQUENCE_CONNECTED",
        "_SEQUENCE_STANDALONE",
    }.issubset(_name_ids(cases))

    run_case_calls = set(_call_names(_function(tree, "_run_case")))
    assert "_execute_public_export" in run_case_calls
    assert "_assert_document" in run_case_calls
    assert "_assert_state_restored" in run_case_calls


def test_connected_mixed_runner_uses_public_supported_scopes() -> None:
    tree = _tree(CONNECTED_MIXED_RUNNER)
    case_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "_Case"
    )
    attributes = {
        _attribute_path(node)
        for node in ast.walk(case_class)
        if isinstance(node, ast.Attribute)
    }
    assert "A1MultiObjectMode.CONNECTED" in attributes
    assert "A1MultiObjectMode.MIXED" in attributes

    multi_settings = _function(tree, "_multi_settings")
    settings_attributes = {
        _attribute_path(node)
        for node in ast.walk(multi_settings)
        if isinstance(node, ast.Attribute)
    }
    assert "ConnectedCameraRenderPolicy.INDIVIDUAL_LAYERS" in settings_attributes


def test_connected_mixed_runner_checks_only_one_native_sequence() -> None:
    tree = _tree(CONNECTED_MIXED_RUNNER)
    function = _function(tree, "_assert_native_sequences")
    strings = set(_string_constants(function))
    calls = set(_call_names(function))

    assert "_slot_sequence_timelines" in calls
    assert "native sequence missing for " in strings
    assert "static attachment inherited sequence metadata: " in strings
    assert "static slot inherited sequence timeline: " in strings


def test_ui_multi_export_remains_standalone_public_contract() -> None:
    objects = (
        SimpleNamespace(
            name="Animated",
            spine2d_bake_settings=SimpleNamespace(
                bake_frame_start=1,
                frames_for_render=2,
            ),
        ),
        SimpleNamespace(
            name="Static",
            spine2d_bake_settings=SimpleNamespace(
                bake_frame_start=0,
                frames_for_render=0,
            ),
        ),
    )
    profiles = _capture_selected_profiles(objects)
    sources = _build_sources_from_profiles(profiles, _scene_profile())

    assert len(sources) == 2
    assert tuple(source.settings.export.sequence_frame_count for source in sources) == (
        2,
        0,
    )
    # This test owns only the object-level timing contract. The public selected-object
    # mode remains standalone and is guarded by test_a1_ui_export_plan.py.
    assert A1MultiObjectMode.STANDALONE.value == "STANDALONE"
