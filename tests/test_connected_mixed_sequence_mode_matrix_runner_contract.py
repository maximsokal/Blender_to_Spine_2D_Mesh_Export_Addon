"""Static contracts for the connected/mixed real-Blender sequence matrix."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = (
    ROOT
    / "tests"
    / "blender_headless"
    / "run_connected_mixed_sequence_mode_matrix_integration.py"
)


def _tree() -> ast.Module:
    return ast.parse(RUNNER.read_text(encoding="utf-8"), filename=RUNNER.name)


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
    return tuple(_attribute_path(element) for element in value.elts)


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
        if not isinstance(candidate, ast.Call):
            continue
        name = _attribute_path(candidate.func)
        if name:
            result.append(name)
    return tuple(result)


def _string_constants(node: ast.AST) -> tuple[str, ...]:
    return tuple(
        candidate.value
        for candidate in ast.walk(node)
        if isinstance(candidate, ast.Constant) and isinstance(candidate.value, str)
    )


def _comparison_sources(node: ast.AST) -> tuple[str, ...]:
    return tuple(
        ast.unparse(candidate)
        for candidate in ast.walk(node)
        if isinstance(candidate, ast.Compare)
    )


def test_matrix_is_two_scopes_by_two_profiles_by_two_texture_modes() -> None:
    tree = _tree()

    assert _tuple_paths(tree, "_SCOPES") == (
        "A1MultiObjectMode.CONNECTED",
        "A1MultiObjectMode.MIXED",
    )
    assert _tuple_paths(tree, "_PROFILES") == (
        "A1RigProfile.THREE_AXIS_ROTATION",
        "A1RigProfile.TWO_AXIS_ROTATION_SCALE",
    )
    assert _tuple_paths(tree, "_TEXTURE_MODES") == (
        "A1TextureExportMode.NORMAL_UV_SEGMENTS",
        "A1TextureExportMode.CAMERA_PROJECTION",
    )

    cases = _function(tree, "_cases")
    generators = tuple(
        node for node in ast.walk(cases) if isinstance(node, ast.GeneratorExp)
    )
    assert len(generators) == 1
    assert tuple(
        _attribute_path(generator.iter)
        for generator in generators[0].generators
    ) == ("_SCOPES", "_PROFILES", "_TEXTURE_MODES")


def test_matrix_uses_spine42_two_frames_128px_and_one_sample() -> None:
    tree = _tree()

    target = _assignment(tree, "_TARGET")
    assert _attribute_path(target) == "SpineJsonTarget.SPINE_4_2"
    assert _constant(tree, "_TEXTURE_SIZE") == 128
    assert _constant(tree, "_SEQUENCE_FRAME_COUNT") == 2

    object_settings = _function(tree, "_object_settings")
    export_call = next(
        node
        for node in ast.walk(object_settings)
        if isinstance(node, ast.Call) and _attribute_path(node.func) == "ExportSettings"
    )
    export_keywords = {
        keyword.arg: keyword.value
        for keyword in export_call.keywords
        if keyword.arg is not None
    }
    for name in ("texture_width", "texture_height"):
        value = export_keywords[name]
        assert isinstance(value, ast.Name) and value.id == "_TEXTURE_SIZE"
    frame_count = export_keywords["sequence_frame_count"]
    assert isinstance(frame_count, ast.Name)
    assert frame_count.id == "_SEQUENCE_FRAME_COUNT"

    execution_call = next(
        node
        for node in ast.walk(object_settings)
        if isinstance(node, ast.Call)
        and _attribute_path(node.func) == "BakeExecutionSettings"
    )
    execution_keywords = {
        keyword.arg: keyword.value
        for keyword in execution_call.keywords
        if keyword.arg is not None
    }
    samples = execution_keywords["samples"]
    assert isinstance(samples, ast.Constant)
    assert samples.value == 1


def test_runner_uses_both_public_composition_output_services() -> None:
    tree = _tree()
    execute = _function(tree, "_execute_public_export")
    calls = _call_names(execute)

    assert calls.count("export_a1_multi_object") == 1
    assert calls.count("export_a1_mixed_object") == 1
    assert "prepare_a1_multi_object" not in calls
    assert "prepare_a1_mixed_object" not in calls
    assert "serialize_spine_document" not in calls

    settings = _function(tree, "_multi_settings")
    attributes = {
        _attribute_path(node)
        for node in ast.walk(settings)
        if isinstance(node, ast.Attribute)
    }
    assert "ConnectedCameraRenderPolicy.INDIVIDUAL_LAYERS" in attributes


def test_runner_validates_real_files_composition_sequences_and_state() -> None:
    tree = _tree()
    run_case_calls = set(_call_names(_function(tree, "_run_case")))

    assert {
        "_execute_public_export",
        "_texture_groups",
        "_validate_images",
        "json.loads",
        "_assert_document",
        "_assert_state_restored",
    }.issubset(run_case_calls)

    document_calls = set(_call_names(_function(tree, "_assert_document")))
    assert {
        "_assert_bone_schema",
        "_assert_constraint_schema",
        "_assert_composition",
        "_assert_native_sequences",
    }.issubset(document_calls)

    composition = _function(tree, "_assert_composition")
    composition_strings = _string_constants(composition)
    composition_calls = _call_names(composition)
    assert "all_objects" in composition_strings
    assert "all_objects_layer_" in composition_strings
    assert "standalone_parent.startswith" in composition_calls

    sequence = _function(tree, "_assert_native_sequences")
    sequence_attributes = {
        _attribute_path(node)
        for node in ast.walk(sequence)
        if isinstance(node, ast.Attribute)
    }
    sequence_strings = _string_constants(sequence)
    comparisons = _comparison_sources(sequence)
    assert "SpineTextureAnimationEncoding.NATIVE_SEQUENCE" in sequence_attributes
    assert "loop" in sequence_strings
    assert "count" in sequence_strings
    assert any("timeline[0].get" in value and "== 'loop'" in value for value in comparisons)
    assert any(
        "sequence.get" in value and "== _SEQUENCE_FRAME_COUNT" in value
        for value in comparisons
    )

    state_calls = set(_call_names(_function(tree, "_assert_state_restored")))
    assert "_material_fingerprint" in state_calls
    assert "_temporary_datablock_names" in state_calls
    assert "_scene_render_fingerprint" not in state_calls


def test_mixed_fixture_contains_two_connected_and_one_standalone_object() -> None:
    tree = _tree()
    case_output = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "_CaseOutput"
    )
    comparisons = _comparison_sources(case_output)

    assert any("len(self.connected_fixtures) != 2" in value for value in comparisons)
    assert any(
        "len(self.standalone_fixtures) not in {0, 1}" in value
        for value in comparisons
    )
    assert any(
        "self.fixtures != self.connected_fixtures + self.standalone_fixtures" in value
        for value in comparisons
    )
