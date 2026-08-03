"""Static contracts for the real-Blender multi-object sequence matrix runner."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = (
    ROOT
    / "tests"
    / "blender_headless"
    / "run_multi_object_sequence_mode_matrix_integration.py"
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


def _constant_assignment(tree: ast.Module, name: str) -> object:
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


def _tuple_attribute_paths(tree: ast.Module, name: str) -> tuple[str, ...]:
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
    names: list[str] = []
    for candidate in ast.walk(node):
        if not isinstance(candidate, ast.Call):
            continue
        path = _attribute_path(candidate.func)
        if path:
            names.append(path)
    return tuple(names)


def _string_constants(node: ast.AST) -> tuple[str, ...]:
    return tuple(
        candidate.value
        for candidate in ast.walk(node)
        if isinstance(candidate, ast.Constant) and isinstance(candidate.value, str)
    )


def test_matrix_is_exactly_five_targets_by_two_texture_modes() -> None:
    tree = _tree()

    assert _tuple_attribute_paths(tree, "_TARGETS") == (
        "SpineJsonTarget.SPINE_3_8",
        "SpineJsonTarget.SPINE_4_0",
        "SpineJsonTarget.SPINE_4_1",
        "SpineJsonTarget.SPINE_4_2",
        "SpineJsonTarget.SPINE_4_3",
    )
    assert _tuple_attribute_paths(tree, "_TEXTURE_MODES") == (
        "A1TextureExportMode.NORMAL_UV_SEGMENTS",
        "A1TextureExportMode.CAMERA_PROJECTION",
    )

    cases = _function(tree, "_cases")
    comprehensions = tuple(
        node for node in ast.walk(cases) if isinstance(node, ast.GeneratorExp)
    )
    assert len(comprehensions) == 1
    iter_paths = tuple(
        _attribute_path(generator.iter)
        for generator in comprehensions[0].generators
    )
    assert iter_paths == ("_TARGETS", "_TEXTURE_MODES")


def test_matrix_uses_two_objects_two_frames_and_128px_textures() -> None:
    tree = _tree()

    assert _constant_assignment(tree, "_TEXTURE_SIZE") == 128
    assert _constant_assignment(tree, "_SEQUENCE_FRAME_COUNT") == 2
    assert _constant_assignment(tree, "_OBJECT_COUNT") == 2

    object_settings = _function(tree, "_object_settings")
    names = _call_names(object_settings)
    assert "ExportSettings" in names
    assert "BakeExecutionSettings" in names

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
    for field_name in ("texture_width", "texture_height"):
        value = export_keywords[field_name]
        assert isinstance(value, ast.Name)
        assert value.id == "_TEXTURE_SIZE"
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


def test_runner_uses_public_standalone_multi_object_export() -> None:
    tree = _tree()
    run_case = _function(tree, "_run_case")
    call_names = _call_names(run_case)

    assert call_names.count("export_a1_multi_object") == 1
    assert "export_a1_single_object" not in call_names
    assert "prepare_a1_multi_object" not in call_names
    assert "serialize_spine_document" not in call_names

    multi_settings = _function(tree, "_multi_settings")
    standalone_attributes = tuple(
        _attribute_path(node)
        for node in ast.walk(multi_settings)
        if isinstance(node, ast.Attribute)
    )
    assert "A1MultiObjectMode.STANDALONE" in standalone_attributes


def test_runner_validates_real_files_runtime_schema_and_state_restoration() -> None:
    tree = _tree()
    run_case_calls = set(_call_names(_function(tree, "_run_case")))

    assert {
        "_texture_paths_by_source",
        "_validate_texture_outputs",
        "json.loads",
        "_assert_document",
        "_assert_state_restored",
    }.issubset(run_case_calls)

    assert "_assert_bone_schema" in _call_names(_function(tree, "_assert_document"))
    assert "_assert_constraint_schema" in _call_names(_function(tree, "_assert_document"))
    assert "_read_image" in _call_names(_function(tree, "_read_image_summary"))
    assert "_material_fingerprint" in _call_names(
        _function(tree, "_assert_state_restored")
    )
    assert "_temporary_datablock_names" in _call_names(
        _function(tree, "_assert_state_restored")
    )


def test_sequence_contract_covers_legacy_swap_and_native_sequence() -> None:
    tree = _tree()
    function = _function(tree, "_assert_sequence_encoding")
    attributes = {
        _attribute_path(node)
        for node in ast.walk(function)
        if isinstance(node, ast.Attribute)
    }
    calls = _call_names(function)
    strings = _string_constants(function)
    comparisons = tuple(
        ast.unparse(node)
        for node in ast.walk(function)
        if isinstance(node, ast.Compare)
    )

    assert "SpineTextureAnimationEncoding.NATIVE_SEQUENCE" in attributes
    assert "_contains_key" in calls
    assert "sequence" in strings
    assert "loop" in strings
    assert any("_SEQUENCE_FRAME_COUNT + 1" in value for value in comparisons)
    assert any("names[-1] == names[0]" in value for value in comparisons)
