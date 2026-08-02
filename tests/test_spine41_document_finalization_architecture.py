"""Architecture guard for canonical rig assembly and target document finalization."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ADAPTER = (
    ROOT
    / "Blender_to_Spine2D_Mesh_Exporter"
    / "blender_adapter"
    / "a1_document_preparation.py"
)
RIG_BUILDER = (
    ROOT
    / "Blender_to_Spine2D_Mesh_Exporter"
    / "domain"
    / "spine"
    / "rig_builder.py"
)


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    return next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _direct_name_calls(function: ast.FunctionDef) -> tuple[tuple[str, int], ...]:
    return tuple(
        (node.func.id, node.lineno)
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    )


def test_document_target_finalization_runs_after_both_assembly_paths() -> None:
    tree = _tree(ADAPTER)
    orchestrator = _function(tree, "prepare_a1_document")
    orchestrator_calls = _direct_name_calls(orchestrator)
    helper_lines = tuple(
        line
        for name, line in orchestrator_calls
        if name == "_assemble_document_for_texture"
    )
    assert len(helper_lines) == 1

    assembly_owner = _function(tree, "_assemble_document_for_texture")
    assert assembly_owner.end_lineno - assembly_owner.lineno + 1 < 140
    calls = _direct_name_calls(assembly_owner)

    normal_lines = tuple(
        line for name, line in calls if name == "assemble_a1_document"
    )
    camera_lines = tuple(
        line
        for name, line in calls
        if name == "assemble_a1_camera_projection_document"
    )
    finalizer_lines = tuple(
        line
        for name, line in calls
        if name == "finalize_a1_document_assembly_for_target"
    )

    assert len(normal_lines) == 1
    assert len(camera_lines) == 1
    assert len(finalizer_lines) == 1
    assert finalizer_lines[0] > normal_lines[0]
    assert finalizer_lines[0] > camera_lines[0]


def test_rig_builder_does_not_apply_target_constraint_mutations() -> None:
    tree = _tree(RIG_BUILDER)
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    imported_names = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }

    assert not any(module.endswith("two_axis_scale_spine38") for module in imported_modules)
    assert not any(module.endswith("two_axis_scale_spine41") for module in imported_modules)
    assert "adapt_two_axis_document_for_spine38_with_report" not in imported_names
    assert "adapt_two_axis_scale_rig_for_spine41" not in imported_names


def test_target_finalizer_routes_model_and_camera_adapters_before_synchronization() -> None:
    function = _function(
        _tree(ADAPTER),
        "finalize_a1_document_assembly_for_target",
    )
    calls = {name for name, _line in _direct_name_calls(function)}
    loaded_names = {
        node.id
        for node in ast.walk(function)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }

    required_adapters = {
        "adapt_two_axis_document_for_spine38_with_report",
        "adapt_two_axis_document_for_spine41_with_report",
        "adapt_camera_relative_two_axis_document_for_spine38_with_report",
        "adapt_camera_relative_two_axis_document_for_spine41_with_report",
    }
    assert required_adapters <= loaded_names
    assert "A1RigSetupPoseMode" in loaded_names
    assert "adapter" in calls
    assert "_synchronize_document_build_for_spine41" in calls
    assert "replace" in calls


def test_spine42_and_spine43_remain_on_the_canonical_document_path() -> None:
    source = ADAPTER.read_text(encoding="utf-8")
    function = _function(_tree(ADAPTER), "finalize_a1_document_assembly_for_target")
    function_source = ast.get_source_segment(source, function)

    assert function_source is not None
    assert "SpineJsonTarget.SPINE_4_2" in function_source
    assert "SpineJsonTarget.SPINE_4_3" in function_source
    identity_return = function_source.index("return document_assembly")
    spine38_adapter_reference = function_source.index(
        "adapt_two_axis_document_for_spine38_with_report"
    )
    spine41_adapter_reference = function_source.index(
        "adapt_two_axis_document_for_spine41_with_report"
    )
    assert identity_return < spine38_adapter_reference
    assert identity_return < spine41_adapter_reference
