from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "Blender_to_Spine2D_Mesh_Exporter"
HEADLESS = ROOT / "tests" / "blender_headless"


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=path.name)


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _direct_call_names(function: ast.FunctionDef) -> tuple[str, ...]:
    names: list[str] = []
    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            names.append(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            names.append(node.func.attr)
    return tuple(names)


def test_multi_object_composition_builds_one_typed_analysis_per_object():
    tree = _tree(PACKAGE / "blender_adapter" / "a1_multi_object_composition.py")
    analyses = _function(tree, "_projected_object_analyses")
    depths = _function(tree, "_object_block_depths")

    assert "analyse_projected_object" in _direct_call_names(analyses)
    assert "_projected_object_analyses" in _direct_call_names(depths)
    assert any(
        isinstance(node, ast.Attribute) and node.attr == "block_depth"
        for node in ast.walk(depths)
    )


def test_connected_assembly_separates_anchor_placement_from_nearest_draw_order():
    tree = _tree(PACKAGE / "domain" / "spine" / "connected_group_assembly.py")
    build = _function(tree, "build_connected_group_document")
    argument_names = tuple(
        argument.arg for argument in (*build.args.args, *build.args.kwonlyargs)
    )
    calls = _direct_call_names(build)

    assert "object_block_depths" in argument_names
    assert "_place_projected_group_main_at_anchor" in calls
    assert "resolve_layers_and_placements" in calls
    assert "apply_connected_setup_draw_order" in calls


def test_mixed_outer_composition_runs_one_cross_group_object_block_pass():
    tree = _tree(PACKAGE / "blender_adapter" / "a1_mixed_composition.py")
    apply_mixed = _function(tree, "_apply_projected_mixed_draw_order")
    compose = _function(tree, "compose_a1_mixed_document")
    apply_calls = _direct_call_names(apply_mixed)
    compose_calls = _direct_call_names(compose)

    for expected in (
        "_shared_object_bake_projection_direction",
        "_document_components",
        "_object_block_depths",
        "apply_object_block_setup_draw_order",
    ):
        assert expected in apply_calls
    assert "_apply_projected_mixed_draw_order" in compose_calls
    assert any(
        isinstance(node, ast.Compare)
        and any(isinstance(operator, ast.Is) for operator in node.ops)
        and any(
            isinstance(comparator, ast.Constant) and comparator.value is None
            for comparator in node.comparators
        )
        for node in ast.walk(compose)
    )


def _assert_worker_contract(worker_name: str, *, camera_state: bool) -> None:
    worker = (HEADLESS / worker_name).read_text(encoding="utf-8")

    assert "prepare_a1_multi_object(" in worker
    assert "prepare_a1_mixed_object(" in worker
    assert "compose_a1_multi_object_document(" in worker
    assert "compose_a1_mixed_document(" in worker
    assert "maximumConnectedPositionDelta" in worker
    assert "connectedLayerFrontOrder" in worker
    assert "mixedObjectOrder" in worker
    assert "setupTransformModel" in worker
    assert "SPINE_AFFINE_NORMAL_ONLY_TRANSLATION" in worker
    assert "_translation_only_setup_position" not in worker
    assert "sourceUnchanged" in worker
    if camera_state:
        assert "evaluate_spine_setup_bone_position" in worker
        assert "cameraUnchanged" in worker
        assert "sceneRenderUnchanged" in worker
    else:
        assert "shared._setup_world_position" in worker


def test_blender_acceptance_covers_signed_axis_and_active_camera_routes():
    _assert_worker_contract(
        "run_axis_projection_connected_mixed_acceptance.py",
        camera_state=False,
    )
    _assert_worker_contract(
        "run_projection_connected_mixed_acceptance.py",
        camera_state=True,
    )


def test_setup_evaluator_is_blender_independent_and_fail_closed():
    source = (HEADLESS / "spine_setup_transform.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    assert "import bpy" not in source
    assert "onlyTranslation" in source
    assert "unsupported inherit mode" in source
    assert "parent cycle" in source
    assert "Missing parent bone" in source
    assert any(
        isinstance(node, ast.ClassDef) and node.name == "SpineSetupAffine2D"
        for node in tree.body
    )
    assert any(
        isinstance(node, ast.FunctionDef)
        and node.name == "evaluate_spine_setup_bone_position"
        for node in tree.body
    )
