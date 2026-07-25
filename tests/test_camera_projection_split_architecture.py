import ast
from pathlib import Path


ADAPTER = (
    Path(__file__).resolve().parents[1]
    / "Blender_to_Spine2D_Mesh_Exporter"
    / "blender_adapter"
)


def _source(name: str) -> str:
    return (ADAPTER / name).read_text(encoding="utf-8")


def _tree(name: str) -> ast.Module:
    return ast.parse(_source(name), filename=name)


def _function_source(name: str, function_name: str) -> str:
    source = _source(name)
    node = next(
        item for item in _tree(name).body
        if isinstance(item, ast.FunctionDef) and item.name == function_name
    )
    return "\n".join(source.splitlines()[node.lineno - 1 : node.end_lineno])


def test_camera_projection_core_is_compatibility_only():
    tree = _tree("camera_projection_executor_core.py")
    assert not any(isinstance(node, (ast.FunctionDef, ast.ClassDef)) for node in tree.body)
    source = _source("camera_projection_executor_core.py")
    assert "camera_projection_error" in source
    assert "camera_projection_output" in source
    for forbidden in (
        "ProjectionAlphaUnionAccumulator",
        "atomic_file_transaction",
        ".reserve(",
        ".commit(",
        ".ops.",
    ):
        assert forbidden not in source


def test_retired_public_projection_executor_is_absent():
    assert not (ADAPTER / "camera_projection_executor.py").exists()


def test_validation_owns_no_scene_mutation_or_output_lifecycle():
    source = _source("camera_projection_validation.py")
    assert "class CameraProjectionRuntime" in source
    assert "validate_camera_projection_request" in source
    assert "validate_camera_projection_reservations" in source
    for forbidden in (
        "preserve_camera_projection_state",
        "configure_scene_for_camera_projection",
        "_call_render_operator",
        "AtomicFileTransaction",
        "atomic_file_transaction",
        ".reserve(",
        ".commit(",
    ):
        assert forbidden not in source


def test_execution_owns_render_scope_but_no_postprocess_or_transaction():
    source = _source("camera_projection_execution.py")
    assert "preserve_camera_projection_state" in source
    assert "def _call_render_operator" in source
    assert "bpy_module.ops.render.render" in source
    for forbidden in (
        "ProjectionAlphaUnionAccumulator",
        "read_staged_alpha_coverage",
        "rewrite_staged_image_with_crop",
        "process_projection_outputs",
        "AtomicFileTransaction",
        "atomic_file_transaction",
        ".reserve(",
        ".commit(",
    ):
        assert forbidden not in source


def test_postprocess_owns_coverage_and_crop_without_scene_or_operator_access():
    source = _source("camera_projection_postprocess.py")
    assert "ProjectionAlphaUnionAccumulator" in source
    assert "read_staged_alpha_coverage" in source
    assert "rewrite_staged_image_with_crop" in source
    assert "def process_projection_outputs" in source
    for forbidden in (
        "preserve_camera_projection_state",
        "configure_scene_for_camera_projection",
        "_call_render_operator",
        "AtomicFileTransaction",
        "atomic_file_transaction",
        ".reserve(",
        ".commit(",
    ):
        assert forbidden not in source


def test_detailed_stage_validates_before_staging_validated_request():
    source = _function_source(
        "camera_projection_output.py",
        "stage_camera_projection_outputs_detailed",
    )
    assert source.index("validate_camera_projection_request") < source.index(
        "_stage_validated_camera_projection"
    )


def test_direct_execution_validates_before_transaction_creation():
    source = _function_source(
        "camera_projection_output.py",
        "execute_camera_projection_plan",
    )
    assert source.index("validate_camera_projection_request") < source.index(
        "atomic_file_transaction"
    )
    assert "committed_paths != expected_commit_order" in source
    assert source.count(".commit(") == 1


def test_postprocess_runs_only_after_reversible_render_returns():
    source = _function_source(
        "camera_projection_output.py",
        "_stage_validated_camera_projection",
    )
    assert source.index("render_camera_projection_frames") < source.index(
        "process_projection_outputs"
    )
    render_source = _function_source(
        "camera_projection_execution.py",
        "render_camera_projection_frames",
    )
    assert "process_projection_outputs" not in render_source


def test_core_retains_historical_export_aliases_without_logic():
    source = _source("camera_projection_executor_core.py")
    for name in (
        "_render_to_reservations",
        "_build_execution_result",
        "_reserve",
        "stage_camera_projection_outputs_detailed",
        "execute_camera_projection_plan",
    ):
        assert name in source


def test_image_layer_uses_shared_error_and_typed_domain_contracts():
    source = _source("camera_projection_image.py")
    assert "camera_projection_error" in source
    assert "ResolvedProjectionOutputPolicy" in source
    assert "AtomicOutputReservation" in source
