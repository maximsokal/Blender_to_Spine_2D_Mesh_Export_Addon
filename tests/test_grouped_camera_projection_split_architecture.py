import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ADAPTER = (
    ROOT
    / "Blender_to_Spine2D_Mesh_Exporter"
    / "blender_adapter"
)


def _source(name: str) -> str:
    return (ADAPTER / name).read_text(encoding="utf-8")


def _tree(name: str) -> ast.Module:
    source = _source(name)
    return ast.parse(source, filename=name)


def _function_source(name: str, function_name: str) -> str:
    source = _source(name)
    tree = ast.parse(source, filename=name)
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.FunctionDef)
        and item.name == function_name
    )
    lines = source.splitlines()
    return "\n".join(lines[node.lineno - 1 : node.end_lineno])


def test_grouped_executor_is_compatibility_only():
    tree = _tree("grouped_camera_projection_executor.py")
    assert not any(
        isinstance(node, (ast.FunctionDef, ast.ClassDef))
        for node in tree.body
    )
    source = _source("grouped_camera_projection_executor.py")
    assert "grouped_camera_projection_output" in source
    assert "grouped_camera_projection_validation" in source
    assert "grouped_camera_projection_visibility" in source
    for forbidden in (
        "preserve_camera_projection_state",
        "ProjectionAlphaUnionAccumulator",
        "read_staged_alpha_coverage",
        "rewrite_staged_image_with_crop",
        "resolve_projection_output_policy",
        ".reserve(",
        ".commit(",
    ):
        assert forbidden not in source


def test_grouped_validation_owns_no_mutation_operator_or_output_lifecycle():
    source = _source("grouped_camera_projection_validation.py")
    assert "class GroupedCameraProjectionRuntime" in source
    assert "validate_grouped_camera_projection_request" in source
    assert "validate_grouped_camera_projection_reservations" in source
    assert "resolve_projection_output_policy" in source
    assert "duplicate output paths" in source
    assert "reservation.final_path != expected" in source
    for forbidden in (
        "preserve_camera_projection_state",
        "configure_group_camera_visibility",
        "_call_render_operator",
        "ProjectionAlphaUnionAccumulator",
        "read_staged_alpha_coverage",
        "rewrite_staged_image_with_crop",
        "AtomicFileTransaction",
        "atomic_file_transaction",
        ".reserve(",
        ".commit(",
    ):
        assert forbidden not in source


def test_grouped_visibility_owns_only_group_camera_mutation():
    source = _source("grouped_camera_projection_visibility.py")
    assert "configure_group_camera_visibility" in source
    assert "obj.hide_render = False" in source
    assert "obj.visible_camera = True" in source
    assert "obj.visible_camera = False" in source
    for forbidden in (
        "validate_grouped_camera_projection_request",
        "resolve_projection_output_policy",
        "_call_render_operator",
        "preserve_camera_projection_state",
        "ProjectionAlphaUnionAccumulator",
        "AtomicFileTransaction",
        ".reserve(",
        ".commit(",
    ):
        assert forbidden not in source


def test_grouped_execution_owns_reversible_render_but_no_postprocess():
    source = _source("grouped_camera_projection_execution.py")
    assert "preserve_camera_projection_state" in source
    assert "configure_group_camera_visibility" in source
    assert "_call_render_operator" in source
    assert "require_nonempty_grouped_staged_output" in source
    for forbidden in (
        "ProjectionAlphaUnionAccumulator",
        "read_staged_alpha_coverage",
        "rewrite_staged_image_with_crop",
        "process_grouped_camera_projection_outputs",
        "AtomicFileTransaction",
        "atomic_file_transaction",
        ".reserve(",
        ".commit(",
    ):
        assert forbidden not in source


def test_shared_postprocess_engine_serves_single_and_grouped_b4():
    shared = _source("camera_projection_postprocess.py")
    grouped = _source("grouped_camera_projection_postprocess.py")
    assert "class ProjectionPostprocessRequest" in shared
    assert "def process_projection_outputs" in shared
    assert "def process_camera_projection_outputs" not in shared
    assert "ProjectionAlphaUnionAccumulator" in shared
    assert "read_staged_alpha_coverage" in shared
    assert "rewrite_staged_image_with_crop" in shared
    assert "del coverage" in shared
    assert "process_projection_outputs" in grouped
    assert "ProjectionAlphaUnionAccumulator" not in grouped
    assert "read_staged_alpha_coverage" not in grouped
    assert "rewrite_staged_image_with_crop" not in grouped


def test_grouped_postprocess_has_no_scene_operator_or_output_lifecycle():
    source = _source("grouped_camera_projection_postprocess.py")
    assert "process_grouped_camera_projection_outputs" in source
    assert "log_grouped_camera_projection_layout" in source
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


def test_grouped_stage_validates_before_transaction_and_reservation():
    source = _function_source(
        "grouped_camera_projection_output.py",
        "stage_grouped_camera_projection_outputs",
    )
    assert source.index(
        "validate_grouped_camera_projection_request"
    ) < source.index("require_grouped_transaction")
    assert source.index(
        "validate_grouped_camera_projection_request"
    ) < source.index("stage_validated_grouped_camera_projection")


def test_grouped_validated_stage_restores_before_postprocess():
    source = _function_source(
        "grouped_camera_projection_output.py",
        "stage_validated_grouped_camera_projection",
    )
    assert source.index(
        "render_grouped_camera_projection_frames"
    ) < source.index("process_grouped_camera_projection_outputs")

    render_source = _function_source(
        "grouped_camera_projection_execution.py",
        "render_grouped_camera_projection_frames",
    )
    assert "process_grouped_camera_projection_outputs" not in render_source
    assert "read_staged_alpha_coverage" not in render_source


def test_grouped_output_never_creates_or_commits_transaction():
    source = _source("grouped_camera_projection_output.py")
    assert "AtomicFileTransaction" in source
    assert "atomic_file_transaction" not in source
    assert ".commit(" not in source
    assert "transaction.reserve" in source


def test_grouped_output_reservation_order_is_revalidated():
    reserve_source = _function_source(
        "grouped_camera_projection_output.py",
        "reserve_grouped_camera_projection_outputs",
    )
    assert "validate_grouped_camera_projection_reservations" in reserve_source

    validation_source = _function_source(
        "grouped_camera_projection_validation.py",
        "validate_grouped_camera_projection_reservations",
    )
    assert "zip(" in validation_source
    assert "strict=True" in validation_source
    assert "reservation.final_path != expected" in validation_source


def test_grouped_facade_retains_historical_private_names():
    source = _source("grouped_camera_projection_executor.py")
    for name in (
        "GroupedCameraProjectionRuntime",
        "GroupedCameraProjectionStageResult",
        "_object_name",
        "_rna_identity",
        "_validate_group_runtime",
        "_configure_group_camera_visibility",
        "_reserve_group_outputs",
        "stage_grouped_camera_projection_outputs",
    ):
        assert name in source


def test_production_callers_import_physical_grouped_output_owner():
    for filename in (
        "a1_multi_object_output.py",
        "a1_mixed_object_output.py",
        "a1_grouped_output.py",
        "a1_output_statistics.py",
    ):
        source = _source(filename)
        assert "grouped_camera_projection_output" in source
        assert "grouped_camera_projection_executor" not in source


def test_crop_writer_accepts_single_and_grouped_plan_types():
    source = _function_source(
        "camera_projection_image.py",
        "rewrite_staged_image_with_crop",
    )
    assert "CameraProjectionPlan | GroupedCameraProjectionPlan" in source
    assert "(CameraProjectionPlan, GroupedCameraProjectionPlan)" in source
