import ast
from pathlib import Path


ADAPTER = (
    Path(__file__).resolve().parents[1]
    / "Blender_to_Spine2D_Mesh_Exporter"
    / "blender_adapter"
)


def _tree(name: str) -> ast.Module:
    return ast.parse((ADAPTER / name).read_text(encoding="utf-8"), filename=name)


def _function(name: str, function_name: str) -> ast.FunctionDef:
    return next(
        node for node in _tree(name).body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )


def test_multi_and_mixed_outputs_use_shared_staging_and_statistics_services():
    for filename, function_name in (
        ("a1_multi_object_output.py", "export_a1_multi_object"),
        ("a1_mixed_object_output.py", "export_a1_mixed_object"),
    ):
        function = _function(filename, function_name)
        calls = {
            node.func.id
            for node in ast.walk(function)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert "stage_and_finalize_a1_objects" in calls
        assert "record_final_document_statistics" in calls
        assert "stage_texture_plan_outputs" not in calls
        assert "finalize_prepared_camera_projection" not in calls
        assert "finalize_prepared_depth_camera_projection" not in calls
        assert function.end_lineno - function.lineno + 1 < 180


def test_output_routes_have_distinct_atomic_transaction_names():
    multi = (ADAPTER / "a1_multi_object_output.py").read_text(encoding="utf-8")
    mixed = (ADAPTER / "a1_mixed_object_output.py").read_text(encoding="utf-8")
    assert '_TRANSACTION_NAME = "a1-multi-object"' in multi
    assert '_TRANSACTION_NAME = "a1-mixed-object"' in mixed
    assert "operation_name=_TRANSACTION_NAME" in multi
    assert "operation_name=_TRANSACTION_NAME" in mixed


def test_shared_staging_service_owns_the_only_per_object_texture_loop():
    tree = _tree("a1_output_staging.py")
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "stage_and_finalize_a1_objects"
    )
    names = {
        node.func.id
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "stage_texture_plan_outputs" in names
    assert "finalize_prepared_rendered_projection" in names
    assert any(isinstance(node, ast.For) for node in ast.walk(function))

    dispatcher = (
        ADAPTER / "a1_rendered_projection_finalization.py"
    ).read_text(encoding="utf-8")
    assert "finalize_prepared_camera_projection" in dispatcher
    assert "finalize_prepared_depth_camera_projection" in dispatcher
