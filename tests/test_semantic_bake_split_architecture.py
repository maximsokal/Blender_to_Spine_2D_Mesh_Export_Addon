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
    tree = ast.parse(source, filename=name)
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == function_name
    )
    lines = source.splitlines()
    return "\n".join(lines[node.lineno - 1 : node.end_lineno])


def test_semantic_executor_is_a_small_compatibility_facade():
    tree = _tree("semantic_bake_executor.py")
    assert not any(
        isinstance(node, (ast.FunctionDef, ast.ClassDef)) for node in tree.body
    )
    assert "semantic_bake_output" in _source("semantic_bake_executor.py")


def test_validation_module_owns_no_transaction_or_mutation_scope():
    source = _source("semantic_bake_validation.py")
    assert "AtomicFileTransaction" not in source
    assert "atomic_file_transaction" not in source
    assert ".reserve(" not in source
    assert "temporary_mesh_object" not in source
    assert "temporary_bake_materials" not in source
    assert "def validate_semantic_bake_reservations" in source


def test_image_io_owns_only_image_uv_and_timeline_primitives():
    source = _source("semantic_bake_image_io.py")
    for function_name in (
        "_activate_uv_layer",
        "_create_bake_image",
        "_remove_image",
        "_save_bake_image",
        "_set_timeline_frame",
    ):
        assert f"def {function_name}" in source
    assert "atomic_file_transaction" not in source
    assert ".commit(" not in source
    assert ".ops." not in source


def test_execution_module_cannot_commit_or_create_transactions():
    source = _source("semantic_bake_execution.py")
    assert "AtomicFileTransaction" not in source
    assert "atomic_file_transaction" not in source
    assert ".commit(" not in source
    assert "def run_semantic_bake" in source
    assert "semantic_bake_image_io" in source
    assert "bake_executor_core" not in source


def test_stage_validates_before_any_reservation():
    source = _function_source(
        "semantic_bake_output.py",
        "stage_bake_plan_outputs",
    )
    assert source.index("validate_semantic_bake_request") < source.index(
        "_stage_validated_request"
    )


def test_direct_execution_validates_before_transaction_creation():
    source = _function_source("semantic_bake_output.py", "execute_bake_plan")
    assert source.index("validate_semantic_bake_request") < source.index(
        "atomic_file_transaction"
    )
    assert "committed_paths != expected_commit_order" in source


def test_typed_texture_gateway_validates_before_dispatch():
    source = _source("texture_executor.py")
    assert "class TextureExecutionRequest" in source
    assert "target_snapshot.source_object_id != self.plan.source_object_id" in source
    for name in (
        "stage_texture_plan_outputs",
        "stage_bake_plan_outputs",
        "execute_bake_plan",
    ):
        function = _function_source("texture_executor.py", name)
        assert "TextureExecutionRequest.capture" in function
