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


def _attribute_path(node: ast.AST) -> str:
    parts = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    return ".".join(reversed(parts))


def _function_source(name: str, function_name: str) -> str:
    source = _source(name)
    node = next(
        item for item in _tree(name).body
        if isinstance(item, ast.FunctionDef) and item.name == function_name
    )
    return "\n".join(source.splitlines()[node.lineno - 1 : node.end_lineno])


def test_retired_bake_executor_files_are_absent():
    assert not (ADAPTER / "bake_executor.py").exists()
    assert not (ADAPTER / "bake_executor_core.py").exists()


def test_semantic_execution_is_the_only_object_bake_operator_owner():
    owners = []
    for path in ADAPTER.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if any(
            isinstance(node, ast.Attribute)
            and _attribute_path(node).endswith(".ops.object.bake")
            for node in ast.walk(tree)
        ):
            owners.append(path.name)
    assert owners == ["semantic_bake_execution.py"]


def test_semantic_modules_do_not_depend_on_retired_core():
    for filename in (
        "semantic_bake_validation.py",
        "semantic_bake_image_io.py",
        "semantic_bake_execution.py",
        "semantic_bake_output.py",
    ):
        source = _source(filename)
        assert "bake_executor_core" not in source
        assert "bake_executor" not in source


def test_image_io_cannot_own_transactions_or_blender_operators():
    source = _source("semantic_bake_image_io.py")
    for fragment in (
        "AtomicFileTransaction",
        "atomic_file_transaction",
        ".reserve(",
        ".commit(",
        ".ops.",
        "temporary_mesh_object",
        "temporary_bake_materials",
    ):
        assert fragment not in source


def test_reservations_are_validated_before_blender_mutation_scope():
    source = _function_source("semantic_bake_execution.py", "run_semantic_bake")
    assert source.index("validate_semantic_bake_reservations") < source.index(
        "preserve_bake_scene_state"
    )


def test_output_owner_exports_current_public_execution_surface():
    source = _source("semantic_bake_output.py")
    for name in (
        '"BakeExecutionError"',
        '"build_bake_execution_result"',
        '"execute_bake_plan"',
        '"stage_bake_plan_outputs"',
    ):
        assert name in source
