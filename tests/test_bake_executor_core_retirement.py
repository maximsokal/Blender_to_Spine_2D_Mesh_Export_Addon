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
    source = _source(name)
    return ast.parse(source, filename=name)


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
    tree = ast.parse(source, filename=name)
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == function_name
    )
    lines = source.splitlines()
    return "\n".join(lines[node.lineno - 1 : node.end_lineno])


def test_retired_core_defines_only_operator_boundary():
    tree = _tree("bake_executor_core.py")
    definitions = [
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    ]
    assert definitions == ["_call_bake_operator"]

    source = _source("bake_executor_core.py")
    for fragment in (
        "atomic_file_transaction",
        "AtomicFileTransaction",
        ".reserve(",
        ".commit(",
        "temporary_mesh_object",
        "temporary_bake_materials",
        "preserve_bake_scene_state",
    ):
        assert fragment not in source


def test_core_reexports_compatibility_helpers_from_physical_owners():
    source = _source("bake_executor_core.py")
    assert "from .bake_execution_error import BakeExecutionError" in source
    assert "from .semantic_bake_image_io import" in source
    assert "from .semantic_bake_output import" in source
    assert "from .semantic_bake_validation import" in source
    assert "validate_semantic_bake_reservations as _require_reservations" in source

    for exported_name in (
        '"_activate_uv_layer"',
        '"_create_bake_image"',
        '"_load_bpy"',
        '"_remove_image"',
        '"_require_reservations"',
        '"_save_bake_image"',
        '"_set_timeline_frame"',
        '"_validate_execution_input"',
        '"build_bake_execution_result"',
        '"execute_bake_plan"',
        '"stage_bake_plan_outputs"',
    ):
        assert exported_name in source


def test_semantic_modules_no_longer_depend_on_retired_core():
    for filename in (
        "semantic_bake_validation.py",
        "semantic_bake_image_io.py",
        "semantic_bake_execution.py",
        "semantic_bake_output.py",
    ):
        assert "bake_executor_core" not in _source(filename)


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


def test_object_bake_operator_remains_confined_to_retired_core():
    owners = []
    for path in ADAPTER.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and _attribute_path(node).endswith(
                ".ops.object.bake"
            ):
                owners.append(path.name)
    assert owners == ["bake_executor_core.py"]


def test_public_executor_preserves_failure_injection_and_exports():
    source = _source("bake_executor.py")
    assert "_core._call_bake_operator" in source
    assert "from .bake_execution_error import BakeExecutionError" in source
    for exported_name in (
        '"BakeExecutionError"',
        '"build_bake_execution_result"',
        '"execute_bake_plan"',
        '"stage_bake_plan_outputs"',
    ):
        assert exported_name in source
