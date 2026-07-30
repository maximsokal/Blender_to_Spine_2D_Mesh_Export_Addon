import ast
from pathlib import Path


ROOT = Path(__file__).parents[1] / "Blender_to_Spine2D_Mesh_Exporter" / "blender_adapter"


def _tree(filename: str) -> ast.Module:
    path = ROOT / filename
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _source(filename: str) -> str:
    return (ROOT / filename).read_text(encoding="utf-8")


def _top_level_functions(tree: ast.Module) -> set[str]:
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _annotated_fields(tree: ast.Module, class_name: str) -> set[str]:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return {
                statement.target.id
                for statement in node.body
                if isinstance(statement, ast.AnnAssign)
                and isinstance(statement.target, ast.Name)
            }
    raise AssertionError(f"class {class_name!r} not found")


def _call_lines(tree: ast.Module, function_name: str) -> list[int]:
    result = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id == function_name:
            result.append(node.lineno)
        elif isinstance(node.func, ast.Attribute) and node.func.attr == function_name:
            result.append(node.lineno)
    return result


def test_preparation_modules_have_no_output_entrypoints_or_side_effect_dependencies():
    forbidden_fragments = (
        "SpineSerializer",
        "atomic_file_transaction",
        "write_staged_utf8_text",
        "stage_bake_plan_outputs",
        "stage_texture_plan_outputs",
        "compose_spine_documents",
    )
    for filename in ("a1_multi_object_export.py", "a1_mixed_object_export.py"):
        source = _source(filename)
        functions = _top_level_functions(_tree(filename))
        assert "export_a1_multi_object" not in functions
        assert "export_a1_mixed_object" not in functions
        for fragment in forbidden_fragments:
            assert fragment not in source, f"{filename} contains forbidden {fragment}"


def test_prepared_multi_object_contract_owns_no_draft_document():
    fields = _annotated_fields(
        _tree("a1_multi_object_contracts.py"),
        "PreparedA1MultiObject",
    )
    assert fields == {
        "settings",
        "sources",
        "objects",
        "texture_output_paths",
        "warnings",
        "statistics",
    }
    assert "document" not in fields
    assert "composition" not in fields


def test_composition_module_is_blender_and_io_independent():
    source = _source("a1_multi_object_composition.py")
    for fragment in (
        "import bpy",
        "from bpy",
        "import bmesh",
        "from bmesh",
        "SpineSerializer",
        "atomic_file_transaction",
        "stage_texture_plan_outputs",
    ):
        assert fragment not in source


def test_shared_staging_finalizes_before_output_composition_and_serialization():
    staging_tree = _tree("a1_output_staging.py")
    staging = _call_lines(staging_tree, "stage_texture_plan_outputs")
    finalization = _call_lines(staging_tree, "finalize_prepared_camera_projection")
    assert staging and finalization and max(staging) < min(finalization)

    expectations = {
        "a1_multi_object_output.py": (
            "compose_a1_multi_object_document",
            "_serialize_composition",
        ),
        "a1_mixed_object_output.py": (
            "compose_a1_mixed_document",
            "_serialize",
        ),
    }
    for filename, (composition_name, serialization_name) in expectations.items():
        tree = _tree(filename)
        stage_calls = _call_lines(tree, "stage_and_finalize_a1_objects")
        composition = _call_lines(tree, composition_name)
        serialization = _call_lines(tree, serialization_name)
        assert stage_calls and composition and serialization
        assert max(stage_calls) < min(composition) < min(serialization)


def test_output_services_use_shared_staging_and_failure_entrypoints():
    for filename in ("a1_multi_object_output.py", "a1_mixed_object_output.py"):
        source = _source(filename)
        assert "from .a1_output_staging import stage_and_finalize_a1_objects" in source
        assert "build_multi_object_failure_result" in source
        assert "_compose_document" not in source
        assert "_record_object_statistics" not in source
