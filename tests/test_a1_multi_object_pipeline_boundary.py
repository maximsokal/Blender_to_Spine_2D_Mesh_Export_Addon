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
        target = node.func
        if isinstance(target, ast.Name) and target.id == function_name:
            result.append(node.lineno)
        elif isinstance(target, ast.Attribute) and target.attr == function_name:
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


def test_prepared_multi_object_does_not_own_a_draft_document():
    fields = _annotated_fields(
        _tree("a1_multi_object_export.py"),
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


def test_output_services_finalize_before_composition_and_serialize_after_composition():
    for filename in ("a1_multi_object_output.py", "a1_mixed_object_output.py"):
        tree = _tree(filename)
        finalization = _call_lines(tree, "finalize_prepared_camera_projection")
        composition = _call_lines(tree, "compose_a1_multi_object_document")
        serialization = _call_lines(tree, "to_json")

        assert finalization, f"{filename} has no projection finalization"
        assert composition, f"{filename} has no typed composition"
        assert serialization, f"{filename} has no serialization"
        assert max(finalization) < min(composition)
        assert max(composition) < min(serialization)


def test_output_services_share_public_composition_and_failure_entrypoints():
    for filename in ("a1_multi_object_output.py", "a1_mixed_object_output.py"):
        source = _source(filename)
        functions = _top_level_functions(_tree(filename))
        assert (
            "from .a1_multi_object_composition import "
            "compose_a1_multi_object_document"
        ) in source
        assert (
            "from .a1_multi_object_result import "
            "build_multi_object_failure_result"
        ) in source
        assert "_compose_document" not in source
        assert "_record_object_statistics" not in source
        assert "_failure_result" not in functions
