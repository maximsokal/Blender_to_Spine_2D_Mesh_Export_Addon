import ast
from pathlib import Path


WORKER = Path(__file__).resolve().parents[1] / "tools" / "blender_a1_pipeline_probe.py"


def test_rewrite_probe_requires_all_staged_object_preparation_calls():
    source = WORKER.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(WORKER))
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_expected_calls"
    )
    strings = {
        node.value
        for node in ast.walk(function)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert {
        "blender_adapter.a1_source_geometry_preparation",
        "prepare_a1_source_geometry",
        "blender_adapter.a1_uv_preparation",
        "prepare_a1_uv",
        "blender_adapter.a1_texture_planning",
        "prepare_a1_texture_plan",
        "blender_adapter.a1_document_preparation",
        "prepare_a1_document",
    }.issubset(strings)
