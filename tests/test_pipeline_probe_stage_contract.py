import ast
from pathlib import Path


WORKER = Path(__file__).resolve().parents[1] / "tools" / "blender_a1_pipeline_probe.py"


def _expected_call_strings() -> set[str]:
    source = WORKER.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(WORKER))
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_expected_calls"
    )
    return {
        node.value
        for node in ast.walk(function)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }


def test_rewrite_probe_requires_all_staged_object_preparation_calls():
    strings = _expected_call_strings()
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


def test_multi_probe_requires_shared_output_services():
    strings = _expected_call_strings()
    assert {
        "blender_adapter.a1_output_staging",
        "stage_and_finalize_a1_objects",
        "blender_adapter.a1_output_statistics",
        "record_final_document_statistics",
    }.issubset(strings)


def test_probe_tracks_ui_router_implementation_not_compatibility_facade():
    strings = _expected_call_strings()
    assert "blender_adapter.a1_ui_router" in strings
    assert "blender_adapter.a1_ui_bridge" not in strings
