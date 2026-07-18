import ast
from pathlib import Path


ROOT = Path(__file__).parents[1]
PACKAGE = ROOT / "Blender_to_Spine2D_Mesh_Exporter"


def _top_level_functions(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def test_pipeline_diagnostics_are_not_eagerly_imported_by_production_infrastructure():
    source = (PACKAGE / "infrastructure" / "__init__.py").read_text(encoding="utf-8")
    assert "pipeline_trace" not in source
    assert "pipeline_static_audit" not in source


def test_single_output_uses_shared_failure_builder_and_named_transaction():
    path = PACKAGE / "blender_adapter" / "a1_single_object_export.py"
    source = path.read_text(encoding="utf-8")
    functions = _top_level_functions(path)
    assert "_failure_result" not in functions
    assert "build_a1_failure_result" in source
    assert 'operation_name="a1-single-object"' in source


def test_probe_supports_module_and_file_focus():
    source = (ROOT / "tools" / "run_a1_pipeline_probe.py").read_text(
        encoding="utf-8"
    )
    assert "--focus-module" in source
    assert "--focus-file" in source
    assert "pipeline-static-audit.json" in source
    assert "pipeline-trace-report.json" in source
    assert 'runtime_report.get("trace", {})' in source
    assert 'runtime_report.get("race", {})' not in source
