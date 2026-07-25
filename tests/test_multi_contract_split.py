import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "Blender_to_Spine2D_Mesh_Exporter" / "blender_adapter"
PROBE = ROOT / "tools" / "blender_a1_pipeline_probe.py"


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _imports(path: Path) -> set[str]:
    return {
        node.module
        for node in ast.walk(_tree(path))
        if isinstance(node, ast.ImportFrom) and node.module
    }


def test_multi_contracts_are_not_defined_by_preparation_implementation():
    definitions = {
        node.name
        for node in _tree(ADAPTER / "a1_multi_object_export.py").body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
    }
    assert {
        "A1MultiObjectSource",
        "PreparedA1MultiObject",
        "A1MultiObjectPreparationError",
        "record_object_statistics",
    }.isdisjoint(definitions)


def test_composition_staging_settings_and_export_plan_import_contracts_directly():
    for name in (
        "a1_multi_object_composition.py",
        "a1_output_staging.py",
        "a1_ui_settings.py",
        "a1_ui_export_plan.py",
    ):
        imports = _imports(ADAPTER / name)
        assert any(value.endswith("a1_multi_object_contracts") for value in imports), name

    router_imports = _imports(ADAPTER / "a1_ui_router.py")
    assert any(value.endswith("a1_ui_export_plan") for value in router_imports)
    assert not any(value.endswith("a1_multi_object_contracts") for value in router_imports)


def test_mixed_output_delegates_document_composition():
    tree = _tree(ADAPTER / "a1_mixed_object_output.py")
    call_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "compose_a1_mixed_document" in call_names
    assert "compose_spine_documents" not in call_names
    assert "compose_a1_multi_object_document" not in call_names


def test_mixed_composition_has_no_render_serialization_or_file_io():
    tree = _tree(ADAPTER / "a1_mixed_composition.py")
    call_names = {
        node.func.id if isinstance(node.func, ast.Name) else node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Name, ast.Attribute))
    }
    assert not call_names.intersection(
        {
            "stage_grouped_camera_projection_outputs",
            "to_json",
            "write_text",
            "write_bytes",
            "commit",
        }
    )


def test_runtime_probe_requires_mixed_composition_boundary():
    source = PROBE.read_text(encoding="utf-8")
    assert "blender_adapter.a1_mixed_composition" in source
    assert "compose_a1_mixed_document" in source
