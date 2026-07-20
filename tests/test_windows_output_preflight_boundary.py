import ast
from pathlib import Path


ROOT = Path(__file__).parents[1] / "Blender_to_Spine2D_Mesh_Exporter" / "blender_adapter"


def _call_lines(filename: str, function_name: str) -> list[int]:
    tree = ast.parse((ROOT / filename).read_text(encoding="utf-8"), filename=filename)
    result: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        target = node.func
        if isinstance(target, ast.Name) and target.id == function_name:
            result.append(node.lineno)
        elif isinstance(target, ast.Attribute) and target.attr == function_name:
            result.append(node.lineno)
    return result


def test_multi_output_preflight_runs_before_object_preparation():
    preflight = _call_lines(
        "a1_multi_object_export.py",
        "preflight_a1_output_namespace",
    )
    preparation = _call_lines("a1_multi_object_export.py", "prepare_a1_object")

    assert preflight
    assert preparation
    assert min(preflight) < min(preparation)


def test_mixed_output_preflight_runs_before_connected_or_standalone_preparation():
    preflight = _call_lines(
        "a1_mixed_object_export.py",
        "preflight_a1_output_namespace",
    )
    connected_preparation = _call_lines(
        "a1_mixed_object_export.py",
        "prepare_a1_multi_object",
    )
    standalone_preparation = _call_lines(
        "a1_mixed_object_export.py",
        "_prepare_standalone_objects",
    )

    assert preflight
    assert connected_preparation
    assert standalone_preparation
    assert min(preflight) < min(connected_preparation)
    assert min(preflight) < min(standalone_preparation)


def test_preflight_modules_remain_blender_independent():
    application_source = (
        Path(__file__).parents[1]
        / "Blender_to_Spine2D_Mesh_Exporter"
        / "application"
        / "a1_output_preflight.py"
    ).read_text(encoding="utf-8")
    naming_source = (
        Path(__file__).parents[1]
        / "Blender_to_Spine2D_Mesh_Exporter"
        / "domain"
        / "baking"
        / "output_naming.py"
    ).read_text(encoding="utf-8")

    for source in (application_source, naming_source):
        assert "import bpy" not in source
        assert "from bpy" not in source
        assert "import bmesh" not in source
        assert "from bmesh" not in source
