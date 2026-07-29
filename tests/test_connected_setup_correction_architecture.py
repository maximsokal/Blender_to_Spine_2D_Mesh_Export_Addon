import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "Blender_to_Spine2D_Mesh_Exporter"
SPINE = PACKAGE / "domain" / "spine"
CORRECTION = SPINE / "connected_group_setup_correction.py"
ASSEMBLY = SPINE / "connected_group_assembly.py"


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_connected_setup_correction_is_blender_independent_and_data_only():
    source = _source(CORRECTION)
    tree = ast.parse(source, filename=str(CORRECTION))
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    } | {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }

    assert "bpy" not in imported_modules
    assert "bmesh" not in imported_modules
    assert "compose_spine_documents" not in source
    assert "SpineSerializer" not in source
    assert "MeshAttachment" not in source
    assert "decode_weighted_vertices" not in source
    assert "encode_weighted_vertices" not in source
    assert "uvs" not in source
    assert "triangles" not in source


def test_connected_assembly_applies_setup_correction_before_final_validation():
    source = _source(ASSEMBLY)

    correction_call = source.index("final_document = correct_connected_setup_pose(")
    validation_call = source.index("SpineValidator().validate_or_raise(final_document)")
    assert correction_call < validation_call
    assert "from .connected_group_setup_correction import" in source


def test_correction_rebuilds_immutable_documents_without_mutating_inputs():
    source = _source(CORRECTION)

    assert "from dataclasses import replace" in source
    assert "return replace(document, bones=tuple(bones))" in source
    assert "return replace(" in source
    assert ".append(" in source
    assert "bpy.ops" not in source
