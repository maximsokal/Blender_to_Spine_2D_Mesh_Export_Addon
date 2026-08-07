import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "Blender_to_Spine2D_Mesh_Exporter"
FILTER = PACKAGE / "application" / "a1_projected_region_filter.py"
ASSEMBLY = PACKAGE / "application" / "a1_document_assembly.py"


def test_projected_region_filter_is_blender_independent_and_preserves_full_regions():
    source = FILTER.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=FILTER.name)
    imported_modules = {
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    } | {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }

    assert not any(
        module == "bpy" or module.startswith("bpy.")
        for module in imported_modules
    )
    assert not any(
        module == "bmesh" or module.startswith("bmesh.")
        for module in imported_modules
    )
    assert "bpy.ops" not in source
    assert "bmesh.new" not in source
    assert "extract_face_subset" not in source
    assert "return (snapshot,)" in source
    assert "without deleting edge-on faces" in source


def test_document_assembly_retains_prepared_regions_before_projection():
    source = ASSEMBLY.read_text(encoding="utf-8")

    assert "split_xy_visible_region_snapshots" in source
    assert "visible_region_snapshots = _xy_visible_region_snapshots(" in source
    assert "for region_offset, snapshot in enumerate(visible_region_snapshots):" in source
    assert source.index("visible_region_snapshots = _xy_visible_region_snapshots(") < source.index(
        "project_triangulated_disk_attachment("
    )
