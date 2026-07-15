from pathlib import Path


def test_geometry_domain_does_not_import_blender_api():
    package_root = (
        Path(__file__).parents[1]
        / "Blender_to_Spine2D_Mesh_Exporter"
        / "domain"
        / "geometry"
    )
    for path in package_root.glob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert "import bpy" not in source
        assert "from bpy" not in source
        assert "import bmesh" not in source
        assert "from bmesh" not in source
