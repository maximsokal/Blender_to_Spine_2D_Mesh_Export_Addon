from pathlib import Path


def test_geometry_domain_has_no_blender_or_random_dependencies():
    package_root = (
        Path(__file__).parents[1]
        / "Blender_to_Spine2D_Mesh_Exporter"
        / "domain"
        / "geometry"
    )
    forbidden_fragments = (
        "import bpy",
        "from bpy",
        "import bmesh",
        "from bmesh",
        "import random",
        "from random",
    )
    for path in package_root.glob("*.py"):
        source = path.read_text(encoding="utf-8")
        for forbidden in forbidden_fragments:
            assert forbidden not in source, f"{path.name} contains forbidden '{forbidden}'"
