import ast
from pathlib import Path


ROOT = Path(__file__).parents[1] / "Blender_to_Spine2D_Mesh_Exporter"


def _attribute_path(node):
    parts = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    return ".".join(reversed(parts))


def _function_for_line(tree, line_number):
    matches = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            end = getattr(node, "end_lineno", node.lineno)
            if node.lineno <= line_number <= end:
                matches.append(node)
    if not matches:
        return None
    return max(matches, key=lambda node: node.lineno)


def test_baking_domain_has_no_blender_dependencies():
    package_root = ROOT / "domain" / "baking"
    forbidden = ("import bpy", "from bpy", "import bmesh", "from bmesh")
    for path in package_root.glob("*.py"):
        source = path.read_text(encoding="utf-8")
        for fragment in forbidden:
            assert fragment not in source, f"{path.name} contains forbidden '{fragment}'"


def test_bake_helpers_use_no_operators():
    for filename in (
        "bake_materials.py",
        "bake_scene_state.py",
        "material_analyzer.py",
    ):
        path = ROOT / "blender_adapter" / filename
        source = path.read_text(encoding="utf-8")
        assert "bpy.ops" not in source
        assert ".ops." not in source


def test_object_bake_operator_is_confined_to_one_helper():
    path = ROOT / "blender_adapter" / "bake_executor.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    operator_attributes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and _attribute_path(node).endswith(".ops.object.bake")
    ]

    assert operator_attributes
    function_names = {
        _function_for_line(tree, node.lineno).name
        for node in operator_attributes
        if _function_for_line(tree, node.lineno) is not None
    }
    assert function_names == {"_call_bake_operator"}
