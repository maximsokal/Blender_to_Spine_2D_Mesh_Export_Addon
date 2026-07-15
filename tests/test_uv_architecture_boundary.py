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


class OperatorLoopVisitor(ast.NodeVisitor):
    def __init__(self):
        self.loop_depth = 0
        self.operator_calls_inside_loops = []

    def visit_For(self, node):
        self.loop_depth += 1
        self.generic_visit(node)
        self.loop_depth -= 1

    def visit_While(self, node):
        self.loop_depth += 1
        self.generic_visit(node)
        self.loop_depth -= 1

    def visit_AsyncFor(self, node):
        self.loop_depth += 1
        self.generic_visit(node)
        self.loop_depth -= 1

    def visit_Call(self, node):
        path = _attribute_path(node.func)
        if self.loop_depth and (".ops." in path or path.startswith("bpy.ops.")):
            self.operator_calls_inside_loops.append((node.lineno, path))
        self.generic_visit(node)


def test_uv_domain_is_blender_independent():
    package_root = ROOT / "domain" / "uv"
    forbidden = ("import bpy", "from bpy", "import bmesh", "from bmesh")
    for path in package_root.glob("*.py"):
        source = path.read_text(encoding="utf-8")
        for fragment in forbidden:
            assert fragment not in source, f"{path.name} contains forbidden '{fragment}'"


def test_uv_operators_are_not_called_inside_python_loops():
    path = ROOT / "blender_adapter" / "uv_unwrap.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    visitor = OperatorLoopVisitor()
    visitor.visit(tree)
    assert visitor.operator_calls_inside_loops == []


def test_mesh_writer_uses_no_blender_operators():
    path = ROOT / "blender_adapter" / "mesh_writer.py"
    source = path.read_text(encoding="utf-8")
    assert "bpy.ops" not in source
    assert ".ops." not in source
