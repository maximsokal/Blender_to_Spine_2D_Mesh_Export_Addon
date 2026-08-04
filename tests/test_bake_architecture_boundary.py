import ast
from pathlib import Path


ROOT = Path(__file__).parents[1] / "Blender_to_Spine2D_Mesh_Exporter"
ADAPTER = ROOT / "blender_adapter"


def _attribute_path(node: ast.AST) -> str:
    parts = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    return ".".join(reversed(parts))


def _function_for_line(tree: ast.Module, line_number: int) -> str | None:
    matches = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            end = getattr(node, "end_lineno", node.lineno)
            if node.lineno <= line_number <= end:
                matches.append(node)
    return max(matches, key=lambda node: node.lineno).name if matches else None


def _operator_owners() -> set[tuple[str, str | None, str]]:
    """Collect stateful bpy operators without classifying low-level bmesh.ops as bpy."""

    owners = set()
    for path in ADAPTER.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Attribute):
                continue
            attribute = _attribute_path(node)
            if attribute.startswith("bmesh.ops."):
                continue
            if ".ops." not in f".{attribute}.":
                continue
            # Only record the deepest concrete operator property, not parent chains.
            if attribute.endswith((".ops", ".ops.object", ".ops.render", ".ops.uv", ".ops.mesh")):
                continue
            owners.add((path.name, _function_for_line(tree, node.lineno), attribute))
    return owners


def test_baking_domain_has_no_blender_dependencies():
    forbidden = ("import bpy", "from bpy", "import bmesh", "from bmesh")
    for path in (ROOT / "domain" / "baking").glob("*.py"):
        source = path.read_text(encoding="utf-8")
        for fragment in forbidden:
            assert fragment not in source, f"{path.name} contains {fragment!r}"


def test_blender_operators_are_confined_to_four_physical_owners():
    owners = _operator_owners()
    assert owners
    owner_files = {filename for filename, _function, _attribute in owners}
    assert owner_files == {
        "camera_projection_execution.py",
        "context_state.py",
        "semantic_bake_execution.py",
        "uv_unwrap.py",
    }
    assert any(
        filename == "camera_projection_execution.py"
        and function == "_call_render_operator"
        and attribute.endswith(".ops.render.render")
        for filename, function, attribute in owners
    )
    assert any(
        filename == "semantic_bake_execution.py"
        and function == "_call_bake_operator"
        and attribute.endswith(".ops.object.bake")
        for filename, function, attribute in owners
    )


def test_strategy_and_projection_domains_remain_blender_independent():
    for filename in ("strategies.py", "camera_projection.py", "projection_layout.py"):
        path = ROOT / "domain" / "baking" / filename
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imported_roots = {
            alias.name.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        assert not imported_roots.intersection({"bpy", "bmesh", "numpy"})


def test_camera_projection_postprocess_streams_one_union_buffer():
    source = (ADAPTER / "camera_projection_postprocess.py").read_text(encoding="utf-8")
    assert "ProjectionAlphaUnionAccumulator" in source
    assert "build_sequence_union_layout" not in source
    assert "masks: list" not in source
    assert "del coverage" in source


def test_retired_monolithic_bake_executors_are_absent():
    for filename in (
        "bake_executor.py",
        "bake_executor_core.py",
        "camera_projection_executor.py",
        "semantic_bake_executor.py",
    ):
        assert not (ADAPTER / filename).exists(), filename


def test_camera_projection_core_is_a_definition_free_compatibility_export():
    path = ADAPTER / "camera_projection_executor_core.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    assert not any(isinstance(node, (ast.FunctionDef, ast.ClassDef)) for node in tree.body)
    assert "camera_projection_error" in source
    assert "camera_projection_output" in source
    assert ".ops." not in source
