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


def _function_names(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def test_baking_domain_has_no_blender_dependencies():
    package_root = ROOT / "domain" / "baking"
    forbidden = ("import bpy", "from bpy", "import bmesh", "from bmesh")
    for path in package_root.glob("*.py"):
        source = path.read_text(encoding="utf-8")
        for fragment in forbidden:
            assert fragment not in source, (
                f"{path.name} contains forbidden '{fragment}'"
            )


def test_bake_helpers_use_no_operator_attributes():
    for filename in (
        "a1_mixed_object_output.py",
        "a1_multi_object_output.py",
        "a1_projection_finalization.py",
        "a1_single_object_export.py",
        "bake_compositor.py",
        "bake_execution_error.py",
        "bake_material_preparation.py",
        "bake_materials.py",
        "bake_scene_state.py",
        "camera_projection_error.py",
        "camera_projection_execution.py",
        "camera_projection_executor.py",
        "camera_projection_executor_core.py",
        "camera_projection_image.py",
        "camera_projection_output.py",
        "camera_projection_postprocess.py",
        "camera_projection_state.py",
        "camera_projection_validation.py",
        "grouped_camera_projection_execution.py",
        "grouped_camera_projection_executor.py",
        "grouped_camera_projection_output.py",
        "grouped_camera_projection_postprocess.py",
        "grouped_camera_projection_validation.py",
        "grouped_camera_projection_visibility.py",
        "material_analyzer.py",
        "semantic_bake_execution.py",
        "semantic_bake_executor.py",
        "semantic_bake_image_io.py",
        "semantic_bake_output.py",
        "semantic_bake_validation.py",
        "shader_graph_analyzer.py",
        "texture_executor.py",
    ):
        path = ROOT / "blender_adapter" / filename
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        operator_attributes = [
            _attribute_path(node)
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and ".ops." in f".{_attribute_path(node)}."
        ]
        assert not operator_attributes, (
            f"{filename} contains Blender operator access: "
            f"{operator_attributes}"
        )


def test_strategy_and_projection_domains_remain_blender_independent():
    for filename in (
        "strategies.py",
        "camera_projection.py",
        "projection_layout.py",
    ):
        path = ROOT / "domain" / "baking" / filename
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        imported_roots = {
            node.names[0].name.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import) and node.names
        }
        assert "bpy" not in imported_roots
        assert "bmesh" not in imported_roots
        assert "numpy" not in imported_roots


def test_camera_projection_postprocess_streams_one_union_buffer():
    path = (
        ROOT
        / "blender_adapter"
        / "camera_projection_postprocess.py"
    )
    source = path.read_text(encoding="utf-8")

    assert "ProjectionAlphaUnionAccumulator" in source
    assert "build_sequence_union_layout" not in source
    assert "masks: list" not in source
    assert "del coverage" in source


def test_object_bake_operator_is_confined_to_core_helper():
    path = ROOT / "blender_adapter" / "bake_executor_core.py"
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


def test_render_operator_is_confined_to_public_failure_injection_hook():
    path = ROOT / "blender_adapter" / "bake_executor.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    operator_attributes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and _attribute_path(node).endswith(".ops.render.render")
    ]

    assert operator_attributes
    function_names = {
        _function_for_line(tree, node.lineno).name
        for node in operator_attributes
        if _function_for_line(tree, node.lineno) is not None
    }
    assert function_names == {"_call_render_operator"}


def test_public_executor_is_a_small_facade():
    path = ROOT / "blender_adapter" / "bake_executor.py"
    source = path.read_text(encoding="utf-8")
    assert _function_names(path) == {
        "_call_bake_operator",
        "_call_render_operator",
    }
    assert "texture_executor" in source
    assert "bake_executor_core" in source
    assert "bake_execution_error" in source


def test_camera_projection_executor_is_a_small_facade():
    path = ROOT / "blender_adapter" / "camera_projection_executor.py"
    source = path.read_text(encoding="utf-8")
    assert _function_names(path) == set()
    assert "camera_projection_error" in source
    assert "camera_projection_output" in source
    assert "camera_projection_state" in source
    assert "camera_projection_executor_core" not in source
