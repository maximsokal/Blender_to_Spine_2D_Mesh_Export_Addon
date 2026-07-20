import ast
from pathlib import Path


ADAPTER = (
    Path(__file__).resolve().parents[1]
    / "Blender_to_Spine2D_Mesh_Exporter"
    / "blender_adapter"
)


def _source(name: str) -> str:
    return (ADAPTER / name).read_text(encoding="utf-8")


def _tree(name: str) -> ast.Module:
    source = _source(name)
    return ast.parse(source, filename=name)


def _top_level_definitions(name: str):
    return tuple(
        node
        for node in _tree(name).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    )


def _function_source(name: str, function_name: str) -> str:
    source = _source(name)
    node = next(
        item
        for item in _tree(name).body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
        and item.name == function_name
    )
    lines = source.splitlines()
    return "\n".join(lines[node.lineno - 1 : node.end_lineno])


def test_shader_graph_analyzer_is_compatibility_only():
    assert _top_level_definitions("shader_graph_analyzer.py") == ()
    source = _source("shader_graph_analyzer.py")
    for owner in (
        "shader_graph_analysis",
        "shader_graph_error",
        "shader_graph_rna",
        "shader_graph_semantics",
        "shader_graph_traversal",
    ):
        assert owner in source
    assert "RecursiveShaderGraphWalker as _RecursiveGraphWalker" in source
    assert "derive_semantic_channels as _semantic_channels" in source
    assert "derive_material_dependencies as _dependencies" in source


def test_rna_owner_has_no_traversal_semantics_or_snapshot_ownership():
    source = _source("shader_graph_rna.py")
    for forbidden in (
        "RecursiveShaderGraphWalker",
        "ShaderGraphTraversalResult",
        "MaterialGraphSnapshot",
        "MaterialDependencyKind",
        "MaterialSemanticChannel",
        "ShaderNodeSnapshot",
        "walk_material_output",
    ):
        assert forbidden not in source
    assert "find_material_output" in source
    assert "matching_socket" in source
    assert "rna_identity" in source


def test_traversal_owner_does_not_classify_or_build_final_snapshot():
    source = _source("shader_graph_traversal.py")
    assert "class RecursiveShaderGraphWalker" in source
    assert "class ShaderGraphTraversalResult" in source
    assert "def build_result" in source
    for forbidden in (
        "MaterialDependencyKind",
        "MaterialSemanticChannel",
        "MaterialGraphSnapshot",
        "derive_semantic_channels",
        "derive_material_dependencies",
    ):
        assert forbidden not in source


def test_semantics_owner_reads_frozen_traversal_without_walking_links():
    source = _source("shader_graph_semantics.py")
    assert "ShaderGraphTraversalResult" in source
    assert "derive_semantic_channels" in source
    assert "derive_material_dependencies" in source
    for forbidden in (
        "ShaderLinkSnapshot",
        "find_material_output",
        "iter_links",
        "matching_socket",
        "walk_input",
        "walk_output",
    ):
        assert forbidden not in source


def test_snapshot_owner_preserves_parallel_deterministic_order():
    source = _source("shader_graph_snapshot.py")
    assert "sorted(" in source
    assert "key=lambda pair: pair[0].casefold()" in source
    assert "tuple(item.node for item in ordered_nodes)" in source
    assert "len(snapshot.reachable_nodes) != len(live_nodes)" in source
    for forbidden in (
        "find_material_output",
        "iter_nodes",
        "RecursiveShaderGraphWalker",
        "walk_material_output",
    ):
        assert forbidden not in source


def test_analysis_owner_coordinates_physical_layers_in_order():
    source = _function_source(
        "shader_graph_analysis.py",
        "analyse_material_graph_detailed",
    )
    assert source.index("find_material_output") < source.index(
        "RecursiveShaderGraphWalker("
    )
    assert source.index("walker.build_result()") < source.index(
        "derive_semantic_channels"
    )
    assert source.index("derive_semantic_channels") < source.index(
        "build_material_graph_snapshot"
    )
    assert "MaterialGraphAnalysisResult" in source


def test_material_and_production_callers_use_physical_analysis_owner():
    material_source = _source("material_graph_resolution.py")
    material_facade = _source("material_analyzer.py")
    production_source = _source("production_shader_capability_runtime.py")
    production_facade = _source("production_shader_capabilities.py")
    package_source = _source("__init__.py")

    assert "from .shader_graph_analysis import analyse_material_graph_detailed" in (
        material_source
    )
    assert "from .shader_graph_error import MaterialGraphAnalysisError" in material_source
    assert "from .shader_graph_analyzer import" not in material_source
    assert "shader_graph_analysis" not in material_facade

    assert "from .shader_graph_analysis import analyse_material_graph_detailed" in (
        production_source
    )
    assert "from .shader_graph_analyzer import" not in production_source
    assert "shader_graph_analysis" not in production_facade

    assert "from .shader_graph_analysis import analyse_material_graph" in package_source
    assert "from .shader_graph_error import MaterialGraphAnalysisError" in package_source
    assert "from .shader_graph_analyzer import" not in package_source


def test_facade_retains_historical_private_names():
    source = _source("shader_graph_analyzer.py")
    for name in (
        "_GraphFrame",
        "_ReachableNode",
        "_RecursiveGraphWalker",
        "_material_name",
        "_node_type",
        "_rna_identity",
        "_find_material_output",
        "_matching_socket",
        "_semantic_channels",
        "_dependencies",
        "analyse_material_graph",
        "analyse_material_graph_detailed",
    ):
        assert name in source
