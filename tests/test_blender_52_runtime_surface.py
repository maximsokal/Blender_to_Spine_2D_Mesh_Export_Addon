"""Regressions for the Rewrite-only Blender 5.2 runtime surface."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.material_analysis_error import (
    MaterialAnalysisError,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.material_slot_analysis import (
    analyse_material_slot,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.shader_graph_analysis import (
    analyse_material_graph_detailed,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.shader_graph_error import (
    MaterialGraphAnalysisError,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.shader_graph_rna import (
    find_material_output,
    node_type,
    normalise_render_target,
)


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "Blender_to_Spine2D_Mesh_Exporter"


def _source(relative_path: str) -> str:
    return (PACKAGE / relative_path).read_text(encoding="utf-8")


def test_extension_startup_does_not_import_legacy_runtime():
    entry = _source("__init__.py")

    assert "legacy_loader" not in entry
    assert "install_legacy_multi_facade" not in entry
    assert "Legacy implementation modules remain lazy" not in entry


def test_single_object_operator_is_rewrite_only():
    source = _source("single_object_operator.py")

    assert "legacy_loader" not in source
    assert "load_legacy_single_backend" not in source
    assert "get_texture_size" not in source
    assert "SINGLE_BACKEND_PROPERTY" not in source
    assert '"LEGACY"' not in source
    assert "export_active_object_a1(context)" in source


def test_multi_object_operator_is_rewrite_only():
    source = _source("ui.py")

    assert "from .multi_object_export import" not in source
    assert "MULTI_BACKEND_PROPERTY" not in source
    assert "resolve_multi_backend" not in source
    assert 'backend == "LEGACY"' not in source
    assert "export_selected_objects_a1(context)" in source


def test_package_import_surface_no_longer_references_removed_config_api():
    combined = "\n".join(
        (
            _source("__init__.py"),
            _source("single_object_operator.py"),
            _source("ui.py"),
        )
    )

    assert "config.get_texture_size" not in combined
    assert "from .config import get_texture_size" not in combined


def test_shader_target_contract_accepts_only_blender_52_values():
    assert normalise_render_target(None) == "ALL"
    assert normalise_render_target("ALL") == "ALL"
    assert normalise_render_target("CYCLES") == "CYCLES"
    assert normalise_render_target("EEVEE") == "EEVEE"
    assert normalise_render_target("BLENDER_EEVEE") == "EEVEE"

    for old_or_invalid in (
        "BLENDER_EEVEE_NEXT",
        "EEVEE_NEXT",
        "MY_CYCLES",
        "CYCLE",
        "WORKBENCH",
    ):
        with pytest.raises(MaterialGraphAnalysisError, match="Unsupported Blender 5.2"):
            normalise_render_target(old_or_invalid)


def test_shader_node_type_normalizes_confirmed_blender_52_rna_aliases():
    assert node_type(SimpleNamespace(type="SHADERTORGB")) == "SHADER_TO_RGB"
    assert node_type(SimpleNamespace(type="OUTPUT_MATERIAL")) == "OUTPUT_MATERIAL"
    assert node_type(SimpleNamespace(type=" tex_image ")) == "TEX_IMAGE"
    assert node_type(SimpleNamespace(type="")) == "UNKNOWN"


def test_material_output_prefers_exact_renderer_target_over_global_active_output():
    cycles = SimpleNamespace(
        name="Cycles Output",
        type="OUTPUT_MATERIAL",
        target="CYCLES",
        is_active_output=True,
    )
    eevee = SimpleNamespace(
        name="Eevee Output",
        type="OUTPUT_MATERIAL",
        target="EEVEE",
        is_active_output=False,
    )
    calls: list[str] = []
    tree = SimpleNamespace(
        get_output_node=lambda target: calls.append(target) or cycles,
    )

    assert find_material_output(tree, (cycles, eevee), "EEVEE") is eevee
    assert calls == []


def test_material_output_falls_back_to_generic_all_for_renderer_target():
    generic = SimpleNamespace(
        name="Generic Output",
        type="OUTPUT_MATERIAL",
        target="ALL",
        is_active_output=True,
    )
    cycles = SimpleNamespace(
        name="Cycles Output",
        type="OUTPUT_MATERIAL",
        target="CYCLES",
        is_active_output=False,
    )

    assert find_material_output(
        SimpleNamespace(),
        (cycles, generic),
        "EEVEE",
    ) is generic


def test_shader_graph_without_material_output_is_rejected():
    material = SimpleNamespace(
        name="NoOutput",
        name_full="NoOutput",
        node_tree=SimpleNamespace(nodes=(), links=()),
    )

    with pytest.raises(MaterialGraphAnalysisError, match="no Material Output"):
        analyse_material_graph_detailed(material, render_target="CYCLES")


def test_material_slot_wraps_shader_graph_failure_with_slot_context():
    material = SimpleNamespace(
        name="NoOutput",
        name_full="NoOutput",
        node_tree=SimpleNamespace(nodes=(), links=()),
    )

    with pytest.raises(MaterialAnalysisError, match="slot 3") as captured:
        analyse_material_slot(3, material, render_target="CYCLES")

    assert isinstance(captured.value.__cause__, MaterialGraphAnalysisError)


def test_material_slot_rejects_boolean_slot_index():
    with pytest.raises(ValueError, match="non-negative integer"):
        analyse_material_slot(True, None, render_target="CYCLES")


def test_material_analysis_has_no_root_node_error_fallback():
    resolution = _source("blender_adapter/material_graph_resolution.py")
    analysis = _source("blender_adapter/shader_graph_analysis.py")
    rna = _source("blender_adapter/shader_graph_rna.py")

    assert "historical no-output fallback" not in resolution
    assert "graph_nodes if graph_nodes is not None" not in resolution
    assert "Shader graph analysis failed:" not in resolution
    assert "walker.walk_all_nodes()" not in analysis
    assert "semantic analysis used all nodes" not in analysis
    assert 'if "CYCLE" in target' not in rna
    assert 'if "EEVEE" in target' not in rna
    assert "get_output_node(target=target)" not in rna
