"""Regressions for strict Blender 5.2 material-analysis ownership."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.material_analysis_error import (
    MaterialAnalysisError,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.material_analysis_rna import (
    require_render_target,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.material_object_analysis import (
    analyse_object_materials,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.material_slot_analysis import (
    analyse_material_slot,
)


ROOT = Path(__file__).resolve().parents[1]
ADAPTER = (
    ROOT
    / "Blender_to_Spine2D_Mesh_Exporter"
    / "blender_adapter"
)


def _source(name: str) -> str:
    return (ADAPTER / name).read_text(encoding="utf-8")


def test_material_render_target_is_explicit_and_exact():
    assert require_render_target("CYCLES") == "CYCLES"
    assert require_render_target("EEVEE") == "EEVEE"
    assert require_render_target("BLENDER_EEVEE") == "EEVEE"

    for invalid in ("", "ALL_RENDERERS", "BLENDER_EEVEE_NEXT", "MY_CYCLES"):
        with pytest.raises(MaterialAnalysisError):
            require_render_target(invalid)


def test_material_slot_requires_target_even_for_empty_slot():
    with pytest.raises(TypeError):
        analyse_material_slot(0, None)

    with pytest.raises(MaterialAnalysisError, match="render_target"):
        analyse_material_slot(0, None, render_target="")

    result = analyse_material_slot(0, None, render_target="CYCLES")
    assert result.material_name is None


def test_object_material_analysis_requires_target_before_slot_iteration():
    obj = SimpleNamespace(
        type="MESH",
        name="Hero",
        name_full="Hero",
        material_slots=(),
    )

    with pytest.raises(TypeError):
        analyse_object_materials(obj)

    with pytest.raises(MaterialAnalysisError, match="render_target"):
        analyse_object_materials(obj, render_target="")

    result = analyse_object_materials(obj, render_target="CYCLES")
    assert result.source_object_id == "Hero"
    assert result.slots == ()


def test_material_analysis_contains_no_scene_or_all_fallback():
    rna = _source("material_analysis_rna.py")
    slot = _source("material_slot_analysis.py")
    object_analysis = _source("material_object_analysis.py")

    assert "def resolve_render_target(" not in rna
    assert "import bpy" not in rna
    assert 'return "ALL"' not in rna
    assert "Unable to resolve active Blender render target" not in rna
    assert "render_target: str | None" not in slot
    assert "render_target: str | None" not in object_analysis
    assert "target = require_render_target(render_target)" in slot
    assert "target = require_render_target(render_target)" in object_analysis


def test_material_classifier_contains_no_legacy_tuple_api():
    classifier = _source("material_node_classification.py")

    assert "def as_legacy_tuple(" not in classifier
    assert "def classify_nodes_legacy(" not in classifier
    assert '"classify_nodes_legacy"' not in classifier


def test_retired_material_and_scene_analyzer_facades_are_deleted():
    assert not (ADAPTER / "material_analyzer.py").exists()
    assert not (ADAPTER / "scene_bake_analyzer.py").exists()


def test_scene_material_preparation_has_exact_target_and_public_owner():
    source = _source("scene_material_preparation.py")

    assert "temporary_prepare_material_pass" in source
    assert "_prepare_proxy_material" not in source
    assert "_restore_mutation" not in source
    assert 'if "CYCLE" in' not in source
    assert 'if "EEVEE" in' not in source
    assert "render_target: str," in source
    assert 'render_target: str = "CYCLES"' not in source
    assert "B2/B3" not in source
