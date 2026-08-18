"""Static UI contract for the Normal-mode modifier mismatch alert."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
UI_LAYOUT = ROOT / "Blender_to_Spine2D_Mesh_Exporter" / "ui_layout.py"


def _function_source(source: str, name: str) -> str:
    tree = ast.parse(source, filename=str(UI_LAYOUT))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    segment = ast.get_source_segment(source, function)
    assert segment is not None
    return segment


def test_analysis_child_panel_draws_normal_uv_modifier_warning() -> None:
    source = UI_LAYOUT.read_text(encoding="utf-8")

    assert "collect_normal_uv_ignored_modifiers" in source
    assert "group_ignored_modifiers_by_object" in source
    assert "def _draw_modifier_analysis_warning(" in source
    assert "_draw_modifier_analysis_warning(layout, context)" in source
    assert 'text="Normal / UV Segments ignores active modifiers"' in source
    assert 'text="Viewport and Spine geometry can look different."' in source
    assert 'text="Apply or convert modifiers before export."' in source
    assert 'icon="MODIFIER"' in source
    assert "class OBJECT_PT_Spine2DAnalysisPanel" in source
    assert 'bl_parent_id = _PARENT_PANEL_ID' in source
    assert 'bl_label = "Analysis"' in source


def test_modifier_warning_is_advisory_and_does_not_disable_export() -> None:
    source = UI_LAYOUT.read_text(encoding="utf-8")
    warning_source = _function_source(source, "_draw_modifier_analysis_warning")

    assert "box.alert = True" in warning_source
    assert ".enabled = False" not in warning_source
    assert "raise" not in warning_source
    assert "operator(" not in warning_source
