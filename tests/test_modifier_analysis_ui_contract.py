"""Static UI contract for the Normal-mode modifier mismatch alert."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
UI_LAYOUT = ROOT / "Blender_to_Spine2D_Mesh_Exporter" / "ui_layout.py"


def test_analysis_foldout_draws_normal_uv_modifier_warning() -> None:
    source = UI_LAYOUT.read_text(encoding="utf-8")

    assert "collect_normal_uv_ignored_modifiers" in source
    assert "group_ignored_modifiers_by_object" in source
    assert "def _draw_modifier_analysis_warning(" in source
    assert "self._draw_modifier_analysis_warning(layout, context)" in source
    assert 'text="Normal / UV Segments ignores active modifiers"' in source
    assert 'text="Viewport and Spine geometry can look different."' in source
    assert 'text="Apply or convert modifiers before export."' in source
    assert 'icon="MODIFIER"' in source
    assert 'property_name="spine2d_show_analysis"' in source


def test_modifier_warning_is_advisory_and_does_not_disable_export() -> None:
    source = UI_LAYOUT.read_text(encoding="utf-8")

    warning_start = source.index("def _draw_modifier_analysis_warning(")
    readiness_start = source.index("def _draw_readiness(", warning_start)
    warning_source = source[warning_start:readiness_start]

    assert "box.alert = True" in warning_source
    assert ".enabled = False" not in warning_source
    assert "raise" not in warning_source
