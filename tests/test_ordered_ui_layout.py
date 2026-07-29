"""Exact UI order and ownership contracts for the Rewrite main panel."""

from __future__ import annotations

from pathlib import Path

from Blender_to_Spine2D_Mesh_Exporter import rig_ui, ui, ui_layout
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter import generated_material_ui


EXPECTED_TITLES = (
    "Export",
    "Rig",
    "Rewrite Generated Materials",
    "Cut",
    "Bake",
    "Analysis",
)


def _source(module) -> str:
    return Path(module.__file__).read_text(encoding="utf-8")


def test_ordered_main_panel_uses_exact_requested_foldout_order():
    source = _source(ui_layout)

    offsets = tuple(source.index(f'title="{title}"') for title in EXPECTED_TITLES)
    assert offsets == tuple(sorted(offsets))
    assert ui_layout.OBJECT_PT_Spine2DOrderedMeshPanel.bl_idname == (
        ui.OBJECT_PT_Spine2DMeshPanel.bl_idname
    )


def test_all_secondary_sections_use_the_same_main_foldout_owner():
    ordered = _source(ui_layout)
    rig = _source(rig_ui)
    materials = _source(generated_material_ui)

    assert ordered.count("self._draw_foldout(") == len(EXPECTED_TITLES)
    assert "class OBJECT_PT_Spine2DRigPanel" not in rig
    assert "class OBJECT_PT_Spine2DGeneratedMaterials" not in materials
    assert "draw_rig_settings" in ordered
    assert "draw_generated_material_settings" in ordered


def test_export_controls_are_not_duplicated_outside_rig_foldout():
    ordered = _source(ui_layout)
    export_start = ordered.index("def _draw_export_settings(")
    export_source = ordered[export_start:ordered.index("def _draw_export_action(")]

    assert "spine2d_control_icons" not in export_source
    assert "spine2d_export_preview_animation" not in export_source
    assert "spine2d_texture_export_mode" not in export_source
    assert "spine2d_connect_settings" not in export_source
    assert "spine2d_control_icons" in _source(rig_ui)
    assert "spine2d_export_preview_animation" in _source(rig_ui)
    assert "spine2d_texture_export_mode" in _source(rig_ui)
    assert "spine2d_connect_settings" not in _source(rig_ui)


def test_ordered_layout_replaces_and_restores_the_original_panel_transactionally():
    source = _source(ui_layout)

    assert "bpy.utils.unregister_class(ui.OBJECT_PT_Spine2DMeshPanel)" in source
    assert "bpy.utils.register_class(OBJECT_PT_Spine2DOrderedMeshPanel)" in source
    assert "def _restore_original_panel()" in source
    assert "bpy.utils.register_class(ui.OBJECT_PT_Spine2DMeshPanel)" in source
    assert "ordered UI RNA registration rollback" in source


def test_analysis_is_the_final_foldout_and_export_action_is_in_export():
    source = _source(ui_layout)
    analysis_title = source.index('title="Analysis"')
    single_export = source.index('"object.spine2d_single_export"')
    multi_export = source.index('"object.spine2d_multi_export"')

    assert single_export < analysis_title
    assert multi_export < analysis_title
    assert 'property_name="spine2d_show_analysis"' in source
    assert 'default=False' in source[source.index('name="Show Analysis"'):]
    assert 'layout.separator()\n            self._draw_export_action(layout, context)' in source
    assert 'row.alert = True' in source
