"""UI order and ownership contracts for the canonical Rewrite exporter panel."""

from __future__ import annotations

from pathlib import Path

from Blender_to_Spine2D_Mesh_Exporter import rig_ui, ui, ui_layout
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter import generated_material_ui


MAIN_FOLDOUT_TITLES = (
    "Paths and Spine 2D version",
    "Cut",
    "Bake",
)


def _source(module) -> str:
    return Path(module.__file__).read_text(encoding="utf-8")


def test_canonical_main_panel_keeps_primary_foldouts_in_order():
    source = _source(ui)
    draw_start = source.index("    def draw(self, context: bpy.types.Context) -> None:")
    draw_source = source[draw_start:source.index("\n\nclass _Spine2DExportOperatorMixin")]

    offsets = tuple(draw_source.index(f'title="{title}"') for title in MAIN_FOLDOUT_TITLES)
    assert offsets == tuple(sorted(offsets))
    assert ui.OBJECT_PT_Spine2DMeshPanel.bl_idname == "OBJECT_PT_spine2d_mesh"


def test_canonical_main_panel_uses_public_title_without_changing_technical_id():
    panel = ui.OBJECT_PT_Spine2DMeshPanel

    assert panel.bl_label == "Spine Mesh Exporter"
    assert panel.bl_category == "Spine Mesh Exporter"
    assert panel.bl_idname == "OBJECT_PT_spine2d_mesh"


def test_secondary_sections_are_ordinary_child_panels_of_canonical_main_panel():
    child_panels = (
        ui_layout.OBJECT_PT_Spine2DRigPanel,
        ui_layout.OBJECT_PT_Spine2DGeneratedMaterialsPanel,
        ui_layout.OBJECT_PT_Spine2DDepthParallaxPanel,
        ui_layout.OBJECT_PT_Spine2DAnalysisPanel,
    )

    assert all(
        panel.bl_parent_id == ui.OBJECT_PT_Spine2DMeshPanel.bl_idname
        for panel in child_panels
    )
    assert tuple(panel.bl_order for panel in child_panels) == tuple(
        sorted(panel.bl_order for panel in child_panels)
    )


def test_child_panels_delegate_to_canonical_section_drawers():
    layout_source = _source(ui_layout)
    rig_source = _source(rig_ui)
    material_source = _source(generated_material_ui)

    assert "rig_ui.draw_rig_settings(self.layout, context)" in layout_source
    assert (
        "generated_material_ui.draw_generated_material_settings(self.layout, context)"
        in layout_source
    )
    assert "_draw_modifier_analysis_warning(layout, context)" in layout_source
    assert "def draw_rig_settings(" in rig_source
    assert "def draw_generated_material_settings(" in material_source


def test_main_panel_and_rig_child_do_not_duplicate_core_export_controls():
    main_source = _source(ui)
    rig_source = _source(rig_ui)

    assert "spine2d_texture_export_mode" in main_source
    assert "spine2d_control_icons" in main_source
    assert "spine2d_export_preview_animation" not in main_source[
        main_source.index("    def _draw_export_settings(") : main_source.index(
            "    @staticmethod\n    def _draw_cut_settings("
        )
    ]

    assert "spine2d_export_preview_animation" in rig_source
    assert "spine2d_shared_selection_pivot" in rig_source
    assert "spine2d_connect_settings" not in rig_source


def test_ui_layout_never_replaces_or_restores_the_main_panel():
    source = _source(ui_layout)

    assert "OBJECT_PT_Spine2DOrderedMeshPanel" not in source
    assert "_ORIGINAL_PANEL_REMOVED" not in source
    assert "_restore_original_panel" not in source
    assert "bpy.utils.unregister_class(ui.OBJECT_PT_Spine2DMeshPanel)" not in source
    assert "bpy.utils.register_class(ui.OBJECT_PT_Spine2DMeshPanel)" not in source
    assert "for cls in CLASSES:" in source
    assert "for cls in reversed(CLASSES):" in source


def test_readiness_is_drawn_after_settings_and_before_export_action():
    source = _source(ui)
    draw_start = source.index("    def draw(self, context: bpy.types.Context) -> None:")
    draw_source = source[draw_start:source.index("\n\nclass _Spine2DExportOperatorMixin")]

    bake = draw_source.index('title="Bake"')
    readiness = draw_source.index("export_allowed = self._draw_readiness(layout, context)")
    single_export = draw_source.index('"object.spine2d_single_export"')
    multi_export = draw_source.index('"object.spine2d_multi_export"')

    assert bake < readiness < single_export
    assert bake < readiness < multi_export
