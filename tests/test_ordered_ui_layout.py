"""UI order and ownership contracts for the canonical Rewrite exporter panel."""

from __future__ import annotations

from pathlib import Path

from Blender_to_Spine2D_Mesh_Exporter import rig_ui, ui, ui_layout
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter import generated_material_ui


MAIN_FOLDOUT_TITLES = (
    "Paths and Spine 2D version",
    "Rig",
    "Generated Materials",
    "Cut",
    "Bake",
    "Analysis",
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

    assert panel.bl_label == "Mesh Exporter"
    assert panel.bl_category == "Mesh Exporter"
    assert panel.bl_idname == "OBJECT_PT_spine2d_mesh"


def test_secondary_sections_are_drawn_by_the_canonical_main_panel_itself():
    """Rig / Generated Materials / Analysis are foldouts inside draw(), not child panels.

    Blender always renders a bl_parent_id child panel after everything the parent's own
    draw() produces, and never gives it the same box styling as a hand-drawn foldout.
    Drawing these sections directly inside the canonical panel's own draw() is what lets
    them share the exact visual style and ordering of Paths, Cut, and Bake.
    """

    source = _source(ui)

    assert not hasattr(ui_layout, "OBJECT_PT_Spine2DRigPanel")
    assert not hasattr(ui_layout, "OBJECT_PT_Spine2DGeneratedMaterialsPanel")
    assert not hasattr(ui_layout, "OBJECT_PT_Spine2DAnalysisPanel")
    assert not hasattr(ui_layout, "OBJECT_PT_Spine2DDepthParallaxPanel")
    assert ui_layout.CLASSES == ()

    for property_name in (
        "spine2d_show_rig_settings",
        "spine2d_show_generated_material_settings",
        "spine2d_show_analysis",
    ):
        assert f'property_name="{property_name}"' in source


def test_child_panels_delegate_to_canonical_section_drawers():
    layout_source = _source(ui_layout)
    ui_source = _source(ui)
    rig_source = _source(rig_ui)
    material_source = _source(generated_material_ui)

    assert "rig_ui.draw_rig_settings(column, context)" in ui_source
    assert "generated_material_ui.draw_generated_material_settings(" in ui_source
    assert "ui_layout._draw_modifier_analysis_warning(" in ui_source
    assert "def draw_rig_settings(" in rig_source
    assert "def draw_generated_material_settings(" in material_source
    assert "def _draw_modifier_analysis_warning(" in layout_source


def test_paths_foldout_does_not_duplicate_export_mode_and_rig_controls():
    """Export mode, projection, and control icons are rig decisions.

    They live in the Rig foldout, not in Paths and Spine 2D version, which now only
    owns the output schema/paths.
    """

    main_source = _source(ui)
    rig_source = _source(rig_ui)

    export_settings_source = main_source[
        main_source.index("    def _draw_export_settings(") : main_source.index(
            "    @staticmethod\n    def _draw_cut_settings("
        )
    ]
    assert "spine2d_texture_export_mode" not in export_settings_source
    assert "spine2d_control_icons" not in export_settings_source
    assert "spine2d_export_preview_animation" not in export_settings_source
    assert "spine2d_connect_settings" not in export_settings_source

    assert "spine2d_texture_export_mode" in rig_source
    assert "spine2d_control_icons" in rig_source
    assert "spine2d_export_preview_animation" in rig_source
    assert "spine2d_shared_selection_pivot" in rig_source


def test_rig_foldout_does_not_expose_ignored_connect_objects_control():
    """Export Selected Objects ignores Connect, so the UI must not offer it."""

    rig_source = _source(rig_ui)
    main_source = _source(ui)

    assert "spine2d_connect_settings" not in rig_source
    assert "Connect objects" not in rig_source
    assert "spine2d_connect_settings, \"enabled\"" not in main_source


def test_ui_layout_never_replaces_or_restores_the_main_panel():
    source = _source(ui_layout)

    assert "OBJECT_PT_Spine2DOrderedMeshPanel" not in source
    assert "_ORIGINAL_PANEL_REMOVED" not in source
    assert "_restore_original_panel" not in source
    assert "bpy.utils.unregister_class(ui.OBJECT_PT_Spine2DMeshPanel)" not in source
    assert "bpy.utils.register_class(ui.OBJECT_PT_Spine2DMeshPanel)" not in source
    assert "for cls in CLASSES:" in source
    assert "for cls in reversed(CLASSES):" in source


def test_analysis_foldout_is_drawn_after_bake_and_before_export_action():
    source = _source(ui)
    draw_start = source.index("    def draw(self, context: bpy.types.Context) -> None:")
    draw_source = source[draw_start:source.index("\n\nclass _Spine2DExportOperatorMixin")]

    bake = draw_source.index('title="Bake"')
    analysis = draw_source.index('title="Analysis"')
    single_export = draw_source.index('"object.spine2d_single_export"')
    multi_export = draw_source.index('"object.spine2d_multi_export"')

    assert bake < analysis < single_export
    assert bake < analysis < multi_export


def test_analysis_foldout_holds_every_analysis_only_control():
    """The Analysis foldout must be the single home for readiness diagnostics.

    Nothing analysis-related may be left drawn unconditionally outside it, or the
    foldout collapses to an empty shell while the real content renders below Export.
    """

    source = _source(ui)

    analysis_method_start = source.index("    def _draw_analysis_content(")
    analysis_method_end = source.index(
        "\n\n    def draw(self, context: bpy.types.Context) -> None:"
    )
    analysis_method_source = source[analysis_method_start:analysis_method_end]

    assert "_draw_modifier_analysis_warning(column, context)" in analysis_method_source
    assert "self._draw_readiness(column, context)" in analysis_method_source

    draw_start = source.index("    def draw(self, context: bpy.types.Context) -> None:")
    draw_source = source[draw_start:source.index("\n\nclass _Spine2DExportOperatorMixin")]
    assert "self._draw_readiness(" not in draw_source
    assert "_draw_modifier_analysis_warning(" not in draw_source
    assert 'draw_content=lambda column: self._draw_analysis_content(' in draw_source
