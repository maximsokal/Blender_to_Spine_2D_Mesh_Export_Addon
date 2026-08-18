"""Focused contracts for the standard Rig child-panel content."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from Blender_to_Spine2D_Mesh_Exporter import rig_ui, ui_layout
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter import scene_properties
from Blender_to_Spine2D_Mesh_Exporter.domain.projection import A1ProjectionDirection
from Blender_to_Spine2D_Mesh_Exporter.domain.spine import A1RigProfile


def test_rig_reset_restores_owned_rig_and_projection_defaults():
    scene = SimpleNamespace(
        spine2d_rig_profile=A1RigProfile.THREE_AXIS_ROTATION.value,
        spine2d_projection_direction=A1ProjectionDirection.NEGATIVE_Y.value,
        spine2d_shared_selection_pivot=False,
        spine2d_depth_parallax_horizon_angle=0.25,
        spine2d_export_preview_animation=True,
        spine2d_texture_size=2048,
        spine2d_seam_maker_mode="CUSTOM",
    )
    context = SimpleNamespace(scene=scene)
    operator = rig_ui.SPINE2D_OT_ResetRigProfile()
    operator.report = MagicMock()

    result = operator.execute(context)

    assert result == {"FINISHED"}
    assert scene.spine2d_rig_profile == A1RigProfile.TWO_AXIS_ROTATION_SCALE.value
    assert scene.spine2d_projection_direction == A1ProjectionDirection.POSITIVE_Z.value
    assert scene.spine2d_shared_selection_pivot is True
    assert scene.spine2d_depth_parallax_horizon_angle == 0.0
    assert scene.spine2d_export_preview_animation is False
    assert scene.spine2d_texture_size == 2048
    assert scene.spine2d_seam_maker_mode == "CUSTOM"
    operator.report.assert_called_once_with(
        {"INFO"},
        "Rig and projection settings reset",
    )


def test_rig_content_is_drawn_by_standard_child_panel_without_main_panel_replacement():
    source = Path(rig_ui.__file__).read_text(encoding="utf-8")
    layout_source = Path(ui_layout.__file__).read_text(encoding="utf-8")

    assert "def draw_rig_settings(" in source
    assert 'header.label(text="2-Axis Rotation + Scale"' in source
    assert "Controls: Rotation X / Y + Scale" in source
    assert "spine2d_shared_selection_pivot" in source
    assert "spine2d_export_preview_animation" in source
    assert "SPINE2D_OT_ResetSettingsWithProjection" not in source
    assert rig_ui.CLASSES == (rig_ui.SPINE2D_OT_ResetRigProfile,)

    assert "class OBJECT_PT_Spine2DRigPanel" in layout_source
    assert 'bl_parent_id = _PARENT_PANEL_ID' in layout_source
    assert "rig_ui.draw_rig_settings(self.layout, context)" in layout_source
    assert "bpy.utils.unregister_class(ui.OBJECT_PT_Spine2DMeshPanel)" not in layout_source


def test_rig_scene_property_default_is_two_axis():
    source = Path(scene_properties.__file__).read_text(encoding="utf-8")
    assert "default=A1RigProfile.TWO_AXIS_ROTATION_SCALE.value" in source


def test_rig_profile_update_invalidates_readiness_and_requests_redraw(monkeypatch):
    context = SimpleNamespace(scene=object())
    events: list[tuple[str, object]] = []

    monkeypatch.setattr(
        scene_properties,
        "_invalidate_readiness_for_setting",
        lambda resolved_context, *, reason: events.append((reason, resolved_context)),
    )
    monkeypatch.setattr(
        scene_properties,
        "_update_ui_for_paths",
        lambda owner, resolved_context: events.append(("redraw", resolved_context)),
    )

    scene_properties._update_rig_profile(object(), context)

    assert events == [
        ("rig profile changed", context),
        ("redraw", context),
    ]
