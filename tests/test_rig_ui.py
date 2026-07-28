"""Focused contracts for the dedicated selectable Rig UI category."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from Blender_to_Spine2D_Mesh_Exporter import rig_ui
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter import scene_properties
from Blender_to_Spine2D_Mesh_Exporter.domain.spine import A1RigProfile


def test_rig_reset_restores_three_axis_without_touching_other_settings():
    scene = SimpleNamespace(
        spine2d_rig_profile=A1RigProfile.TWO_AXIS_ROTATION_SCALE.value,
        spine2d_texture_size=2048,
        spine2d_seam_maker_mode="CUSTOM",
    )
    context = SimpleNamespace(scene=scene)
    operator = rig_ui.SPINE2D_OT_ResetRigProfile()
    operator.report = MagicMock()

    result = operator.execute(context)

    assert result == {"FINISHED"}
    assert scene.spine2d_rig_profile == A1RigProfile.THREE_AXIS_ROTATION.value
    assert scene.spine2d_texture_size == 2048
    assert scene.spine2d_seam_maker_mode == "CUSTOM"
    operator.report.assert_called_once_with(
        {"INFO"},
        "Rig profile reset to 3-Axis Rotation",
    )


def test_rig_panel_is_a_separate_child_category_with_profile_specific_copy():
    source = Path(rig_ui.__file__).read_text(encoding="utf-8")

    assert rig_ui.OBJECT_PT_Spine2DRigPanel.bl_parent_id == "OBJECT_PT_spine2d_mesh"
    assert rig_ui.OBJECT_PT_Spine2DRigPanel.bl_label == "Rig"
    assert 'layout.prop(scene, "spine2d_rig_profile"' not in source
    assert 'header.prop(scene, "spine2d_rig_profile"' in source
    assert "Controls: Rotation X / Y / Z" in source
    assert "Controls: Rotation X / Y + Scale" in source
    assert "No Rotation Z control is generated" in source
    assert "spine2d_control_icons" in source
    assert "spine2d_export_preview_animation" in source


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
