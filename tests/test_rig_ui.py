"""Focused contracts for the selectable Rig foldout content."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from Blender_to_Spine2D_Mesh_Exporter import rig_ui
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter import scene_properties
from Blender_to_Spine2D_Mesh_Exporter.domain.spine import A1RigProfile


def test_rig_reset_restores_two_axis_without_touching_other_settings():
    scene = SimpleNamespace(
        spine2d_rig_profile=A1RigProfile.THREE_AXIS_ROTATION.value,
        spine2d_texture_size=2048,
        spine2d_seam_maker_mode="CUSTOM",
    )
    context = SimpleNamespace(scene=scene)
    operator = rig_ui.SPINE2D_OT_ResetRigProfile()
    operator.report = MagicMock()

    result = operator.execute(context)

    assert result == {"FINISHED"}
    assert scene.spine2d_rig_profile == A1RigProfile.TWO_AXIS_ROTATION_SCALE.value
    assert scene.spine2d_texture_size == 2048
    assert scene.spine2d_seam_maker_mode == "CUSTOM"
    operator.report.assert_called_once_with(
        {"INFO"},
        "Rig profile reset to 2-Axis Rotation + Scale",
    )


def test_rig_content_is_drawn_by_the_ordered_main_foldout_not_a_child_panel():
    source = Path(rig_ui.__file__).read_text(encoding="utf-8")

    assert "class OBJECT_PT_Spine2DRigPanel" not in source
    assert "def draw_rig_settings(" in source
    assert 'row.label(text="Rig profile")' in source
    assert 'row.label(text="2-Axis Rotation + Scale"' in source
    assert "Controls: Rotation X / Y + Scale" in source
    assert "Main bone matches Blender Object Origin" in source
    assert "Depth layers use the object's local Z=0 pivot" in source
    assert "spine2d_control_icons" in source
    assert "spine2d_export_preview_animation" in source
    assert "Connect objects:" not in source
    assert rig_ui.CLASSES == (rig_ui.SPINE2D_OT_ResetRigProfile,)


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
