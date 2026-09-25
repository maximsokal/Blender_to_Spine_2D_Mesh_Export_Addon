"""Regression coverage for live exact-version UI refresh across Blender windows."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

from Blender_to_Spine2D_Mesh_Exporter import addon_preferences
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter import a1_export_readiness


def _window(*area_types: str):
    areas = tuple(
        SimpleNamespace(type=area_type, tag_redraw=Mock(name=f"redraw_{index}"))
        for index, area_type in enumerate(area_types)
    )
    return SimpleNamespace(screen=SimpleNamespace(areas=areas)), areas


def test_tag_all_view3d_areas_refreshes_every_blender_window() -> None:
    main_window, main_areas = _window("VIEW_3D", "OUTLINER", "VIEW_3D")
    preferences_window, preferences_areas = _window("PREFERENCES", "VIEW_3D")
    context = SimpleNamespace(
        window_manager=SimpleNamespace(
            windows=(main_window, preferences_window),
        )
    )

    redraw_count = addon_preferences._tag_all_view3d_areas_for_redraw(context)

    assert redraw_count == 3
    assert main_areas[0].tag_redraw.call_count == 1
    assert main_areas[1].tag_redraw.call_count == 0
    assert main_areas[2].tag_redraw.call_count == 1
    assert preferences_areas[0].tag_redraw.call_count == 0
    assert preferences_areas[1].tag_redraw.call_count == 1


def test_deferred_redraw_is_coalesced_until_timer_runs(monkeypatch) -> None:
    registrations: list[tuple[object, float]] = []

    def register(callback, *, first_interval: float):
        registrations.append((callback, first_interval))

    monkeypatch.setattr(
        addon_preferences.bpy.app,
        "timers",
        SimpleNamespace(register=register),
        raising=False,
    )
    monkeypatch.setattr(addon_preferences, "_view3d_redraw_scheduled", False)
    redraw = Mock(return_value=2)
    monkeypatch.setattr(addon_preferences, "_tag_all_view3d_areas_for_redraw", redraw)

    addon_preferences._schedule_view3d_redraw()
    addon_preferences._schedule_view3d_redraw()

    assert len(registrations) == 1
    callback, first_interval = registrations[0]
    assert first_interval == 0.0
    assert addon_preferences._view3d_redraw_scheduled is True

    assert callback() is None
    redraw.assert_called_once_with(None)
    assert addon_preferences._view3d_redraw_scheduled is False

    addon_preferences._schedule_view3d_redraw()
    assert len(registrations) == 2


def test_cancel_deferred_redraw_unregisters_owned_timer(monkeypatch) -> None:
    unregistered: list[object] = []

    timers = SimpleNamespace(
        is_registered=lambda callback: callback is addon_preferences._deferred_view3d_redraw,
        unregister=lambda callback: unregistered.append(callback),
    )
    monkeypatch.setattr(addon_preferences.bpy.app, "timers", timers, raising=False)
    monkeypatch.setattr(addon_preferences, "_view3d_redraw_scheduled", True)

    addon_preferences._cancel_deferred_view3d_redraw()

    assert unregistered == [addon_preferences._deferred_view3d_redraw]
    assert addon_preferences._view3d_redraw_scheduled is False


def test_cancel_deferred_redraw_is_safe_after_timer_already_completed(monkeypatch) -> None:
    unregister = Mock()
    timers = SimpleNamespace(
        is_registered=lambda _callback: False,
        unregister=unregister,
    )
    monkeypatch.setattr(addon_preferences.bpy.app, "timers", timers, raising=False)
    monkeypatch.setattr(addon_preferences, "_view3d_redraw_scheduled", False)

    addon_preferences._cancel_deferred_view3d_redraw()

    unregister.assert_not_called()
    assert addon_preferences._view3d_redraw_scheduled is False


def test_preference_update_redraws_even_if_readiness_invalidation_fails(
    monkeypatch,
) -> None:
    def fail_readiness(_scene) -> None:
        raise RuntimeError("synthetic readiness failure")

    monkeypatch.setattr(
        a1_export_readiness,
        "clear_a1_export_readiness",
        fail_readiness,
    )
    monkeypatch.setattr(
        addon_preferences.bpy.data,
        "scenes",
        (object(),),
        raising=False,
    )

    immediate_redraw = Mock(return_value=1)
    deferred_redraw = Mock()
    monkeypatch.setattr(
        addon_preferences,
        "_tag_all_view3d_areas_for_redraw",
        immediate_redraw,
    )
    monkeypatch.setattr(
        addon_preferences,
        "_schedule_view3d_redraw",
        deferred_redraw,
    )

    context = SimpleNamespace(window_manager=SimpleNamespace(windows=()))
    addon_preferences._update_spine_project_version(object(), context)

    immediate_redraw.assert_called_once_with(context)
    deferred_redraw.assert_called_once_with()
