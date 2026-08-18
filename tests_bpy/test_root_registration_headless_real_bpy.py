"""Real Blender 5.2 regressions for direct repository package registration."""

from __future__ import annotations

import bpy

import Blender_to_Spine2D_Mesh_Exporter as extension


_PACKAGE_SUFFIX = "Blender_to_Spine2D_Mesh_Exporter"


def _matching_addon_preference_keys() -> tuple[str, ...]:
    """Return enabled-addon keys that could own this package's preferences."""

    return tuple(
        sorted(
            str(key)
            for key in bpy.context.preferences.addons.keys()
            if str(key) == extension.__name__
            or str(key).endswith(_PACKAGE_SUFFIX)
        )
    )


def _assert_manual_readiness_runtime_registered() -> None:
    """Require the manual readiness bridge without background analysis hooks."""

    assert extension.auto_readiness._REGISTERED is True
    assert not bpy.app.timers.is_registered(extension.auto_readiness._automatic_timer)
    assert extension.auto_readiness.a1_auto_readiness_depsgraph_update_post not in (
        bpy.app.handlers.depsgraph_update_post
    )
    assert extension.auto_readiness.a1_auto_readiness_load_pre not in (
        bpy.app.handlers.load_pre
    )
    assert extension.auto_readiness.a1_auto_readiness_load_post not in (
        bpy.app.handlers.load_post
    )


def _assert_manual_readiness_runtime_unregistered() -> None:
    """Require complete teardown of the manual readiness bridge and old hooks."""

    assert extension.auto_readiness._REGISTERED is False
    assert not bpy.app.timers.is_registered(extension.auto_readiness._automatic_timer)
    assert extension.auto_readiness.a1_auto_readiness_depsgraph_update_post not in (
        bpy.app.handlers.depsgraph_update_post
    )
    assert extension.auto_readiness.a1_auto_readiness_load_pre not in (
        bpy.app.handlers.load_pre
    )
    assert extension.auto_readiness.a1_auto_readiness_load_post not in (
        bpy.app.handlers.load_post
    )


def _assert_root_runtime_registered() -> None:
    assert hasattr(bpy.types.Scene, "spine2d_texture_export_mode")
    assert bpy.context.scene.spine2d_texture_export_mode == "NORMAL_UV_SEGMENTS"
    assert bpy.types.Operator.bl_rna_get_subclass_py(
        "OBJECT_OT_spine2d_single_export"
    ) is not None
    assert tuple(bpy.app.handlers.depsgraph_update_post).count(
        extension.a1_readiness_invalidation.a1_readiness_depsgraph_update_post
    ) == 1
    _assert_manual_readiness_runtime_registered()


def _assert_root_runtime_unregistered() -> None:
    assert not hasattr(bpy.types.Scene, "spine2d_texture_export_mode")
    assert bpy.types.Operator.bl_rna_get_subclass_py(
        "OBJECT_OT_spine2d_single_export"
    ) is None
    assert extension.a1_readiness_invalidation.a1_readiness_depsgraph_update_post not in (
        bpy.app.handlers.depsgraph_update_post
    )
    _assert_manual_readiness_runtime_unregistered()

    is_registered = getattr(bpy.app.timers, "is_registered", None)
    if callable(is_registered):
        assert not is_registered(extension.addon_preferences._deferred_view3d_redraw)


def test_direct_root_registration_without_enabled_addon_entry(clean_blender_data):
    """Direct package import must not require Preferences > Add-ons ownership."""

    assert _matching_addon_preference_keys() == ()
    _assert_root_runtime_unregistered()

    extension.register()
    try:
        _assert_root_runtime_registered()
        # Registering classes directly must not invent an enabled-addon preference key.
        assert _matching_addon_preference_keys() == ()
    finally:
        extension.unregister()

    _assert_root_runtime_unregistered()


def test_logging_preference_initialization_failure_is_nonfatal_after_registration(
    clean_blender_data,
    monkeypatch,
):
    """Diagnostics preference errors must not corrupt an otherwise valid enable."""

    _assert_root_runtime_unregistered()
    fake_preferences = object()
    monkeypatch.setattr(
        extension.config,
        "_addon_preferences",
        lambda: fake_preferences,
    )

    calls: list[object] = []

    def fail_initialization(prefs):
        calls.append(prefs)
        assert prefs is fake_preferences
        raise RuntimeError("forced preference initialization failure")

    monkeypatch.setattr(
        extension,
        "initialize_logging_preferences",
        fail_initialization,
    )

    extension.register()
    try:
        _assert_root_runtime_registered()
        assert calls == [fake_preferences]
    finally:
        extension.unregister()

    _assert_root_runtime_unregistered()
