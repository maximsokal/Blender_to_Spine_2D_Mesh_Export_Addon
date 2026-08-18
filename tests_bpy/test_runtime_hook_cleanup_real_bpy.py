"""Registration must leave every Blender handler and addon keymap at baseline."""

from __future__ import annotations

import bpy

import Blender_to_Spine2D_Mesh_Exporter as extension


def _handler_signature():
    return tuple(
        (name, tuple(getattr(bpy.app.handlers, name)))
        for name in sorted(dir(bpy.app.handlers))
        if not name.startswith("_") and isinstance(getattr(bpy.app.handlers, name), list)
    )


def _keymap_signature():
    keyconfig = bpy.context.window_manager.keyconfigs.addon
    if keyconfig is None:
        return ()
    result = []
    for keymap in keyconfig.keymaps:
        items = tuple(
            sorted(
                (
                    item.idname,
                    item.type,
                    item.value,
                    bool(item.ctrl),
                    bool(item.shift),
                    bool(item.alt),
                    bool(item.oskey),
                    str(item.key_modifier),
                )
                for item in keymap.keymap_items
            )
        )
        result.append((keymap.name, keymap.space_type, keymap.region_type, items))
    return tuple(sorted(result))


def test_ten_registration_cycles_restore_all_handlers_keymaps_and_owned_timer(
    clean_blender_data,
):
    handlers_before = _handler_signature()
    keymaps_before = _keymap_signature()

    for _cycle in range(10):
        extension.register()
        try:
            handler = extension.ui.a1_readiness_depsgraph_update_post
            assert tuple(bpy.app.handlers.depsgraph_update_post).count(handler) == 1
            assert _keymap_signature() == keymaps_before
        finally:
            extension.unregister()

        assert _handler_signature() == handlers_before
        assert _keymap_signature() == keymaps_before

        is_registered = getattr(bpy.app.timers, "is_registered", None)
        if callable(is_registered):
            assert not is_registered(extension.addon_preferences._deferred_view3d_redraw)
            assert not is_registered(extension.auto_readiness._automatic_timer)
