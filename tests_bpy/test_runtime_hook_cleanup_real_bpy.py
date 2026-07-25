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


def _register_steps():
    completed = []
    try:
        for step in extension.REGISTRATION_STEPS:
            step[1]()
            completed.append(step)
        return tuple(completed)
    except Exception:
        for step in reversed(completed):
            step[2]()
        raise


def _unregister_steps(completed):
    failures = []
    for label, _register, unregister in reversed(completed):
        try:
            unregister()
        except Exception as exc:
            failures.append(f"{label}: {type(exc).__name__}: {exc}")
    assert failures == []


def test_ten_registration_cycles_restore_all_handlers_and_addon_keymaps(clean_blender_data):
    handlers_before = _handler_signature()
    keymaps_before = _keymap_signature()

    for _cycle in range(10):
        completed = _register_steps()
        try:
            handler = extension.ui.a1_readiness_depsgraph_update_post
            assert tuple(bpy.app.handlers.depsgraph_update_post).count(handler) == 1
            assert _keymap_signature() == keymaps_before
        finally:
            _unregister_steps(completed)
        assert _handler_signature() == handlers_before
        assert _keymap_signature() == keymaps_before
