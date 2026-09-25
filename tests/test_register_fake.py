from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import Blender_to_Spine2D_Mesh_Exporter as addon


def test_register_unregister_no_errors(monkeypatch):
    preferences = object()
    monkeypatch.setattr(addon.bpy.app, "version", (5, 2, 0))
    monkeypatch.setitem(
        addon.bpy.context.preferences.addons,
        addon.__name__,
        SimpleNamespace(preferences=preferences),
    )

    owners = (
        addon.addon_preferences,
        addon.scene_settings_migration,
        addon.ui,
        addon.rig_ui,
        addon.a1_readiness_invalidation,
        addon.auto_readiness,
        addon.generated_material_ui,
        addon.ui_layout,
        addon.single_object_operator,
    )
    register_callbacks = []
    unregister_callbacks = []
    for owner in owners:
        register_callback = MagicMock()
        unregister_callback = MagicMock()
        monkeypatch.setattr(owner, "register", register_callback)
        monkeypatch.setattr(owner, "unregister", unregister_callback)
        register_callbacks.append(register_callback)
        unregister_callbacks.append(unregister_callback)

    register_config = MagicMock()
    unregister_config = MagicMock()
    monkeypatch.setattr(addon, "_register_config_rna", register_config)
    monkeypatch.setattr(addon, "_unregister_config_rna", unregister_config)

    with patch.object(addon.config, "_setup_default_logging"), patch.object(
        addon.config, "setup_logging"
    ), patch.object(
        addon.config, "_addon_preferences", return_value=preferences
    ), patch.object(
        addon, "initialize_logging_preferences"
    ) as initialize:
        addon.register()
        for callback in register_callbacks:
            callback.assert_called_once_with()
        register_config.assert_called_once_with()
        initialize.assert_called_once_with(preferences)

        addon.unregister()
        for callback in unregister_callbacks:
            callback.assert_called_once_with()
        unregister_config.assert_called_once_with()
