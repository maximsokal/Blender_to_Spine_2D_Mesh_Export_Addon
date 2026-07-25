from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import Blender_to_Spine2D_Mesh_Exporter as addon


def test_register_unregister_no_errors(monkeypatch):
    register_callback = MagicMock()
    unregister_callback = MagicMock()
    preferences = object()
    monkeypatch.setattr(addon.bpy.app, "version", (5, 2, 0))
    monkeypatch.setitem(
        addon.bpy.context.preferences.addons,
        addon.__name__,
        SimpleNamespace(preferences=preferences),
    )

    steps = (("fake owner", register_callback, unregister_callback),)
    with patch.object(addon, "REGISTRATION_STEPS", steps), patch.object(
        addon.config, "_setup_default_logging"
    ), patch.object(addon.config, "setup_logging"), patch.object(
        addon, "initialize_logging_preferences"
    ) as initialize:
        addon.register()
        register_callback.assert_called_once_with()
        initialize.assert_called_once_with(preferences)

        addon.unregister()
        unregister_callback.assert_called_once_with()
