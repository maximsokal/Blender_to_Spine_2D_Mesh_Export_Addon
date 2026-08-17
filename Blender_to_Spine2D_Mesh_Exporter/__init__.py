# pylint: disable=import-error
"""Main entry point for the Blender 5.2+ Rewrite extension."""

from __future__ import annotations

import logging
from typing import Any


logger = logging.getLogger(__package__ or __name__)


try:
    import bpy as _bpy  # type: ignore
except ModuleNotFoundError:
    _bpy = None

if _bpy is not None and all(
    hasattr(_bpy, attribute) for attribute in ("types", "props", "utils", "context")
):
    bpy = _bpy
else:
    bpy = None


if bpy is not None:
    from . import (
        addon_preferences,
        auto_readiness,
        config,
        rig_ui,
        single_object_operator,
        ui,
        ui_layout,
    )
    from .blender_adapter import (
        a1_readiness_invalidation,
        generated_material_ui,
        scene_properties,
        scene_settings_migration,
    )
    from .infrastructure.blender_version import require_supported_blender_runtime

    AddonLoggingSettings = addon_preferences.AddonLoggingSettings
    CLASSES_TO_REGISTER = addon_preferences.CLASSES_TO_REGISTER
    LoggingModuleSettings = addon_preferences.LoggingModuleSettings
    ModelToSpine2DAddonPreferences = addon_preferences.ModelToSpine2DAddonPreferences
    SPINE2D_OT_RefreshLoggingModules = addon_preferences.SPINE2D_OT_RefreshLoggingModules
    initialize_logging_preferences = addon_preferences.initialize_logging_preferences

    # Keep the root owner intentionally boring: each module owns its Blender resources
    # and exposes the standard register()/unregister() pair.
    MODULES = (
        addon_preferences,
        scene_settings_migration,
        ui,
        rig_ui,
        a1_readiness_invalidation,
        auto_readiness,
        generated_material_ui,
        ui_layout,
        single_object_operator,
    )

    CONFIG_RNA_PROPERTIES = tuple(scene_properties.PROPERTIES)

    def _module_owns_runtime_registration(module: Any) -> bool:
        return module in MODULES

    def _register_config_rna() -> None:
        """Register Scene properties through the normal Blender RNA pattern."""

        scene_settings_migration.capture_pre_registration_scene_state()
        for name, value in CONFIG_RNA_PROPERTIES:
            setattr(bpy.types.Scene, name, value)

    def _unregister_config_rna() -> None:
        """Remove Scene properties owned by this extension in reverse order."""

        for name, _value in reversed(CONFIG_RNA_PROPERTIES):
            if hasattr(bpy.types.Scene, name):
                delattr(bpy.types.Scene, name)

    def _initialize_registered_logging() -> None:
        """Apply enabled add-on preferences when Blender exposes them."""

        prefs = config._addon_preferences()
        if prefs is not None:
            initialize_logging_preferences(prefs)
        config.setup_logging()

    def register() -> None:
        """Register the extension using the standard ordered Blender add-on pattern."""

        require_supported_blender_runtime(bpy)
        config._setup_default_logging()

        addon_preferences.register()
        _register_config_rna()
        scene_settings_migration.register()
        ui.register()
        rig_ui.register()
        a1_readiness_invalidation.register()
        auto_readiness.register()
        generated_material_ui.register()
        ui_layout.register()
        single_object_operator.register()

        _initialize_registered_logging()
        logger.info("Spine2D Mesh Exporter registered")

    def unregister() -> None:
        """Unregister extension owners in the reverse of registration order."""

        single_object_operator.unregister()
        ui_layout.unregister()
        generated_material_ui.unregister()
        auto_readiness.unregister()
        a1_readiness_invalidation.unregister()
        rig_ui.unregister()
        ui.unregister()
        scene_settings_migration.unregister()
        _unregister_config_rna()
        addon_preferences.unregister()
        logger.info("Spine2D Mesh Exporter unregistered")

else:
    MODULES: tuple[Any, ...] = ()
    CLASSES_TO_REGISTER: tuple[Any, ...] = ()
    CONFIG_RNA_PROPERTIES: tuple[Any, ...] = ()

    def _module_owns_runtime_registration(_module: Any) -> bool:
        return False

    def initialize_logging_preferences(_prefs: Any) -> tuple[str, ...]:
        return ()

    def register() -> None:
        """Outside Blender there is nothing to register."""

        return None

    def unregister() -> None:
        """Outside Blender there is nothing to unregister."""

        return None


if __name__ == "__main__" and bpy is not None:
    register()
