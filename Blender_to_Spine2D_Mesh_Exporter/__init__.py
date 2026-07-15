# __init__.py
# pylint: disable=import-error
"""Blender to Spine2D Mesh Exporter add-on entry point."""

bl_info = {
    "name": "Blender to Spine2D Mesh Exporter",
    "author": "Maxim Sokolenko",
    "version": (0, 24, 0),
    "blender": (4, 4, 0),
    "location": "View3D > UI > Blender to Spine2D Mesh Exporter",
    "description": "Converts 3D objects into a Spine2D JSON structure",
    "warning": "",
    "category": "3D View",
}

import bpy
import logging
import os

from . import config
from .config import AddonLoggingSettings, LoggingModuleSettings

logger = logging.getLogger("Blender_to_Spine2D_Mesh_Exporter")

# Install the resource-safe compatibility layer before modules such as UI and
# multi-object export bind functions from ``main``.
from . import main
from . import pipeline_v2

pipeline_v2.install(main)

from . import (
    ui,
    plane_cut,
    uv_operations,
    utils,
    json_export,
    json_merger,
    texture_baker,
    texture_baker_integration,
    seam_marker,
    multi_object_export,
)

MODULES = (
    config,
    ui,
    main,
    plane_cut,
    uv_operations,
    utils,
    json_export,
    json_merger,
    texture_baker,
    texture_baker_integration,
    seam_marker,
    multi_object_export,
)


class WM_OT_UninstallAddon(bpy.types.Operator):
    bl_idname = "b2s.uninstall_addon"
    bl_label = "Uninstall Addon"
    module: bpy.props.StringProperty(default=__package__ or __name__)

    def execute(self, context):
        mod = getattr(self, "module", None)
        if not mod:
            base = (__package__ or __name__).split(".")[-1]
            mods = bpy.context.preferences.addons.keys()
            candidates = [key for key in mods if key.endswith(base)]
            mod = candidates[0] if candidates else (__package__ or __name__)

        logger.debug("Starting addon uninstallation for: %s", mod)
        try:
            bpy.ops.preferences.addon_disable(module=mod)
        except Exception as disable_error:
            logger.error("Error disabling addon %s: %s", mod, disable_error)

        try:
            bpy.ops.preferences.addon_remove(module=mod)
            self.report({"INFO"}, "Addon uninstalled successfully.")
            return {"FINISHED"}
        except Exception as remove_error:
            logger.error("Error removing addon %s: %s", mod, remove_error)
            self.report({"ERROR"}, f"Uninstall failed: {remove_error}")
            return {"CANCELLED"}


MODULE_NAMES_FOR_LOGGING = [
    "Blender_to_Spine2D_Mesh_Exporter",
    "config",
    "ui",
    "main",
    "pipeline_v2",
    "blender_context",
    "plane_cut",
    "uv_operations",
    "utils",
    "json_export",
    "json_merger",
    "texture_baker",
    "texture_baker_integration",
    "seam_marker",
    "multi_object_export",
]


def initialize_logging_preferences(prefs):
    if not hasattr(prefs, "logging_settings"):
        return

    if not prefs.logging_settings.log_file_path:
        prefs.logging_settings.log_file_path = os.path.join(
            os.path.expanduser("~"), "Blender_to_Spine2D_Mesh_Exporter.log"
        )

    if not prefs.logging_settings.modules:
        for name in MODULE_NAMES_FOR_LOGGING:
            module = prefs.logging_settings.modules.add()
            module.name = name
            module.level = "ERROR"


class ModelToSpine2DAddonPreferences(bpy.types.AddonPreferences):
    bl_idname = __name__

    logging_settings: bpy.props.PointerProperty(type=AddonLoggingSettings)

    def update_logging_config(self):
        from . import config

        config.setup_logging()

    def draw(self, context):
        layout = self.layout

        box = layout.box()
        box.label(text="Info & Help")
        box.operator(
            "wm.url_open", text="Project Website", icon="URL"
        ).url = "https://github.com/maximsokal/Blender_to_Spine_2D_Mesh_Export_Addon"

        box = layout.box()
        box.label(text="Logging Settings")

        if not hasattr(self, "logging_settings"):
            box.label(
                text="Error initializing logging settings. See console.", icon="ERROR"
            )
            return

        log_prefs = self.logging_settings
        box.prop(log_prefs, "enable_file_logging")
        if log_prefs.enable_file_logging:
            box.prop(log_prefs, "log_file_path")

        box.separator()
        col = box.column(align=True)
        col.label(text="Module Log Levels:")
        if hasattr(log_prefs, "modules"):
            for module_setting in log_prefs.modules:
                row = col.row()
                row.label(text=module_setting.name)
                row.prop(module_setting, "level", text="")
        else:
            col.label(text="Error: Modules not registered.", icon="ERROR")

        layout.separator()
        layout.label(text="Uninstall this add-on:")
        try:
            operator = layout.operator("b2s.uninstall_addon", text="Uninstall")
            operator.module = __package__ or __name__
        except Exception as error:
            logger.error("Error adding Uninstall button: %s", error)
            layout.label(text="Uninstall not available", icon="ERROR")


CLASSES_TO_REGISTER = (
    LoggingModuleSettings,
    AddonLoggingSettings,
    ModelToSpine2DAddonPreferences,
    WM_OT_UninstallAddon,
)


def register() -> None:
    config._setup_default_logging()
    logger.debug("Registering Blender_to_Spine2D_Mesh_Exporter Add-on")

    for cls in CLASSES_TO_REGISTER:
        try:
            bpy.utils.register_class(cls)
        except Exception:
            logger.exception("Failed to register class %s", cls.__name__)

    for module in MODULES:
        try:
            if hasattr(module, "register"):
                module.register()
        except Exception:
            logger.exception("Failed to register module %s", module.__name__)

    try:
        prefs = bpy.context.preferences.addons[__name__].preferences
        initialize_logging_preferences(prefs)
        config.setup_logging()
        logger.info("User preferences for logging applied.")
    except Exception as error:
        logger.error("Could not initialize user preferences for logging: %s", error)


def unregister() -> None:
    logger.debug("Unregistering Blender_to_Spine2D_Mesh_Exporter Add-on")
    for module in reversed(MODULES):
        try:
            if hasattr(module, "unregister"):
                module.unregister()
        except Exception:
            logger.exception("Failed to unregister module %s", module.__name__)

    for cls in reversed(CLASSES_TO_REGISTER):
        try:
            bpy.utils.unregister_class(cls)
        except Exception:
            logger.exception("Failed to unregister class %s", cls.__name__)


if __name__ == "__main__":
    register()
