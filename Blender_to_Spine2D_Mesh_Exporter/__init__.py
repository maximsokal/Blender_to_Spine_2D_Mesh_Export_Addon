# __init__.py
# pylint: disable=import-error

"""
Blender to Spine2D Mesh Exporter
Copyright (c) 2025 Maxim Sokolenko

This file is part of Blender to Spine2D Mesh Exporter.

Blender to Spine2D Mesh Exporter is free software: you can redistribute it
and/or modify it under the terms of the GNU General Public License as
published by the Free Software Foundation, either version 3 of the License,
or (at your option) any later version.

Blender to Spine2D Mesh Exporter is distributed in the hope that it will
be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with Blender to Spine2D Mesh Exporter. If not, see <https://www.gnu.org/licenses/>.
This file serves as the main entry point for the 'Blender to Spine2D Mesh Exporter' addon.
Its primary responsibilities are:
1.  Addon Metadata: It contains the `bl_info` dictionary, which provides Blender with essential information about the addon, such as its name, author, version, and category.
2.  Module Management: It imports all the necessary sub-modules of the addon (like `ui`, `main`, `texture_baker`, etc.) and organizes them into a `MODULES` tuple.
3.  Registration and Unregistration: It defines the `register()` and `unregister()` functions, which are the core hooks Blender uses to load and unload the addon. These functions iterate through the `MODULES` tuple and call the respective `register()` or `unregister()` function within each sub-module, ensuring that all classes, operators, and properties are correctly added to or removed from Blender.
4.  Logging Initialization: It calls the `setup_logging()` function from the `config` module during registration to configure the addon's logging system.
5.  Addon Preferences and Uninstallation: It defines a custom addon preferences class (`ModelToSpine2DAddonPreferences`) and an operator (`WM_OT_UninstallAddon`) to provide a user-friendly way to uninstall the addon directly from the Blender preferences window.

ATTENTION: - This file orchestrates the entire addon's lifecycle. The order of modules in the `MODULES` tuple is important, especially for unregistration, which happens in reverse order to prevent dependency errors. Modifying the `register()` or `unregister()` logic can break the addon's ability to load or unload correctly. The `bl_info` dictionary must be kept up-to-date, particularly the "blender" version, to ensure compatibility.
Author: Maxim Sokolenko
"""

bl_info = {
    "name": "Blender to Spine2D Mesh Exporter",
    "author": "Maxim Sokolenko",
    "version": (0, 23, 0),
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

from . import (
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
            candidates = [k for k in mods if k.endswith(base)]
            mod = candidates[0] if candidates else (__package__ or __name__)

        logger.debug(f"Starting addon uninstallation for: {mod}")
        try:
            logger.debug(
                "Attempting to disable addon via bpy.ops.preferences.addon_disable"
            )
            bpy.ops.preferences.addon_disable(module=mod)
        except Exception as e_disable:
            logger.error(f"Error disabling addon {mod}: {e_disable}")
        try:
            logger.debug(
                "Attempting to remove addon via bpy.ops.preferences.addon_remove"
            )
            bpy.ops.preferences.addon_remove(module=mod)
            self.report({"INFO"}, "Addon uninstalled successfully.")
            logger.debug(f"Addon {mod} successfully removed!")
            return {"FINISHED"}
        except Exception as e_remove:
            logger.error(f"Error removing addon {mod}: {e_remove}")
            self.report({"ERROR"}, f"Uninstall failed: {e_remove}")
            return {"CANCELLED"}


MODULE_NAMES_FOR_LOGGING = [
    "Blender_to_Spine2D_Mesh_Exporter",
    "config",
    "ui",
    "main",
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
        default_path = os.path.join(
            os.path.expanduser("~"), "Blender_to_Spine2D_Mesh_Exporter.log"
        )
        prefs.logging_settings.log_file_path = default_path

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
            op = layout.operator("b2s.uninstall_addon", text="Uninstall")
            op.module = __package__ or __name__
        except Exception as e:
            logger.error("Error adding Uninstall button: " + str(e))
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
            logger.exception(f"Failed to register class {cls.__name__}")

    for module in MODULES:
        try:
            if hasattr(module, "register"):
                module.register()
        except Exception:
            logger.exception(f"Failed to register module {module.__name__}")
    try:
        prefs = bpy.context.preferences.addons[__name__].preferences
        initialize_logging_preferences(prefs)
        config.setup_logging()
        logger.info("User preferences for logging applied.")
    except Exception as e:
        logger.error(f"Could not initialize user preferences for logging: {e}")


def unregister() -> None:
    logger.debug("Unregistering Blender_to_Spine2D_Mesh_Exporter Add-on")
    for module in reversed(MODULES):
        try:
            if hasattr(module, "unregister"):
                module.unregister()
        except Exception:
            logger.exception(f"Failed to unregister module {module.__name__}")
    for cls in reversed(CLASSES_TO_REGISTER):
        try:
            bpy.utils.unregister_class(cls)
        except Exception:
            logger.exception(f"Failed to unregister class {cls.__name__}")


if __name__ == "__main__":
    register()
