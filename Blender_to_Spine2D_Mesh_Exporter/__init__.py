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

import logging
import os

import bpy

from . import config
from .config import AddonLoggingSettings, LoggingModuleSettings

logger = logging.getLogger("Blender_to_Spine2D_Mesh_Exporter")

# Import the legacy geometry/conversion module first, then install the new
# resource-safe orchestration before UI and multi-object modules import public
# functions from ``main``.  This preserves the add-on API during the staged
# rewrite without duplicating operator identifiers or registration hooks.
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
    """Disable and remove the current add-on package."""

    bl_idname = "b2s.uninstall_addon"
    bl_label = "Uninstall Addon"
    module: bpy.props.StringProperty(default=__package__ or __name__)

    def execute(self, context):
        del context
        module_name = getattr(self, "module", None)
        if not module_name:
            base = (__package__ or __name__).split(".")[-1]
            installed_modules = bpy.context.preferences.addons.keys()
            candidates = [key for key in installed_modules if key.endswith(base)]
            module_name = candidates[0] if candidates else (__package__ or __name__)

        logger.debug("Starting add-on uninstallation for %s", module_name)
        try:
            bpy.ops.preferences.addon_disable(module=module_name)
        except (RuntimeError, TypeError):
            logger.exception("Failed to disable add-on %s", module_name)

        try:
            bpy.ops.preferences.addon_remove(module=module_name)
            self.report({"INFO"}, "Addon uninstalled successfully.")
            return {"FINISHED"}
        except (RuntimeError, TypeError) as exc:
            logger.exception("Failed to remove add-on %s", module_name)
            self.report({"ERROR"}, f"Uninstall failed: {exc}")
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


def initialize_logging_preferences(prefs) -> None:
    """Populate missing per-module logging preferences."""
    if not hasattr(prefs, "logging_settings"):
        return

    settings = prefs.logging_settings
    if not settings.log_file_path:
        settings.log_file_path = os.path.join(
            os.path.expanduser("~"),
            "Blender_to_Spine2D_Mesh_Exporter.log",
        )

    existing_names = {
        module.name for module in settings.modules
    } if settings.modules else set()
    for name in MODULE_NAMES_FOR_LOGGING:
        if name in existing_names:
            continue
        module = settings.modules.add()
        module.name = name
        module.level = "ERROR"


class ModelToSpine2DAddonPreferences(bpy.types.AddonPreferences):
    """Add-on information, logging controls and uninstall action."""

    bl_idname = __name__
    logging_settings: bpy.props.PointerProperty(type=AddonLoggingSettings)

    def update_logging_config(self):
        config.setup_logging()

    def draw(self, context):
        del context
        layout = self.layout

        info_box = layout.box()
        info_box.label(text="Info & Help")
        info_box.operator(
            "wm.url_open",
            text="Project Website",
            icon="URL",
        ).url = "https://github.com/maximsokal/Blender_to_Spine_2D_Mesh_Export_Addon"

        logging_box = layout.box()
        logging_box.label(text="Logging Settings")
        if not hasattr(self, "logging_settings"):
            logging_box.label(
                text="Error initializing logging settings. See console.",
                icon="ERROR",
            )
            return

        settings = self.logging_settings
        logging_box.prop(settings, "enable_file_logging")
        if settings.enable_file_logging:
            logging_box.prop(settings, "log_file_path")

        logging_box.separator()
        column = logging_box.column(align=True)
        column.label(text="Module Log Levels:")
        if hasattr(settings, "modules"):
            for module_setting in settings.modules:
                row = column.row()
                row.label(text=module_setting.name)
                row.prop(module_setting, "level", text="")
        else:
            column.label(text="Error: Modules not registered.", icon="ERROR")

        layout.separator()
        layout.label(text="Uninstall this add-on:")
        try:
            operator = layout.operator("b2s.uninstall_addon", text="Uninstall")
            operator.module = __package__ or __name__
        except (AttributeError, RuntimeError):
            logger.exception("Failed to draw the uninstall button")
            layout.label(text="Uninstall not available", icon="ERROR")


CLASSES_TO_REGISTER = (
    LoggingModuleSettings,
    AddonLoggingSettings,
    ModelToSpine2DAddonPreferences,
    WM_OT_UninstallAddon,
)


def register() -> None:
    """Register preferences first, then add-on modules in dependency order."""
    config._setup_default_logging()
    logger.debug("Registering Blender_to_Spine2D_Mesh_Exporter")

    for cls in CLASSES_TO_REGISTER:
        try:
            bpy.utils.register_class(cls)
        except (RuntimeError, ValueError):
            logger.exception("Failed to register class %s", cls.__name__)

    for module in MODULES:
        try:
            register_function = getattr(module, "register", None)
            if callable(register_function):
                register_function()
        except Exception:
            logger.exception("Failed to register module %s", module.__name__)

    try:
        preferences = bpy.context.preferences.addons[__name__].preferences
        initialize_logging_preferences(preferences)
        config.setup_logging()
        logger.info("User logging preferences applied")
    except (AttributeError, KeyError, RuntimeError):
        logger.exception("Could not initialize logging preferences")


def unregister() -> None:
    """Unregister modules and classes in reverse dependency order."""
    logger.debug("Unregistering Blender_to_Spine2D_Mesh_Exporter")
    for module in reversed(MODULES):
        try:
            unregister_function = getattr(module, "unregister", None)
            if callable(unregister_function):
                unregister_function()
        except Exception:
            logger.exception("Failed to unregister module %s", module.__name__)

    for cls in reversed(CLASSES_TO_REGISTER):
        try:
            bpy.utils.unregister_class(cls)
        except (RuntimeError, ValueError):
            logger.exception("Failed to unregister class %s", cls.__name__)


if __name__ == "__main__":
    register()
