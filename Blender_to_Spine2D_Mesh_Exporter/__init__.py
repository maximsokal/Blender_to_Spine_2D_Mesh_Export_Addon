# __init__.py
# pylint: disable=import-error

"""Main entry point for the Blender to Spine2D Mesh Exporter add-on.

The package also contains Blender-independent domain/application code and command-line
parity tools. Importing those modules with normal Python must not eagerly import every
legacy ``bpy`` module. Blender lifecycle classes are therefore defined only when a
real Blender Python API is available; the in-Blender registration path is unchanged.
"""

from __future__ import annotations

import logging
import os
from typing import Any


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

logger = logging.getLogger("Blender_to_Spine2D_Mesh_Exporter")

try:
    import bpy as _bpy  # type: ignore
except ModuleNotFoundError:
    _bpy = None

# Some normal Python environments contain a placeholder package named ``bpy`` that
# is not Blender's runtime API. Importing legacy modules against it fails later with
# misleading AttributeError exceptions. Require the core lifecycle surfaces instead.
if _bpy is not None and all(
    hasattr(_bpy, attribute) for attribute in ("types", "props", "utils", "context")
):
    bpy = _bpy
else:
    bpy = None


MODULE_NAMES_FOR_LOGGING = [
    "Blender_to_Spine2D_Mesh_Exporter",
    "config",
    "ui",
    "single_object_operator",
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


def initialize_logging_preferences(prefs: Any) -> None:
    if not hasattr(prefs, "logging_settings"):
        return

    if not prefs.logging_settings.log_file_path:
        default_path = os.path.join(
            os.path.expanduser("~"),
            "Blender_to_Spine2D_Mesh_Exporter.log",
        )
        prefs.logging_settings.log_file_path = default_path

    if not prefs.logging_settings.modules:
        for name in MODULE_NAMES_FOR_LOGGING:
            module = prefs.logging_settings.modules.add()
            module.name = name
            module.level = "ERROR"


if bpy is not None:
    from . import config
    from .config import AddonLoggingSettings, LoggingModuleSettings
    from . import (
        ui,
        main,
        single_object_operator,
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

    # ``main`` remains available as the legacy implementation/reference. Its
    # register hooks are skipped because ``single_object_operator`` owns the same
    # public ``object.save_uv_as_json`` operator ID.
    MODULES = (
        config,
        ui,
        main,
        single_object_operator,
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

    class ModelToSpine2DAddonPreferences(bpy.types.AddonPreferences):
        bl_idname = __name__

        logging_settings: bpy.props.PointerProperty(type=AddonLoggingSettings)

        def update_logging_config(self):
            config.setup_logging()

        def draw(self, context):
            layout = self.layout

            box = layout.box()
            box.label(text="Info & Help")
            box.operator(
                "wm.url_open",
                text="Project Website",
                icon="URL",
            ).url = "https://github.com/maximsokal/Blender_to_Spine_2D_Mesh_Export_Addon"

            box = layout.box()
            box.label(text="Logging Settings")
            if not hasattr(self, "logging_settings"):
                box.label(
                    text="Error initializing logging settings. See console.",
                    icon="ERROR",
                )
                return

            log_prefs = self.logging_settings
            box.prop(log_prefs, "enable_file_logging")
            if log_prefs.enable_file_logging:
                box.prop(log_prefs, "log_file_path")

            box.separator()
            column = box.column(align=True)
            column.label(text="Module Log Levels:")
            if hasattr(log_prefs, "modules"):
                for module_setting in log_prefs.modules:
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
            except Exception as exc:
                logger.error("Error adding Uninstall button: %s", exc)
                layout.label(text="Uninstall not available", icon="ERROR")

    CLASSES_TO_REGISTER = (
        LoggingModuleSettings,
        AddonLoggingSettings,
        ModelToSpine2DAddonPreferences,
        WM_OT_UninstallAddon,
    )

    def _module_owns_runtime_registration(module: Any) -> bool:
        return module is not main

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
                if _module_owns_runtime_registration(module) and hasattr(
                    module,
                    "register",
                ):
                    module.register()
            except Exception:
                logger.exception("Failed to register module %s", module.__name__)
        try:
            prefs = bpy.context.preferences.addons[__name__].preferences
            initialize_logging_preferences(prefs)
            config.setup_logging()
            logger.info("User preferences for logging applied.")
        except Exception as exc:
            logger.error("Could not initialize user preferences for logging: %s", exc)

    def unregister() -> None:
        logger.debug("Unregistering Blender_to_Spine2D_Mesh_Exporter Add-on")
        for module in reversed(MODULES):
            try:
                if _module_owns_runtime_registration(module) and hasattr(
                    module,
                    "unregister",
                ):
                    module.unregister()
            except Exception:
                logger.exception("Failed to unregister module %s", module.__name__)
        for cls in reversed(CLASSES_TO_REGISTER):
            try:
                bpy.utils.unregister_class(cls)
            except Exception:
                logger.exception("Failed to unregister class %s", cls.__name__)

else:
    MODULES: tuple[Any, ...] = ()
    CLASSES_TO_REGISTER: tuple[Any, ...] = ()

    def _module_owns_runtime_registration(module: Any) -> bool:
        return False

    def register() -> None:
        raise RuntimeError("Blender bpy module is required to register the add-on")

    def unregister() -> None:
        return None


if __name__ == "__main__" and bpy is not None:
    register()
