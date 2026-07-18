"""Addon preferences, diagnostics controls, and preference-only operators."""

import logging
from typing import Any

import bpy

from . import config
from .config import AddonLoggingSettings, LoggingModuleSettings


logger = logging.getLogger(__name__)
ADDON_ID = __package__ or __name__.rpartition(".")[0]


def initialize_logging_preferences(prefs: Any) -> tuple[str, ...]:
    """Discover every Python module and preserve existing per-file log levels."""

    if not hasattr(prefs, "logging_settings"):
        return ()
    return config.synchronize_logging_preferences(prefs)


class SPINE2D_OT_RefreshLoggingModules(bpy.types.Operator):
    """Rescan addon Python files while preserving configured levels."""

    bl_idname = "spine2d.refresh_logging_modules"
    bl_label = "Refresh Module List"
    bl_description = "Rescan every addon Python file used by per-module logging"

    def execute(self, context):
        try:
            prefs = context.preferences.addons[ADDON_ID].preferences
            modules = initialize_logging_preferences(prefs)
            config.setup_logging()
            self.report({"INFO"}, f"Logging modules refreshed: {len(modules)}")
            return {"FINISHED"}
        except Exception as exc:
            logger.exception("Unable to refresh logging module list")
            self.report({"ERROR"}, f"Logging refresh failed: {exc}")
            return {"CANCELLED"}


class WM_OT_UninstallAddon(bpy.types.Operator):
    bl_idname = "b2s.uninstall_addon"
    bl_label = "Uninstall Addon"
    module: bpy.props.StringProperty(default=ADDON_ID)

    def execute(self, _context):
        module_name = getattr(self, "module", None)
        if not module_name:
            base = ADDON_ID.split(".")[-1]
            installed = bpy.context.preferences.addons.keys()
            candidates = [key for key in installed if key.endswith(base)]
            module_name = candidates[0] if candidates else ADDON_ID

        logger.debug("Starting addon uninstallation for %s", module_name)
        try:
            bpy.ops.preferences.addon_disable(module=module_name)
        except Exception:
            logger.exception("Unable to disable addon %s before removal", module_name)
        try:
            bpy.ops.preferences.addon_remove(module=module_name)
            self.report({"INFO"}, "Addon uninstalled successfully.")
            return {"FINISHED"}
        except Exception as exc:
            logger.exception("Unable to remove addon %s", module_name)
            self.report({"ERROR"}, f"Uninstall failed: {exc}")
            return {"CANCELLED"}


class ModelToSpine2DAddonPreferences(bpy.types.AddonPreferences):
    bl_idname = ADDON_ID

    logging_settings: bpy.props.PointerProperty(type=AddonLoggingSettings)

    def update_logging_config(self):
        config.setup_logging()

    def draw(self, _context):
        layout = self.layout

        info_box = layout.box()
        info_box.label(text="Info & Help")
        info_box.operator(
            "wm.url_open",
            text="Project Website",
            icon="URL",
        ).url = "https://github.com/maximsokal/Blender_to_Spine_2D_Mesh_Export_Addon"

        diagnostics_box = layout.box()
        diagnostics_box.label(text="Export diagnostics")
        if not hasattr(self, "logging_settings"):
            diagnostics_box.label(
                text="Error initializing diagnostics settings. See console.",
                icon="ERROR",
            )
            return

        settings = self.logging_settings
        diagnostics_box.prop(settings, "preserve_failed_work_files")
        diagnostics_box.prop(settings, "recover_stale_work_files")
        diagnostics_box.label(
            text=(
                "Failed .spine2d-stage-* files will be kept"
                if settings.preserve_failed_work_files
                else "Failed working files are removed automatically"
            ),
            icon="INFO" if settings.preserve_failed_work_files else "CHECKMARK",
        )

        logging_box = layout.box()
        header = logging_box.row(align=True)
        header.label(text="Per-file logging")
        header.operator(
            "spine2d.refresh_logging_modules",
            text="Refresh",
            icon="FILE_REFRESH",
        )
        logging_box.prop(settings, "enable_file_logging")
        if settings.enable_file_logging:
            logging_box.prop(settings, "log_file_path")
        logging_box.prop(settings, "module_filter", icon="VIEWZOOM")

        filter_text = str(settings.module_filter or "").strip().casefold()
        visible = 0
        modules_column = logging_box.column(align=True)
        for module_setting in settings.modules:
            module_name = str(module_setting.name)
            if filter_text and filter_text not in module_name.casefold():
                continue
            row = modules_column.row(align=True)
            row.label(text=module_name)
            row.prop(module_setting, "level", text="")
            visible += 1
        logging_box.label(
            text=f"Visible modules: {visible} / {len(settings.modules)}",
            icon="INFO",
        )

        layout.separator()
        layout.label(text="Uninstall this add-on:")
        try:
            operator = layout.operator("b2s.uninstall_addon", text="Uninstall")
            operator.module = ADDON_ID
        except Exception:
            logger.exception("Unable to draw uninstall operator")
            layout.label(text="Uninstall not available", icon="ERROR")


CLASSES_TO_REGISTER = (
    LoggingModuleSettings,
    AddonLoggingSettings,
    SPINE2D_OT_RefreshLoggingModules,
    ModelToSpine2DAddonPreferences,
    WM_OT_UninstallAddon,
)


__all__ = [
    "CLASSES_TO_REGISTER",
    "ModelToSpine2DAddonPreferences",
    "SPINE2D_OT_RefreshLoggingModules",
    "WM_OT_UninstallAddon",
    "initialize_logging_preferences",
]
