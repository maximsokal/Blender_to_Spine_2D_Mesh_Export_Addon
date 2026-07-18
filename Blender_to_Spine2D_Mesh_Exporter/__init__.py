# pylint: disable=import-error
"""Main entry point for the Blender to Spine2D Mesh Exporter add-on."""

from __future__ import annotations

import logging
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


def initialize_logging_preferences(prefs: Any) -> tuple[str, ...]:
    """Discover every Python module and preserve existing per-file log levels."""

    if bpy is None or not hasattr(prefs, "logging_settings"):
        return ()
    return config.synchronize_logging_preferences(prefs)


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

    # Legacy modules remain importable only because the explicit Legacy backend still uses
    # them. ``main`` does not own runtime registration; the Rewrite-aware operator owns the
    # public ``object.save_uv_as_json`` identifier.
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

    class SPINE2D_OT_RefreshLoggingModules(bpy.types.Operator):
        """Rescan addon Python files while preserving configured levels."""

        bl_idname = "spine2d.refresh_logging_modules"
        bl_label = "Refresh Module List"
        bl_description = "Rescan every addon Python file used by per-module logging"

        def execute(self, context):
            try:
                prefs = context.preferences.addons[__name__].preferences
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
        module: bpy.props.StringProperty(default=__package__ or __name__)

        def execute(self, context):
            module_name = getattr(self, "module", None)
            if not module_name:
                base = (__package__ or __name__).split(".")[-1]
                installed = bpy.context.preferences.addons.keys()
                candidates = [key for key in installed if key.endswith(base)]
                module_name = candidates[0] if candidates else (__package__ or __name__)

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
        bl_idname = __name__

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
            if settings.preserve_failed_work_files:
                diagnostics_box.label(
                    text="Failed .spine2d-stage-* files will be kept",
                    icon="INFO",
                )
            else:
                diagnostics_box.label(
                    text="Failed working files are removed automatically",
                    icon="CHECKMARK",
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
                operator.module = __package__ or __name__
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

    def _module_owns_runtime_registration(module: Any) -> bool:
        return module is not main

    def register() -> None:
        config._setup_default_logging()
        logger.debug("Registering Blender_to_Spine2D_Mesh_Exporter")

        registered_classes: list[type] = []
        registered_modules: list[Any] = []
        try:
            for cls in CLASSES_TO_REGISTER:
                bpy.utils.register_class(cls)
                registered_classes.append(cls)

            for module in MODULES:
                if _module_owns_runtime_registration(module) and hasattr(module, "register"):
                    module.register()
                    registered_modules.append(module)

            prefs = bpy.context.preferences.addons[__name__].preferences
            initialize_logging_preferences(prefs)
            config.setup_logging()
            logger.info("User logging and diagnostics preferences applied")
        except Exception:
            logger.exception("Addon registration failed")
            for module in reversed(registered_modules):
                try:
                    if hasattr(module, "unregister"):
                        module.unregister()
                except Exception:
                    logger.exception(
                        "Registration rollback failed for module %s",
                        module.__name__,
                    )
            for cls in reversed(registered_classes):
                try:
                    bpy.utils.unregister_class(cls)
                except Exception:
                    logger.exception("Registration rollback failed for %s", cls.__name__)
            raise

    def unregister() -> None:
        logger.debug("Unregistering Blender_to_Spine2D_Mesh_Exporter")
        errors: list[Exception] = []
        for module in reversed(MODULES):
            try:
                if _module_owns_runtime_registration(module) and hasattr(module, "unregister"):
                    module.unregister()
            except Exception as exc:
                errors.append(exc)
                logger.exception("Failed to unregister module %s", module.__name__)
        for cls in reversed(CLASSES_TO_REGISTER):
            try:
                bpy.utils.unregister_class(cls)
            except Exception as exc:
                errors.append(exc)
                logger.exception("Failed to unregister class %s", cls.__name__)
        if errors:
            raise RuntimeError(
                f"Addon unregistration failed {len(errors)} time(s)"
            ) from errors[0]

else:
    MODULES: tuple[Any, ...] = ()
    CLASSES_TO_REGISTER: tuple[Any, ...] = ()

    def _module_owns_runtime_registration(_module: Any) -> bool:
        return False

    def register() -> None:
        raise RuntimeError("Blender bpy module is required to register the add-on")

    def unregister() -> None:
        return None


if __name__ == "__main__" and bpy is not None:
    register()
