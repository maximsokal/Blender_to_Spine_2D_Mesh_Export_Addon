"""Addon preferences, diagnostics controls, and preference-only operators."""

import logging
from typing import Any

import bpy

from . import config
from .blender_adapter.spine_version_preferences import (
    SPINE_EXACT_VERSION_PREFERENCE_SPECS,
)
from .config import AddonLoggingSettings, LoggingModuleSettings
from .domain.spine.version_target import (
    SpineJsonTarget,
    validate_spine_json_exact_version_for_target,
)
from .infrastructure.blender_registration import (
    class_cleanup_actions,
    register_classes_transactionally,
    unregister_all_best_effort,
)


logger = logging.getLogger(__name__)
ADDON_ID = __package__ or __name__.rpartition(".")[0]


def initialize_logging_preferences(prefs: Any) -> tuple[str, ...]:
    """Discover every Python module and preserve existing per-file log levels."""

    if not hasattr(prefs, "logging_settings"):
        return ()
    return config.synchronize_logging_preferences(prefs)


def _tag_all_view3d_areas_for_redraw(context: Any | None) -> None:
    """Refresh visible exporter panels after a global exact-version edit."""

    window_manager = getattr(context, "window_manager", None)
    if window_manager is None:
        window_manager = getattr(bpy.context, "window_manager", None)
    for window in tuple(getattr(window_manager, "windows", ())):
        screen = getattr(window, "screen", None)
        for area in tuple(getattr(screen, "areas", ())):
            if getattr(area, "type", None) == "VIEW_3D":
                area.tag_redraw()


def _update_spine_project_version(_self: Any, context: Any) -> None:
    """Invalidate readiness immediately while allowing users to finish typing.

    Validation intentionally happens at draw/export resolution. Raising from a Blender
    StringProperty update callback on intermediate text such as ``"4.2."`` would make
    normal editing hostile and can leave the UI in a partially applied RNA state.
    """

    try:
        from .blender_adapter.a1_export_readiness import clear_a1_export_readiness

        for scene in tuple(getattr(bpy.data, "scenes", ())):
            clear_a1_export_readiness(scene)
        _tag_all_view3d_areas_for_redraw(context)
    except Exception:
        logger.exception(
            "Unable to invalidate export readiness after Spine project version change"
        )


class SPINE2D_OT_RefreshLoggingModules(bpy.types.Operator):
    """Rescan add-on modules without changing stored per-file log levels."""

    bl_idname = "spine2d.refresh_logging_modules"
    bl_label = "Refresh Module List"
    bl_description = "Rescan every add-on Python file used by per-module logging"

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


class ModelToSpine2DAddonPreferences(bpy.types.AddonPreferences):
    bl_idname = ADDON_ID

    logging_settings: bpy.props.PointerProperty(type=AddonLoggingSettings)

    spine2d_exact_version_3_8: bpy.props.StringProperty(
        name="Spine 3.8 project version",
        description=(
            "Exact Spine Editor/project version written to JSON while using the 3.8 codec"
        ),
        default=SpineJsonTarget.SPINE_3_8.exact_version,
        update=_update_spine_project_version,
    )
    spine2d_exact_version_4_0: bpy.props.StringProperty(
        name="Spine 4.0 project version",
        description=(
            "Exact Spine Editor/project version written to JSON while using the 4.0 codec"
        ),
        default=SpineJsonTarget.SPINE_4_0.exact_version,
        update=_update_spine_project_version,
    )
    spine2d_exact_version_4_1: bpy.props.StringProperty(
        name="Spine 4.1 project version",
        description=(
            "Exact Spine Editor/project version written to JSON while using the 4.1 codec"
        ),
        default=SpineJsonTarget.SPINE_4_1.exact_version,
        update=_update_spine_project_version,
    )
    spine2d_exact_version_4_2: bpy.props.StringProperty(
        name="Spine 4.2 project version",
        description=(
            "Exact Spine Editor/project version written to JSON while using the 4.2 codec"
        ),
        default=SpineJsonTarget.SPINE_4_2.exact_version,
        update=_update_spine_project_version,
    )
    spine2d_exact_version_4_3: bpy.props.StringProperty(
        name="Spine 4.3 project version",
        description=(
            "Exact Spine Editor/project version written to JSON while using the 4.3 codec"
        ),
        default=SpineJsonTarget.SPINE_4_3.exact_version,
        update=_update_spine_project_version,
    )

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

        versions_box = layout.box()
        versions_box.label(text="Spine project JSON versions")
        versions_box.label(
            text="Choose the exact Editor/project patch version for each schema family.",
            icon="INFO",
        )
        versions_box.label(
            text="These are global add-on preferences and are reused across .blend files.",
        )
        for spec in SPINE_EXACT_VERSION_PREFERENCE_SPECS:
            versions_box.prop(self, spec.property_name, text=spec.label)
            raw_value = getattr(self, spec.property_name, "")
            try:
                validate_spine_json_exact_version_for_target(spec.target, raw_value)
            except (TypeError, ValueError) as exc:
                versions_box.label(
                    text=f"{spec.target.label}: {exc}",
                    icon="ERROR",
                )

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
        layout.label(
            text="Manage or uninstall this extension in Preferences > Extensions.",
            icon="INFO",
        )


CLASSES_TO_REGISTER = (
    LoggingModuleSettings,
    AddonLoggingSettings,
    SPINE2D_OT_RefreshLoggingModules,
    ModelToSpine2DAddonPreferences,
)


def register() -> None:
    """Register all preference classes atomically in dependency order."""

    try:
        register_classes_transactionally(
            CLASSES_TO_REGISTER,
            register_class=bpy.utils.register_class,
            unregister_class=bpy.utils.unregister_class,
        )
    except Exception:
        logger.exception("Addon preference registration failed")
        raise
    logger.debug("Addon preference classes registered")


def unregister() -> None:
    """Attempt every preference-class cleanup before reporting aggregate failure."""

    try:
        unregister_all_best_effort(
            class_cleanup_actions(
                CLASSES_TO_REGISTER,
                unregister_class=bpy.utils.unregister_class,
            ),
            operation="addon preference unregistration",
        )
    except Exception:
        logger.exception("Addon preference unregistration failed")
        raise
    logger.debug("Addon preference classes unregistered")


__all__ = [
    "CLASSES_TO_REGISTER",
    "ModelToSpine2DAddonPreferences",
    "SPINE2D_OT_RefreshLoggingModules",
    "initialize_logging_preferences",
    "register",
    "unregister",
]
