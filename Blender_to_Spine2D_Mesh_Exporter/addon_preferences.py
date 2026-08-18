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


logger = logging.getLogger(__name__)
ADDON_ID = __package__ or __name__.rpartition(".")[0]

# Blender may edit AddonPreferences in a separate native Preferences window while
# the exporter panel lives in the main window. The immediate redraw covers visible
# areas; this one-shot Blender application timer schedules one redraw after the RNA
# update event returns to Blender's event loop.
_view3d_redraw_scheduled = False


def initialize_logging_preferences(prefs: Any) -> tuple[str, ...]:
    """Discover every Python module and preserve existing per-file log levels."""

    if not hasattr(prefs, "logging_settings"):
        return ()
    return config.synchronize_logging_preferences(prefs)


def _tag_all_view3d_areas_for_redraw(context: Any | None) -> int:
    """Request redraw for every visible 3D View area in every Blender window.

    Add-on Preferences can be edited in a window other than the main Blender
    window. Iterating ``WindowManager.windows`` instead of only ``context.area``
    keeps every exporter sidebar synchronized with the global preference value.

    Returns the number of VIEW_3D areas tagged. Returning the count makes the
    helper observable in tests without changing Blender state beyond redraw tags.
    """

    window_manager = getattr(context, "window_manager", None)
    if window_manager is None:
        window_manager = getattr(bpy.context, "window_manager", None)
    if window_manager is None:
        return 0

    redraw_count = 0
    for window in tuple(getattr(window_manager, "windows", ())):
        screen = getattr(window, "screen", None)
        for area in tuple(getattr(screen, "areas", ())):
            if getattr(area, "type", None) != "VIEW_3D":
                continue
            tag_redraw = getattr(area, "tag_redraw", None)
            if not callable(tag_redraw):
                continue
            tag_redraw()
            redraw_count += 1
    return redraw_count


def _deferred_view3d_redraw() -> None:
    """One-shot Blender timer callback used after an AddonPreferences edit."""

    global _view3d_redraw_scheduled

    try:
        redraw_count = _tag_all_view3d_areas_for_redraw(None)
        logger.debug(
            "Deferred Spine project-version redraw tagged %d VIEW_3D area(s)",
            redraw_count,
        )
    except Exception:
        logger.exception("Deferred Spine project-version VIEW_3D redraw failed")
    finally:
        _view3d_redraw_scheduled = False

    # Returning None unregisters a bpy.app.timers one-shot callback.
    return None


def _schedule_view3d_redraw() -> None:
    """Coalesce exact-version UI refreshes into one Blender event-loop redraw."""

    global _view3d_redraw_scheduled

    if _view3d_redraw_scheduled:
        return

    try:
        timers = getattr(getattr(bpy, "app", None), "timers", None)
        register_timer = getattr(timers, "register", None)
        if not callable(register_timer):
            logger.debug(
                "bpy.app.timers.register is unavailable; immediate redraw remains active"
            )
            return

        _view3d_redraw_scheduled = True
        register_timer(
            _deferred_view3d_redraw,
            first_interval=0.0,
        )
    except Exception:
        _view3d_redraw_scheduled = False
        logger.exception("Unable to schedule Spine project-version VIEW_3D redraw")


def _cancel_deferred_view3d_redraw() -> None:
    """Release the owned one-shot Blender timer during add-on unregistration.

    Blender 5.2 exposes ``bpy.app.timers.is_registered``. The fallback branch uses
    the process-local scheduled flag only for test/fallback environments where that
    query is unavailable. Cleanup is best-effort because a one-shot callback may have
    completed between the registration query and the unregister call.
    """

    global _view3d_redraw_scheduled

    try:
        timers = getattr(getattr(bpy, "app", None), "timers", None)
        unregister_timer = getattr(timers, "unregister", None)
        if not callable(unregister_timer):
            return

        is_registered = getattr(timers, "is_registered", None)
        if callable(is_registered):
            registered = bool(is_registered(_deferred_view3d_redraw))
        else:
            registered = bool(_view3d_redraw_scheduled)

        if registered:
            unregister_timer(_deferred_view3d_redraw)
    except Exception:
        # Do not prevent class cleanup when the one-shot callback completed between
        # the check and unregister operation or Blender is already tearing down.
        logger.debug(
            "Unable to unregister deferred Spine project-version redraw timer",
            exc_info=True,
        )
    finally:
        _view3d_redraw_scheduled = False


def _update_spine_project_version(_self: Any, context: Any) -> None:
    """Invalidate readiness and refresh all exporter sidebars after preference edits.

    Validation intentionally happens at draw/export resolution. Raising from a Blender
    StringProperty update callback on intermediate text such as ``"4.2."`` would make
    normal editing hostile and can leave the UI in a partially applied RNA state.

    Readiness invalidation and UI redraw are intentionally isolated. A failure while
    clearing one scene cache must never prevent the current exact version from becoming
    visible in an already-open exporter panel.
    """

    try:
        from .blender_adapter.a1_export_readiness import clear_a1_export_readiness

        for scene in tuple(getattr(bpy.data, "scenes", ())):
            clear_a1_export_readiness(scene)
    except Exception:
        logger.exception(
            "Unable to invalidate export readiness after Spine project version change"
        )

    try:
        redraw_count = _tag_all_view3d_areas_for_redraw(context)
        logger.debug(
            "Spine project-version edit tagged %d VIEW_3D area(s) for redraw",
            redraw_count,
        )
    except Exception:
        logger.exception("Immediate Spine project-version VIEW_3D redraw failed")

    # The Preferences editor can be a separate Blender window. A deferred redraw
    # runs after the RNA update event and keeps the main-window sidebar in sync even
    # when that window did not receive the input event itself.
    _schedule_view3d_redraw()


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
        versions_box.label(
            text="With Auto-Save Preferences off, use Blender's Save Preferences button.",
            icon="INFO",
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
    """Register preference classes in their dependency order."""

    for cls in CLASSES_TO_REGISTER:
        bpy.utils.register_class(cls)
    logger.debug("Addon preference classes registered")


def unregister() -> None:
    """Release the owned redraw timer, then unregister classes in reverse order."""

    _cancel_deferred_view3d_redraw()
    for cls in reversed(CLASSES_TO_REGISTER):
        bpy.utils.unregister_class(cls)
    logger.debug("Addon preference classes unregistered")


__all__ = [
    "CLASSES_TO_REGISTER",
    "ModelToSpine2DAddonPreferences",
    "SPINE2D_OT_RefreshLoggingModules",
    "initialize_logging_preferences",
    "register",
    "unregister",
]
