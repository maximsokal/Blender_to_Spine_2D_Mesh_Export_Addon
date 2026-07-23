# pylint: disable=import-error
"""Logging, diagnostics, and shared numeric defaults for Blender 5.2+."""

from __future__ import annotations

import logging
import logging.config
import os
from pathlib import Path
from typing import Any

import bpy

from .infrastructure.export_diagnostics import configure_export_diagnostics
from .infrastructure.logging_registry import (
    discover_python_modules,
    merge_module_levels,
    resolve_logger_name,
)


PACKAGE_LOGGER_ROOT = (__package__ or "Blender_to_Spine2D_Mesh_Exporter").strip()
ADDON_DISPLAY_NAME = PACKAGE_LOGGER_ROOT.rsplit(".", 1)[-1]
_PACKAGE_DIRECTORY = Path(__file__).resolve().parent
_LOGGING_PREFERENCES_SYNCING = False


class ShortNameFormatter(logging.Formatter):
    """Display a readable relative module path without mutating LogRecord.name."""

    def format(self, record: logging.LogRecord) -> str:
        full_name = str(record.name)
        prefix = PACKAGE_LOGGER_ROOT + "."
        if full_name == PACKAGE_LOGGER_ROOT:
            short_name = ADDON_DISPLAY_NAME
        elif full_name.startswith(prefix):
            short_name = full_name[len(prefix) :]
        else:
            short_name = full_name
        record.short_name = short_name
        return super().format(record)


logger = logging.getLogger(__name__)


def _update_logging_config(_self: Any, _context: bpy.types.Context) -> None:
    """Apply one preference change without overwriting other module levels."""

    if _LOGGING_PREFERENCES_SYNCING:
        return
    try:
        setup_logging()
    except Exception:
        logger.exception("Unable to apply updated logging/diagnostics preferences")


class LoggingModuleSettings(bpy.types.PropertyGroup):
    """Persisted logging level for one concrete Python module/file."""

    name: bpy.props.StringProperty(name="Module Name")
    level: bpy.props.EnumProperty(
        name="Log Level",
        description="Logging level for this exact Python module",
        items=(
            ("ERROR", "Error", "Errors only"),
            ("WARNING", "Warning", "Warnings and errors"),
            ("INFO", "Info", "Informational messages"),
            ("DEBUG", "Debug", "Detailed diagnostics for this module"),
        ),
        default="ERROR",
        update=_update_logging_config,
    )


class AddonLoggingSettings(bpy.types.PropertyGroup):
    """Logging and failed-work diagnostics stored in extension preferences."""

    enable_file_logging: bpy.props.BoolProperty(
        name="Enable file logging",
        description="Write extension logs to the configured file",
        default=False,
        update=_update_logging_config,
    )
    log_file_path: bpy.props.StringProperty(
        name="Log file path",
        description="File used for extension logs",
        subtype="FILE_PATH",
        default="",
        update=_update_logging_config,
    )
    module_filter: bpy.props.StringProperty(
        name="Filter modules",
        description="Show only module paths containing this text",
        default="",
    )
    preserve_failed_work_files: bpy.props.BoolProperty(
        name="Preserve failed work files",
        description=(
            "Keep .spine2d-stage-* files after a failed export for diagnostics. "
            "Disabled by default so failed working files are removed automatically"
        ),
        default=False,
        update=_update_logging_config,
    )
    recover_stale_work_files: bpy.props.BoolProperty(
        name="Recover stale work files",
        description=(
            "On the next export, restore interrupted backups and remove abandoned "
            "stage files unless preservation is enabled"
        ),
        default=True,
        update=_update_logging_config,
    )
    modules: bpy.props.CollectionProperty(type=LoggingModuleSettings)


def discover_logging_modules() -> tuple[str, ...]:
    """Discover every extension Python module, including nested Rewrite modules."""

    return discover_python_modules(
        _PACKAGE_DIRECTORY,
        root_display_name=ADDON_DISPLAY_NAME,
    )


def synchronize_logging_preferences(prefs: Any) -> tuple[str, ...]:
    """Reconcile persisted levels with source files without losing choices."""

    global _LOGGING_PREFERENCES_SYNCING
    logging_settings = getattr(prefs, "logging_settings", None)
    if logging_settings is None:
        return ()

    _LOGGING_PREFERENCES_SYNCING = True
    try:
        if not logging_settings.log_file_path:
            logging_settings.log_file_path = os.path.join(
                os.path.expanduser("~"),
                f"{ADDON_DISPLAY_NAME}.log",
            )

        existing = {
            str(item.name): str(item.level)
            for item in getattr(logging_settings, "modules", ())
            if str(getattr(item, "name", "")).strip()
        }
        discovered = discover_logging_modules()
        merged = merge_module_levels(
            discovered,
            existing,
            package_root=PACKAGE_LOGGER_ROOT,
            root_display_name=ADDON_DISPLAY_NAME,
        )
        logging_settings.modules.clear()
        for resolved in merged:
            item = logging_settings.modules.add()
            item.name = resolved.module_name
            item.level = resolved.level
    finally:
        _LOGGING_PREFERENCES_SYNCING = False
    return discovered


def _logging_formatter_config() -> dict[str, object]:
    return {
        "()": ShortNameFormatter,
        "format": "%(asctime)s - %(short_name)s - %(levelname)s - %(message)s",
        "datefmt": "%H:%M:%S",
    }


def _setup_default_logging() -> None:
    """Install an ERROR-only package logger before preferences are available."""

    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {"standard": _logging_formatter_config()},
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "standard",
                "level": "DEBUG",
            }
        },
        "loggers": {
            PACKAGE_LOGGER_ROOT: {
                "handlers": ["console"],
                "level": "ERROR",
                "propagate": False,
            }
        },
    }
    try:
        logging.config.dictConfig(logging_config)
    except Exception as exc:
        print(f"Error setting up default extension logging: {exc}")


def _addon_preferences() -> Any | None:
    try:
        addons = bpy.context.preferences.addons
    except (AttributeError, RuntimeError):
        return None
    candidates = (
        PACKAGE_LOGGER_ROOT,
        __package__,
        ADDON_DISPLAY_NAME,
    )
    for key in candidates:
        if key and key in addons:
            return addons[key].preferences
    for key in addons.keys():
        if str(key).endswith(ADDON_DISPLAY_NAME):
            return addons[key].preferences
    return None


def setup_logging() -> None:
    """Apply exact per-file levels and the failed-work diagnostics policy."""

    prefs = _addon_preferences()
    if prefs is None or not hasattr(prefs, "logging_settings"):
        configure_export_diagnostics(
            preserve_failed_work_files=False,
            recover_stale_work_files=True,
        )
        _setup_default_logging()
        return

    log_prefs = prefs.logging_settings
    if not getattr(log_prefs, "modules", None) or len(log_prefs.modules) == 0:
        synchronize_logging_preferences(prefs)
        log_prefs = prefs.logging_settings
    configure_export_diagnostics(
        preserve_failed_work_files=bool(log_prefs.preserve_failed_work_files),
        recover_stale_work_files=bool(log_prefs.recover_stale_work_files),
    )

    logging_config: dict[str, object] = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {"standard": _logging_formatter_config()},
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "standard",
                "level": "DEBUG",
            }
        },
        "loggers": {},
    }
    active_handlers = ["console"]

    if log_prefs.enable_file_logging and log_prefs.log_file_path:
        try:
            filepath = Path(bpy.path.abspath(log_prefs.log_file_path)).expanduser()
            filepath.parent.mkdir(parents=True, exist_ok=True)
            logging_config["handlers"]["file"] = {
                "class": "logging.FileHandler",
                "filename": str(filepath),
                "formatter": "standard",
                "level": "DEBUG",
                "encoding": "utf-8",
            }
            active_handlers.append("file")
        except Exception:
            logger.exception("Invalid log file path: %s", log_prefs.log_file_path)

    logger_configs: dict[str, object] = logging_config["loggers"]
    for module_setting in log_prefs.modules:
        runtime_name = resolve_logger_name(
            module_setting.name,
            package_root=PACKAGE_LOGGER_ROOT,
            root_display_name=ADDON_DISPLAY_NAME,
        )
        logger_configs[runtime_name] = {
            "handlers": active_handlers,
            "level": module_setting.level,
            "propagate": False,
        }

    logger_configs.setdefault(
        PACKAGE_LOGGER_ROOT,
        {
            "handlers": active_handlers,
            "level": "ERROR",
            "propagate": False,
        },
    )
    try:
        logging.config.dictConfig(logging_config)
    except Exception as exc:
        print(f"Error applying user logging config: {exc}")
        _setup_default_logging()


SCALING_FACTOR_WIDTH = 1.0
SCALING_FACTOR_LENGTH = 1.0
SCALING_FACTOR_HEIGHT = 1.0
UNIFORM_SCALE_MODE = "average"
REFERENCE_SCALE_MODE = "pre_unwrap"
FIXED_PIXELS_PER_BU = 100
TEXTURE_WIDTH = 1024
TEXTURE_HEIGHT = 1024
SEQUENCE_FRAME_DIGITS = 4
SEQUENCE_FRAME_DELAY = 0.0333
BAKE_MARGIN: int = 4
BAKE_TEXTURE_FORMAT: str = "PNG"
BAKE_ACTIVE_UV_NAME: str = "UVMap_for_texturing"


def calc_uniform_scale(
    texture_width: float,
    texture_height: float,
    mode: str = UNIFORM_SCALE_MODE,
) -> float:
    try:
        width = float(texture_width)
        height = float(texture_height)
    except (TypeError, ValueError):
        width = height = 1.0
    if mode == "max":
        return max(width, height)
    if mode == "min":
        return min(width, height)
    return (width + height) / 2.0


def get_default_output_dir() -> str:
    """Resolve the saved blend directory or the user home directory."""

    try:
        filepath = getattr(bpy.data, "filepath", None)
        if isinstance(filepath, str) and filepath:
            return os.path.dirname(filepath)
    except Exception:
        logger.exception("Could not resolve bpy.data.filepath")
    return os.path.expanduser("~")


__all__ = [
    "ADDON_DISPLAY_NAME",
    "AddonLoggingSettings",
    "LoggingModuleSettings",
    "PACKAGE_LOGGER_ROOT",
    "calc_uniform_scale",
    "discover_logging_modules",
    "get_default_output_dir",
    "setup_logging",
    "synchronize_logging_preferences",
]
