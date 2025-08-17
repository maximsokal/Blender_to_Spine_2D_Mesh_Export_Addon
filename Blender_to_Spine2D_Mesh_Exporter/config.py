# config.py
# pylint: disable=import-error
"""
This file serves as the central configuration module for the 'Blender_to_Spine2D_Mesh_Exporter' addon.
It is responsible for several key functions:
1.  Logging Setup: It configures a custom logging system for the entire addon, with specific levels for different modules, using a custom formatter to shorten logger names for readability.
2.  Default Parameters: It defines global constants and default values for various addon operations, including scaling factors, texture dimensions, and texture baking settings (e.g., margin, format, UV map names).
3.  Blender UI Properties: It defines a list of custom properties (bpy.props) and registers them with Blender's Scene object (bpy.types.Scene). This allows users to access and modify addon settings directly from the Blender UI.
4.  Dynamic Configuration: It includes getter and setter functions to handle dynamic property updates, such as validating texture sizes and determining the default output directory based on the current .blend file's location.

ATTENTION: - This file is a critical dependency for the entire addon. Any changes to the property names, global constants (like BAKE_MARGIN or BAKE_ACTIVE_UV_NAME), or the logging setup will have a direct impact on other modules, including UI, baking, and export logic. The register() and unregister() functions are essential for correctly integrating the addon's settings into Blender.
Author: Maxim Sokolenko
"""
import os
import bpy
import logging
import logging.config


class ShortNameFormatter(logging.Formatter):
    def format(self, record):
        record.name = record.name.split(".")[-1]
        return super().format(record)


logger = logging.getLogger(__name__)


def _update_ui_for_paths(self, context):
    """Force UI refresh for path-dependent labels."""
    for window in context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()


def _update_logging_config(self, context):
    """
    Callback function when a log level is changed in the UI.
    'self' refers to the LoggingModuleSettings instance that was modified.
    """
    try:
        # Get the addon's preferences to access all logging settings
        prefs = context.preferences.addons[__package__].preferences
        log_prefs = prefs.logging_settings

        # Check if 'self' is valid before accessing its properties.
        # If the main "Blender_to_Spine2D_Mesh_Exporter" logger was changed, apply its level to all other modules.
        if self and self.name == "Blender_to_Spine2D_Mesh_Exporter":
            new_level = self.level
            # Iterate through all module settings and update their level.
            for module_setting in log_prefs.modules:
                if module_setting.name != "Blender_to_Spine2D_Mesh_Exporter":
                    module_setting.level = new_level

        # Always call the main update function to apply the new configuration.
        # This will now be reached in all successful cases.
        prefs.update_logging_config()

    except (AttributeError, KeyError):
        # This can happen during Blender's startup/shutdown, so we fail gracefully.
        logger.warning("Could not find addon preferences to update logging config.")


class LoggingModuleSettings(bpy.types.PropertyGroup):
    """Logging settings for a single module"""

    name: bpy.props.StringProperty(name="Module Name")
    level: bpy.props.EnumProperty(
        name="Log Level",
        description="Logging level for this module",
        items=[
            ("ERROR", "Error", "Errors only (recommended)"),
            ("WARNING", "Warning", "Warnings and errors"),
            ("INFO", "Info", "Informational messages"),
            ("DEBUG", "Debug", "All messages for debugging"),
        ],
        default="ERROR",
        update=_update_logging_config,
    )


class AddonLoggingSettings(bpy.types.PropertyGroup):
    """Global addon logging settings"""

    enable_file_logging: bpy.props.BoolProperty(
        name="Enable file logging",
        description="Write all addon logs to the specified file",
        default=False,
        update=_update_logging_config,  # CHANGE: Using a named function
    )
    log_file_path: bpy.props.StringProperty(
        name="Log file path",
        description="File for saving logs. Will be created if it does not exist",
        subtype="FILE_PATH",
        default="",
        update=_update_logging_config,  # CHANGE: Using a named function
    )
    modules: bpy.props.CollectionProperty(type=LoggingModuleSettings)


# =============================================================================


def _setup_default_logging():
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "()": ShortNameFormatter,
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "datefmt": "%H:%M:%S",
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "standard",
                "level": "ERROR",
            }
        },
        "loggers": {
            "Blender_to_Spine2D_Mesh_Exporter": {
                "handlers": ["console"],
                "level": "ERROR",
                "propagate": False,
            },
            "bl_ext.user_default.Blender_to_Spine2D_Mesh_Exporter.config": {
                "handlers": ["console"],
                "level": "ERROR",
                "propagate": False,
            },
            "bl_ext.user_default.Blender_to_Spine2D_Mesh_Exporter.json_export": {
                "handlers": ["console"],
                "level": "ERROR",
                "propagate": False,
            },
            "bl_ext.user_default.Blender_to_Spine2D_Mesh_Exporter.json_merger": {
                "handlers": ["console"],
                "level": "ERROR",
                "propagate": False,
            },
            "bl_ext.user_default.Blender_to_Spine2D_Mesh_Exporter.multi_object_export": {
                "handlers": ["console"],
                "level": "ERROR",
                "propagate": False,
            },
            "bl_ext.user_default.Blender_to_Spine2D_Mesh_Exporter.main": {
                "handlers": ["console"],
                "level": "ERROR",
                "propagate": False,
            },
            "bl_ext.user_default.Blender_to_Spine2D_Mesh_Exporter.plane_cut": {
                "handlers": ["console"],
                "level": "ERROR",
                "propagate": False,
            },
            "bl_ext.user_default.Blender_to_Spine2D_Mesh_Exporter.seam_marker": {
                "handlers": ["console"],
                "level": "ERROR",
                "propagate": False,
            },
            "bl_ext.user_default.Blender_to_Spine2D_Mesh_Exporter.texture_baker_integration": {
                "handlers": ["console"],
                "level": "ERROR",
                "propagate": False,
            },
            "bl_ext.user_default.Blender_to_Spine2D_Mesh_Exporter.texture_baker": {
                "handlers": ["console"],
                "level": "ERROR",
                "propagate": False,
            },
            "bl_ext.user_default.Blender_to_Spine2D_Mesh_Exporter.ui": {
                "handlers": ["console"],
                "level": "ERROR",
                "propagate": False,
            },
            "bl_ext.user_default.Blender_to_Spine2D_Mesh_Exporter.utils": {
                "handlers": ["console"],
                "level": "ERROR",
                "propagate": False,
            },
            "bl_ext.user_default.Blender_to_Spine2D_Mesh_Exporter.uv_operations": {
                "handlers": ["console"],
                "level": "ERROR",
                "propagate": False,
            },
        },
    }
    try:
        logging.config.dictConfig(logging_config)
        logger.debug(
            "Default logging configuration applied successfully from config.py."
        )
    except Exception as e:
        print("Error setting up default logging in config.py:", e)


def setup_logging():
    try:
        prefs = bpy.context.preferences.addons[__package__].preferences
        log_prefs = prefs.logging_settings
    except (AttributeError, KeyError):
        _setup_default_logging()
        return

    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "()": ShortNameFormatter,
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "datefmt": "%H:%M:%S",
            }
        },
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
            filepath = bpy.path.abspath(log_prefs.log_file_path)
            if filepath:
                log_dir = os.path.dirname(filepath)
                if not os.path.exists(log_dir):
                    os.makedirs(log_dir, exist_ok=True)
                logging_config["handlers"]["file"] = {
                    "class": "logging.FileHandler",
                    "filename": filepath,
                    "formatter": "standard",
                    "level": "DEBUG",
                    "encoding": "utf-8",
                }
                active_handlers.append("file")
        except Exception as e:
            logger.error(f"Invalid log file path: {e}")

    if hasattr(log_prefs, "modules"):
        for module_setting in log_prefs.modules:
            if module_setting.name == "Blender_to_Spine2D_Mesh_Exporter":
                logger_name = "Blender_to_Spine2D_Mesh_Exporter"
            else:
                logger_name = f"bl_ext.user_default.Blender_to_Spine2D_Mesh_Exporter.{module_setting.name}"

            logging_config["loggers"][logger_name] = {
                "handlers": active_handlers,
                "level": module_setting.level,
                "propagate": False,
            }

    try:
        logging.config.dictConfig(logging_config)
    except Exception as e:
        print(f"Error applying user logging config: {e}")
        _setup_default_logging()


logger.debug("[LOG] Loading config.py")

SCALING_FACTOR_WIDTH = 1.0
SCALING_FACTOR_LENGTH = 1.0
SCALING_FACTOR_HEIGHT = 1.0

UNIFORM_SCALE_MODE = "average"


def calc_uniform_scale(
    texture_width: float, texture_height: float, mode: str = UNIFORM_SCALE_MODE
) -> float:
    """
    Returns a uniform scale factor based on the chosen strategy.
    """
    try:
        w = float(texture_width)
        h = float(texture_height)
    except Exception:
        w = h = 1.0
    if mode == "max":
        return max(w, h)
    if mode == "min":
        return min(w, h)
    return (w + h) / 2.0


REFERENCE_SCALE_MODE = "pre_unwrap"
FIXED_PIXELS_PER_BU = 100
TEXTURE_WIDTH = 1024
TEXTURE_HEIGHT = 1024
SEQUENCE_FRAME_DIGITS = 4
SEQUENCE_FRAME_DELAY = 0.0333

# Baking defaults. These values were previously hard‑coded in
# texture_baker_integration.py.  Moving them to the configuration file
# allows end‑users to adjust baking parameters without modifying code.
BAKE_MARGIN: int = 4
BAKE_TEXTURE_FORMAT: str = "PNG"
# The name of the UV layer used when baking into a texture.  Should
# correspond to the UV map prepared for texturing the target object.
BAKE_ACTIVE_UV_NAME: str = "UVMap_for_texturing"


# --- Property Definitions ---


def set_frames_for_render(self, value):
    max_f = getattr(bpy.context.scene, "frame_end", 0)
    iv = max(0, int(value))
    self["spine2d_frames_for_render"] = min(iv, max_f)


def get_frames_for_render(self):
    return int(self.get("spine2d_frames_for_render", 0))


def get_default_output_dir():
    """
    Returns the directory of the saved .blend file, or the user's home directory.
    """
    try:
        fp = getattr(bpy.data, "filepath", None)
        if isinstance(fp, str) and fp:
            return os.path.dirname(fp)
    except Exception as e:
        logger.error(f"[ERROR] Could not get bpy.data.filepath: {e}")
    return os.path.expanduser("~")


def set_texture_size(self, value):
    """
    Setter for Scene.spine2d_texture_size property.
    """
    try:
        if value < 64:
            value = 64
        elif value > 4096:
            value = 4096
        if value % 2 != 0:
            value = (
                value - 1
                if (value - (value - 1)) <= ((value + 1) - value)
                else value + 1
            )
        self["spine2d_texture_size"] = value
        global TEXTURE_WIDTH, TEXTURE_HEIGHT
        TEXTURE_WIDTH = value
        TEXTURE_HEIGHT = value
        logger.debug(f"[set_texture_size] Texture size set to = {value}")
    except Exception as e:
        logger.error(f"[ERROR] set_texture_size: {e}")


def get_texture_size(self):
    """
    Getter for Scene.spine2d_texture_size property.
    """
    return self.get("spine2d_texture_size", 1024)


# List of properties to register/unregister
PROPERTIES = [
    (
        "spine2d_angle_limit",
        bpy.props.IntProperty(
            name="Angle Limit",
            description="Angle limit for cutting (1–89°)",
            default=30,
            min=1,
            max=89,
        ),
    ),
    (
        "spine2d_seam_maker_mode",
        bpy.props.EnumProperty(
            name="Seam Maker",
            description="Seam placement mode",
            items=[
                ("AUTO", "Auto", "Automatic placement"),
                ("CUSTOM", "Custom", "Use user-defined seams"),
            ],
            default="AUTO",
        ),
    ),
    (
        "spine2d_frames_for_render",
        bpy.props.IntProperty(
            name="Frames for render",
            description="0 for current frame; >0 for a sequence from playback",
            get=get_frames_for_render,
            set=set_frames_for_render,
            min=0,
        ),
    ),
    (
        "spine2d_texture_size",
        bpy.props.IntProperty(
            name="Texture size",
            description="Texture dimensions (power of 2, from 64 to 4096)",
            get=get_texture_size,
            set=set_texture_size,
        ),
    ),
    (
        "spine2d_images_path",
        bpy.props.StringProperty(
            name="Images Subfolder",
            description="Subfolder for textures, relative to the JSON path",
            default="images/",
        ),
    ),
    (
        "spine2d_json_path",
        bpy.props.StringProperty(
            name="JSON",
            description="Folder for saving the JSON file",
            default="",
            subtype="DIR_PATH",
            update=_update_ui_for_paths,
        ),
    ),
]


def register():
    logger.debug("[LOG] Registering config.py properties")
    for name, prop in PROPERTIES:
        setattr(bpy.types.Scene, name, prop)


def unregister():
    logger.debug("[LOG] Unregistering config.py properties")
    for name, _ in PROPERTIES:
        if hasattr(bpy.types.Scene, name):
            delattr(bpy.types.Scene, name)
