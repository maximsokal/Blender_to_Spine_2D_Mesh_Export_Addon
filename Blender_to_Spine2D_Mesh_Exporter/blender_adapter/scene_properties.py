"""Blender 5.2+ Scene RNA properties owned by the Rewrite UI boundary."""

from __future__ import annotations

import logging
from typing import Any

import bpy

from .. import config


logger = logging.getLogger(__name__)
_TEXTURE_SIZE_SYNCING = False


def _update_ui_for_paths(_self: Any, context: bpy.types.Context) -> None:
    """Refresh visible 3D View panels after an output path changes."""

    window_manager = getattr(context, "window_manager", None)
    for window in getattr(window_manager, "windows", ()):
        screen = getattr(window, "screen", None)
        for area in getattr(screen, "areas", ()):
            if getattr(area, "type", None) == "VIEW_3D":
                area.tag_redraw()


def _update_texture_size(self: Any, _context: bpy.types.Context) -> None:
    """Keep the persisted RNA value even and synchronize transitional globals."""

    global _TEXTURE_SIZE_SYNCING
    if _TEXTURE_SIZE_SYNCING:
        return
    try:
        value = int(getattr(self, "spine2d_texture_size", 1024))
        resolved = min(4096, max(64, value))
        if resolved % 2:
            resolved -= 1
        if resolved != value:
            _TEXTURE_SIZE_SYNCING = True
            try:
                setattr(self, "spine2d_texture_size", resolved)
            finally:
                _TEXTURE_SIZE_SYNCING = False
        config.TEXTURE_WIDTH = resolved
        config.TEXTURE_HEIGHT = resolved
        logger.debug("Texture size synchronized to %s", resolved)
    except Exception:
        logger.exception("Unable to synchronize Blender 5.2 Scene texture size")
        raise


PROPERTIES = (
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
            items=(
                ("AUTO", "Auto", "Automatic placement"),
                ("CUSTOM", "Custom", "Use user-defined seams"),
            ),
            default="AUTO",
        ),
    ),
    (
        "spine2d_frames_for_render",
        bpy.props.IntProperty(
            name="Frames for render",
            description="0 for current frame; >0 for a sequence from playback",
            default=0,
            min=0,
        ),
    ),
    (
        "spine2d_texture_size",
        bpy.props.IntProperty(
            name="Texture size",
            description="Even texture dimensions from 64 to 4096",
            default=1024,
            min=64,
            max=4096,
            step=2,
            update=_update_texture_size,
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
)


__all__ = ["PROPERTIES"]
