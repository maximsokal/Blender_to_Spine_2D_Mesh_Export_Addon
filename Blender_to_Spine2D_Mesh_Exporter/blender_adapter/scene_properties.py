"""Blender 5.2+ Scene RNA properties owned by the Rewrite UI boundary."""

from __future__ import annotations

from contextlib import contextmanager
import logging
from typing import Any, Iterator

import bpy

from .. import config
from ..domain.baking import A1TextureExportMode
from .scene_settings_migration import (
    CURRENT_SETTINGS_SCHEMA_VERSION,
    migration_file_loading,
)


logger = logging.getLogger(__name__)
_TEXTURE_SIZE_SYNCING = False
_SEAM_SCHEMA_UPDATE_SUSPEND_DEPTH = 0


@contextmanager
def suspend_seam_mode_schema_updates() -> Iterator[None]:
    """Temporarily prevent RNA restore callbacks from marking a Scene current.

    Blender may invoke EnumProperty update callbacks while Scene RNA properties are being
    registered over already persisted ID-property values. Those callbacks are not user
    interaction and must not advance the migration marker before the migration owner runs.
    The depth counter keeps nested registration or test scopes safe and deterministic.
    """

    global _SEAM_SCHEMA_UPDATE_SUSPEND_DEPTH
    _SEAM_SCHEMA_UPDATE_SUSPEND_DEPTH += 1
    try:
        yield
    finally:
        _SEAM_SCHEMA_UPDATE_SUSPEND_DEPTH -= 1
        if _SEAM_SCHEMA_UPDATE_SUSPEND_DEPTH < 0:
            _SEAM_SCHEMA_UPDATE_SUSPEND_DEPTH = 0
            raise RuntimeError("Seam schema update suspension depth became negative")


def seam_mode_schema_updates_suspended() -> bool:
    """Return whether Seam Maker callbacks must avoid advancing the schema marker."""

    return _SEAM_SCHEMA_UPDATE_SUSPEND_DEPTH > 0


def _extension_registration_active() -> bool:
    """Return whether the root extension is currently registering its RNA surface."""

    try:
        from .. import get_registration_state

        state = get_registration_state()
    except Exception:
        # Isolated tests and partial imports may not expose the root lifecycle owner.
        return False
    return str(getattr(state, "value", state)).upper() == "REGISTERING"


def _update_ui_for_paths(_self: Any, context: bpy.types.Context) -> None:
    """Refresh visible 3D View panels after an output path changes."""

    window_manager = getattr(context, "window_manager", None)
    for window in getattr(window_manager, "windows", ()):
        screen = getattr(window, "screen", None)
        for area in getattr(screen, "areas", ()):
            if getattr(area, "type", None) == "VIEW_3D":
                area.tag_redraw()


def _update_texture_export_mode(_self: Any, context: bpy.types.Context) -> None:
    """Invalidate diagnostics and schedule one debounced analysis for the new mode."""

    scene = getattr(context, "scene", None)
    try:
        if scene is not None:
            from .a1_export_readiness import clear_a1_export_readiness

            clear_a1_export_readiness(scene)
    except Exception:
        logger.exception("Unable to invalidate readiness after export-mode change")

    try:
        from .. import auto_readiness

        auto_readiness.request_auto_analysis(
            context,
            reason="texture export mode changed",
        )
    except Exception:
        # Registration, file loading, and test doubles may not expose the automatic
        # readiness owner yet. The cache has already been invalidated above.
        logger.debug(
            "Automatic readiness is unavailable during export-mode update",
            exc_info=True,
        )

    _update_ui_for_paths(_self, context)


def _update_seam_maker_mode(self: Any, context: bpy.types.Context) -> None:
    """Mark only a deliberate post-registration Seam Maker choice as current."""

    lifecycle_update = (
        migration_file_loading()
        or seam_mode_schema_updates_suspended()
        or _extension_registration_active()
    )
    if not lifecycle_update:
        try:
            current = int(getattr(self, "spine2d_settings_schema_version", 0) or 0)
        except (TypeError, ValueError, OverflowError):
            current = 0
        if current < CURRENT_SETTINGS_SCHEMA_VERSION:
            self.spine2d_settings_schema_version = CURRENT_SETTINGS_SCHEMA_VERSION
    _update_ui_for_paths(self, context)


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
        "spine2d_settings_schema_version",
        bpy.props.IntProperty(
            name="Spine2D Settings Schema",
            description="Internal version marker for one-time Rewrite Scene migrations",
            default=0,
            min=0,
            options={"HIDDEN"},
        ),
    ),
    (
        "spine2d_texture_export_mode",
        bpy.props.EnumProperty(
            name="Export Mode",
            description=(
                "Choose segmented UV object baking or an explicit active-camera projection"
            ),
            items=(
                (
                    A1TextureExportMode.NORMAL_UV_SEGMENTS.value,
                    "Normal — UV Segments",
                    "Preserve cut regions and bake textures onto their generated UV layout",
                ),
                (
                    A1TextureExportMode.CAMERA_PROJECTION.value,
                    "Camera Projection",
                    "Render from the active camera and export a screen-space projection mesh",
                ),
            ),
            default=A1TextureExportMode.NORMAL_UV_SEGMENTS.value,
            update=_update_texture_export_mode,
        ),
    ),
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
            update=_update_seam_maker_mode,
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
        "spine2d_projection_alpha_threshold",
        bpy.props.FloatProperty(
            name="Projection Alpha Threshold",
            description=(
                "Minimum rendered alpha included in camera-projection crop bounds"
            ),
            default=1.0 / 255.0,
            min=0.0,
            max=1.0,
            precision=6,
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


__all__ = [
    "PROPERTIES",
    "seam_mode_schema_updates_suspended",
    "suspend_seam_mode_schema_updates",
]
