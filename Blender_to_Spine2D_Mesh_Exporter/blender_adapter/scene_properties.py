"""Blender 5.2+ Scene RNA properties owned by the Rewrite UI boundary."""

from __future__ import annotations

import logging
from typing import Any

import bpy

from .. import config
from ..domain.baking import A1TextureExportMode
from ..domain.projection import A1ProjectionDirection
from ..domain.spine.rig_profiles import A1RigProfile
from ..domain.spine.version_target import (
    DEFAULT_SPINE_JSON_TARGET,
    spine_json_target_enum_items,
)
from .scene_settings_migration import (
    CURRENT_SETTINGS_SCHEMA_VERSION,
    migration_file_loading,
)


logger = logging.getLogger(__name__)
_TEXTURE_SIZE_SYNCING = False


def projection_direction_rna_enum_items() -> tuple[tuple[str, str, str], ...]:
    """Return stable persisted projection choices for Normal - UV Segments."""

    descriptions = {
        A1ProjectionDirection.POSITIVE_X: (
            "Project world +Y to Spine X, world +Z to Spine Y, and use world +X as depth"
        ),
        A1ProjectionDirection.NEGATIVE_X: (
            "Project world -Y to Spine X, world +Z to Spine Y, and use world -X as depth"
        ),
        A1ProjectionDirection.POSITIVE_Y: (
            "Project world -X to Spine X, world +Z to Spine Y, and use world +Y as depth"
        ),
        A1ProjectionDirection.NEGATIVE_Y: (
            "Project world +X to Spine X, world +Z to Spine Y, and use world -Y as depth"
        ),
        A1ProjectionDirection.POSITIVE_Z: (
            "Project world +X to Spine X, world +Y to Spine Y, and use world +Z as depth"
        ),
        A1ProjectionDirection.NEGATIVE_Z: (
            "Project world -X to Spine X, world +Y to Spine Y, and use world -Z as depth"
        ),
        A1ProjectionDirection.ACTIVE_CAMERA: (
            "Project evaluated geometry through the active Perspective or Orthographic "
            "camera using the selected export texture dimensions"
        ),
    }
    return tuple(
        (direction.value, direction.label, descriptions[direction])
        for direction in A1ProjectionDirection
    )


def rig_profile_rna_enum_items() -> tuple[tuple[str, str, str], ...]:
    """Return persisted RNA choices required to load historical `.blend` files.

    Three Axis remains an internal compatibility value so schema migration can bind old
    Scene ID-properties safely. The public panel deliberately does not draw this Enum as
    a selector; public export capture normalizes every hidden value to Two Axis.
    """

    return tuple(
        (profile.value, profile.label, profile.description)
        for profile in A1RigProfile
    )


def _extension_registration_active() -> bool:
    """Return whether the root extension is currently registering its RNA surface.

    Blender may invoke EnumProperty update callbacks when persisted ID-property values are
    rebound to newly registered RNA. That lifecycle callback is not a deliberate user edit
    and must not advance the Scene schema before the migration owner runs.
    """

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


def _invalidate_readiness_for_setting(
    context: bpy.types.Context,
    *,
    reason: str,
) -> None:
    """Invalidate cached analysis; refresh is deliberately manual."""

    scene = getattr(context, "scene", None)
    try:
        if scene is not None:
            from .a1_export_readiness import clear_a1_export_readiness

            clear_a1_export_readiness(scene)
    except Exception:
        logger.exception("Unable to invalidate readiness after %s", reason)

    # Do not schedule background analysis here. A settings edit must never start
    # production work or make the UI busy; the user explicitly presses Analyze.


def _update_texture_export_mode(_self: Any, context: bpy.types.Context) -> None:
    """Invalidate diagnostics after changing the texture export mode."""

    _invalidate_readiness_for_setting(
        context,
        reason="texture export mode changed",
    )
    _update_ui_for_paths(_self, context)


def _update_projection_direction(_self: Any, context: bpy.types.Context) -> None:
    """Invalidate diagnostics after changing the Normal/UV projection frame."""

    _invalidate_readiness_for_setting(
        context,
        reason="projection direction changed",
    )
    _update_ui_for_paths(_self, context)


def _update_spine_target_version(_self: Any, context: bpy.types.Context) -> None:
    """Invalidate diagnostics because the final JSON schema has changed."""

    _invalidate_readiness_for_setting(
        context,
        reason="Spine target version changed",
    )
    _update_ui_for_paths(_self, context)


def _update_rig_profile(_self: Any, context: bpy.types.Context) -> None:
    """Invalidate diagnostics because rig bones, constraints, and weights changed."""

    _invalidate_readiness_for_setting(
        context,
        reason="rig profile changed",
    )
    _update_ui_for_paths(_self, context)


def _update_seam_maker_mode(self: Any, context: bpy.types.Context) -> None:
    """Mark only a deliberate post-registration Seam Maker choice as current."""

    lifecycle_update = migration_file_loading() or _extension_registration_active()
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
        "spine2d_projection_direction",
        bpy.props.EnumProperty(
            name="Projection Direction",
            description=(
                "Choose the world axis or active camera used by Normal - UV Segments"
            ),
            items=projection_direction_rna_enum_items(),
            default=A1ProjectionDirection.POSITIVE_Z.value,
            update=_update_projection_direction,
        ),
    ),
    (
        "spine2d_target_spine_version",
        bpy.props.EnumProperty(
            name="Spine Version",
            description="Choose the target Spine JSON schema and exact version metadata",
            items=spine_json_target_enum_items(),
            default=DEFAULT_SPINE_JSON_TARGET.value,
            update=_update_spine_target_version,
        ),
    ),
    (
        "spine2d_rig_profile",
        bpy.props.EnumProperty(
            name="Rig Profile",
            description=(
                "Persisted rig profile compatibility value; public UI exports Two Axis"
            ),
            items=rig_profile_rna_enum_items(),
            default=A1RigProfile.TWO_AXIS_ROTATION_SCALE.value,
            update=_update_rig_profile,
            options={"HIDDEN"},
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
    "projection_direction_rna_enum_items",
    "rig_profile_rna_enum_items",
]
