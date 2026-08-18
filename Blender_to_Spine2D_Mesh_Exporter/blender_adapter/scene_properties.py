"""Blender 5.2+ Scene RNA properties owned by the Rewrite UI boundary."""

from __future__ import annotations

import logging
from math import radians
from typing import Any

import bpy

from .. import config
from ..domain.baking import (
    A1TextureExportMode,
    DepthProjectionBaseMode,
)
from ..domain.projection import A1ProjectionDirection
from ..domain.spine.rig_profiles import A1RigProfile
from ..domain.spine.version_target import (
    DEFAULT_SPINE_JSON_TARGET,
    spine_json_target_enum_items,
)
from .scene_settings_migration import (
    CURRENT_SETTINGS_SCHEMA_VERSION,
    migration_file_loading,
    migration_registration_pending,
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
            "Project through the active camera while keeping each exported object's "
            "Blender Object Origin as its own Spine main-bone pivot"
        ),
        A1ProjectionDirection.ACTIVE_CAMERA_CAMERA_ROOT: (
            "Project through the active camera and use camera-space zero as the Spine "
            "main-bone pivot, with the projected Object Origin stored inside the rig"
        ),
    }
    return tuple(
        (direction.value, direction.label, descriptions[direction])
        for direction in A1ProjectionDirection
    )


def rig_profile_rna_enum_items() -> tuple[tuple[str, str, str], ...]:
    """Return persisted RNA choices required to load historical `.blend` files."""

    return tuple(
        (profile.value, profile.label, profile.description)
        for profile in A1RigProfile
    )


def _update_ui_for_paths(_self: Any, context: bpy.types.Context) -> None:
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
    scene = getattr(context, "scene", None)
    try:
        if scene is not None:
            from .a1_export_readiness import clear_a1_export_readiness

            clear_a1_export_readiness(scene)
    except Exception:
        logger.exception("Unable to invalidate readiness after %s", reason)


def _update_texture_export_mode(_self: Any, context: bpy.types.Context) -> None:
    _invalidate_readiness_for_setting(
        context,
        reason="texture export mode changed",
    )
    _update_ui_for_paths(_self, context)


def _update_projection_direction(_self: Any, context: bpy.types.Context) -> None:
    _invalidate_readiness_for_setting(
        context,
        reason="projection direction changed",
    )
    _update_ui_for_paths(_self, context)


def _update_shared_selection_pivot(_self: Any, context: bpy.types.Context) -> None:
    _invalidate_readiness_for_setting(
        context,
        reason="shared selection pivot changed",
    )
    _update_ui_for_paths(_self, context)


def _update_bake_settings(_self: Any, context: bpy.types.Context) -> None:
    _invalidate_readiness_for_setting(
        context,
        reason="Bake settings changed",
    )
    _update_ui_for_paths(_self, context)


def _update_spine_target_version(_self: Any, context: bpy.types.Context) -> None:
    _invalidate_readiness_for_setting(
        context,
        reason="Spine target version changed",
    )
    _update_ui_for_paths(_self, context)


def _update_rig_profile(_self: Any, context: bpy.types.Context) -> None:
    _invalidate_readiness_for_setting(
        context,
        reason="rig profile changed",
    )
    _update_ui_for_paths(_self, context)


def _update_seam_maker_mode(self: Any, context: bpy.types.Context) -> None:
    """Advance schema only for an actual user-side seam setting change.

    During initial RNA binding the migration owner keeps a raw snapshot for each
    already-open Scene. During .blend loading it exposes a separate file-loading flag.
    Either condition means this callback belongs to lifecycle restoration rather than a
    new user edit, so the stored schema version must not be advanced prematurely.
    """

    lifecycle_update = (
        migration_file_loading()
        or migration_registration_pending(self)
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
            description="Choose the public mesh and texture representation",
            items=(
                (
                    A1TextureExportMode.NORMAL_UV_SEGMENTS.value,
                    "Normal / UV Segments",
                    "Preserve cut regions and bake textures onto their generated UV layout",
                ),
                (
                    A1TextureExportMode.CAMERA_PROJECTION.value,
                    "Camera Projection",
                    "Render from the active camera and export a flat screen-space mesh",
                ),
                (
                    A1TextureExportMode.DEPTH_CAMERA_PROJECTION.value,
                    "Depth Camera Projection",
                    "Render from the active camera and build an optimized depth-relief "
                    "mesh with generated vertex bones",
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
                "Choose the world axis or active-camera rig root used by "
                "Normal / UV Segments"
            ),
            items=projection_direction_rna_enum_items(),
            default=A1ProjectionDirection.POSITIVE_Z.value,
            update=_update_projection_direction,
        ),
    ),
    (
        "spine2d_shared_selection_pivot",
        bpy.props.BoolProperty(
            name="Shared Selection Pivot",
            description=(
                "Use the center of all selected exported Mesh geometry as one common "
                "Spine rotation pivot for signed-axis Normal / UV multi-object export"
            ),
            default=True,
            update=_update_shared_selection_pivot,
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
            description="0 for current frame; >0 for a texture sequence",
            default=0,
            min=0,
            update=_update_bake_settings,
        ),
    ),
    (
        "spine2d_sequence_fps_override",
        bpy.props.FloatProperty(
            name="Sequence FPS Override",
            description="0 uses Scene FPS; a positive value overrides sequence playback FPS",
            default=0.0,
            min=0.0,
            max=1000.0,
            precision=3,
            update=_update_bake_settings,
        ),
    ),
    (
        "spine2d_include_scene_shadows",
        bpy.props.BoolProperty(
            name="Include shadows from scene objects",
            description=(
                "Allow non-exported scene objects to cast shadows into rendered camera modes"
            ),
            default=True,
            update=_update_bake_settings,
        ),
    ),
    (
        "spine2d_include_scene_reflection_transmission",
        bpy.props.BoolProperty(
            name="Include reflection/transmission objects",
            description=(
                "Allow non-exported scene objects to appear in reflection and transmission rays"
            ),
            default=True,
            update=_update_bake_settings,
        ),
    ),
    (
        "spine2d_world_affects_lighting_reflections",
        bpy.props.BoolProperty(
            name="World affects lighting/reflections",
            description=(
                "Keep the Scene World active for lighting, reflections, and transmission"
            ),
            default=True,
            update=_update_bake_settings,
        ),
    ),
    (
        "spine2d_projection_alpha_threshold",
        bpy.props.FloatProperty(
            name="Projection Alpha Threshold",
            description="Minimum rendered alpha included in camera-projection crop bounds",
            default=1.0 / 255.0,
            min=0.0,
            max=1.0,
            precision=6,
            update=_update_bake_settings,
        ),
    ),
    (
        "spine2d_depth_smoothing",
        bpy.props.FloatProperty(
            name="Depth Smoothing",
            description=(
                "Blend neighboring depth samples without crossing protected depth edges"
            ),
            default=0.35,
            min=0.0,
            max=1.0,
            precision=3,
            update=_update_bake_settings,
        ),
    ),
    (
        "spine2d_depth_edge_threshold",
        bpy.props.FloatProperty(
            name="Depth Edge Threshold",
            description=(
                "Maximum neighboring depth jump as a fraction of the visible depth range"
            ),
            default=0.08,
            min=0.0,
            max=1.0,
            precision=3,
            update=_update_bake_settings,
        ),
    ),
    (
        "spine2d_depth_mesh_error_pixels",
        bpy.props.FloatProperty(
            name="Depth Mesh Error",
            description=(
                "Requested screen-space sampling distance in pixels; lower values retain "
                "more depth points"
            ),
            default=4.0,
            min=0.25,
            max=128.0,
            precision=2,
            update=_update_bake_settings,
        ),
    ),
    (
        "spine2d_depth_max_points",
        bpy.props.IntProperty(
            name="Max Depth Points",
            description=(
                "Hard limit for front and parallax-reserve depth points and vertex bones"
            ),
            default=128,
            min=4,
            max=4096,
            update=_update_bake_settings,
        ),
    ),
    (
        "spine2d_depth_parallax_horizon_angle",
        bpy.props.FloatProperty(
            name="Parallax Horizon Angle",
            description=(
                "Accumulated surface angle retained beyond the active-camera horizon. "
                "Zero exports only the current front surface; positive values add fitted "
                "virtual-view textures and reserve attachments for X/Y parallax"
            ),
            default=0.0,
            min=0.0,
            max=radians(89.0),
            soft_max=radians(45.0),
            subtype="ANGLE",
            unit="ROTATION",
            precision=2,
            update=_update_bake_settings,
        ),
    ),
    (
        "spine2d_depth_base_mode",
        bpy.props.EnumProperty(
            name="Depth Base",
            description="Internal depth-relief base policy",
            items=(
                (
                    DepthProjectionBaseMode.FARTHEST_VISIBLE.value,
                    "Farthest Visible Point",
                    "Use the farthest visible depth point as the relief base",
                ),
                (
                    DepthProjectionBaseMode.OBJECT_ORIGIN.value,
                    "Object Origin",
                    "Use Object Origin only when it is behind every visible point",
                ),
            ),
            default=DepthProjectionBaseMode.FARTHEST_VISIBLE.value,
            update=_update_bake_settings,
            options={"HIDDEN"},
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
