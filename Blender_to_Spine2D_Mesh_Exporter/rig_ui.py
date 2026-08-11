# pylint: disable=import-error
"""Rig-profile and projection controls shared by the ordered Rewrite UI."""

from __future__ import annotations

import logging
from typing import Set

import bpy

from . import ui
from .application.a1_shared_pivot import supports_a1_shared_pivot
from .domain.baking import A1TextureExportMode
from .domain.projection import (
    A1ProjectionDirection,
    resolve_a1_projection_direction,
)
from .domain.spine.rig_profiles import A1RigProfile
from .infrastructure.blender_registration import (
    class_cleanup_actions,
    register_classes_transactionally,
    unregister_all_best_effort,
)


logger = logging.getLogger(__name__)
_ORIGINAL_RESET_REMOVED = False
_REGISTERED_CLASSES: tuple[type, ...] = ()


def _selected_mesh_count(context: bpy.types.Context) -> int:
    """Count the Mesh objects that the public selected-export route can consume."""

    return sum(
        1
        for candidate in getattr(context, "selected_objects", ())
        if getattr(candidate, "type", None) == "MESH"
        and getattr(candidate, "data", None) is not None
    )


def _resolved_projection_direction(
    scene: bpy.types.Scene,
) -> A1ProjectionDirection | None:
    try:
        return resolve_a1_projection_direction(
            getattr(
                scene,
                "spine2d_projection_direction",
                A1ProjectionDirection.POSITIVE_Z.value,
            )
        )
    except (TypeError, ValueError):
        return None


def _shared_pivot_available(
    context: bpy.types.Context,
    texture_mode: A1TextureExportMode,
) -> bool:
    """Mirror the pure exporter capability contract for conditional UI visibility."""

    direction = _resolved_projection_direction(context.scene)
    if direction is None:
        return False
    return supports_a1_shared_pivot(
        texture_mode,
        direction,
        _selected_mesh_count(context),
    )


def _draw_shared_selection_pivot(
    layout: bpy.types.UILayout,
    context: bpy.types.Context,
) -> None:
    """Draw the default-on toggle only for multi-object signed-axis Normal export."""

    if not _shared_pivot_available(
        context,
        A1TextureExportMode.NORMAL_UV_SEGMENTS,
    ):
        return

    scene = context.scene
    layout.prop(
        scene,
        "spine2d_shared_selection_pivot",
        text="Shared selection pivot",
    )
    if bool(getattr(scene, "spine2d_shared_selection_pivot", True)):
        layout.label(
            text="Pivot: center of all selected exported Mesh geometry",
            icon="CON_PIVOT",
        )


def _draw_projection_direction(
    layout: bpy.types.UILayout,
    scene: bpy.types.Scene,
) -> None:
    """Draw the public Normal/UV projection selector and route explanation."""

    layout.prop(
        scene,
        "spine2d_projection_direction",
        text="Projection direction",
    )
    direction = _resolved_projection_direction(scene)
    if direction is None:
        layout.label(
            text="Invalid projection direction; Reset restores +Z",
            icon="ERROR",
        )
        return

    if direction.active_camera:
        layout.label(
            text="Projects UV-segment geometry through the active camera",
            icon="CAMERA_DATA",
        )
        layout.label(
            text="Perspective and Orthographic cameras are supported",
            icon="INFO",
        )
        if direction.camera_root:
            layout.label(
                text="Main bone pivot: active camera / camera-space zero",
                icon="CON_PIVOT",
            )
            layout.label(
                text="Projected Object Origin is stored inside one rigid depth layer",
                icon="BONE_DATA",
            )
        else:
            layout.label(
                text="Main bone pivot: each object's Blender Object Origin",
                icon="OBJECT_ORIGIN",
            )
            layout.label(
                text="Per-vertex camera depth remains available to the rig",
                icon="BONE_DATA",
            )
        return

    layout.label(
        text=f"World-axis object-bake projection: {direction.label}",
        icon="ORIENTATION_GLOBAL",
    )


def _draw_forced_active_camera_projection(
    layout: bpy.types.UILayout,
) -> None:
    """Show the camera-only projection contract without exposing an invalid selector."""

    row = layout.row(align=True)
    row.label(text="Projection")
    row.label(text="Active Camera", icon="CAMERA_DATA")
    layout.label(
        text="Perspective and Orthographic cameras are supported",
        icon="INFO",
    )


def draw_rig_settings(
    layout: bpy.types.UILayout,
    context: bpy.types.Context,
) -> None:
    """Draw the single public rig and the approved projection controls."""

    scene = context.scene
    layout.label(text="Texture export", icon="TEXTURE")
    layout.prop(scene, "spine2d_texture_export_mode", text="Export mode")
    texture_mode = str(
        getattr(
            scene,
            "spine2d_texture_export_mode",
            A1TextureExportMode.NORMAL_UV_SEGMENTS.value,
        )
    ).upper()
    if texture_mode == A1TextureExportMode.CAMERA_PROJECTION.value:
        layout.label(
            text="Active camera render -> one screen-space mesh",
            icon="CAMERA_DATA",
        )
        _draw_forced_active_camera_projection(layout)
        layout.prop(
            scene,
            "spine2d_projection_alpha_threshold",
            text="Projection alpha threshold",
        )
    elif texture_mode == A1TextureExportMode.DEPTH_CAMERA_PROJECTION.value:
        layout.label(
            text="Active camera render -> optimized depth-relief mesh",
            icon="CAMERA_DATA",
        )
        _draw_forced_active_camera_projection(layout)
    else:
        layout.label(
            text="Preserves cut regions and generated UV meshes",
            icon="UV",
        )
        _draw_projection_direction(layout, scene)
        _draw_shared_selection_pivot(layout, context)

    # Keep the historical RNA value loadable, but do not expose a profile selector
    # until Three Axis receives the same Object Origin implementation and validation.
    row = layout.row(align=True)
    row.label(text="Rig profile")
    row.label(text="2-Axis Rotation + Scale", icon="CON_ROTLIKE")

    description = layout.box()
    description.label(
        text="Controls: Rotation X / Y + Scale",
        icon="FULLSCREEN_ENTER",
    )
    if texture_mode == A1TextureExportMode.CAMERA_PROJECTION.value:
        description.label(text="Camera Projection keeps compatibility placement")
    elif texture_mode == A1TextureExportMode.DEPTH_CAMERA_PROJECTION.value:
        description.label(text="Depth geometry and rig depth use the active camera")
        description.label(text="Main bone matches projected Blender Object Origin")
    else:
        direction = _resolved_projection_direction(scene)
        if direction is None:
            direction = A1ProjectionDirection.POSITIVE_Z
        if direction.camera_root:
            description.label(text="Main bone uses active-camera space as its pivot")
            description.label(text="Object placement is stored below one rigid layer")
        elif (
            _shared_pivot_available(
                context,
                A1TextureExportMode.NORMAL_UV_SEGMENTS,
            )
            and bool(getattr(scene, "spine2d_shared_selection_pivot", True))
        ):
            description.label(text="All selected parts share one geometry-center pivot")
            description.label(text="Blender Object origins remain unchanged")
        else:
            description.label(text="Main bone matches projected Blender Object Origin")
            description.label(text="Depth uses the selected axis or active camera")

    layout.separator()
    row = layout.row(align=True)
    row.label(text="Control icons")
    row.prop(scene, "spine2d_control_icons", text="")
    row = layout.row(align=True)
    row.label(text="Preview animation")
    row.prop(scene, "spine2d_export_preview_animation", text="")


class SPINE2D_OT_ResetSettingsWithProjection(bpy.types.Operator):
    """Extend the main Reset operator with current projection and parallax defaults."""

    bl_idname = ui.SPINE2D_OT_ResetSettings.bl_idname
    bl_label = ui.SPINE2D_OT_ResetSettings.bl_label
    bl_description = ui.SPINE2D_OT_ResetSettings.__doc__ or "Reset export settings"
    bl_options = ui.SPINE2D_OT_ResetSettings.bl_options

    def execute(self, context: bpy.types.Context) -> Set[str]:
        result = ui.SPINE2D_OT_ResetSettings.execute(self, context)
        if result != {"FINISHED"}:
            return result
        try:
            context.scene.spine2d_projection_direction = (
                A1ProjectionDirection.POSITIVE_Z.value
            )
            context.scene.spine2d_shared_selection_pivot = True
            context.scene.spine2d_depth_parallax_horizon_angle = 0.0
            return result
        except Exception as exc:
            logger.exception(
                "Unable to reset Spine2D projection, shared pivot, and parallax settings"
            )
            self.report({"ERROR"}, f"Projection/shared-pivot reset error: {exc}")
            return {"CANCELLED"}


class SPINE2D_OT_ResetRigProfile(bpy.types.Operator):
    """Restore the production two-axis profile without changing other settings."""

    bl_idname = "spine2d.reset_rig_profile"
    bl_label = "Reset Rig Profile"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context: bpy.types.Context) -> Set[str]:
        try:
            context.scene.spine2d_rig_profile = (
                A1RigProfile.TWO_AXIS_ROTATION_SCALE.value
            )
            self.report({"INFO"}, "Rig profile reset to 2-Axis Rotation + Scale")
            return {"FINISHED"}
        except Exception as exc:
            logger.exception("Unable to reset Spine2D rig profile")
            self.report({"ERROR"}, f"Rig reset error: {exc}")
            return {"CANCELLED"}


CLASSES = (
    SPINE2D_OT_ResetSettingsWithProjection,
    SPINE2D_OT_ResetRigProfile,
)


def _restore_original_reset_operator() -> None:
    """Restore the base Reset class after removing our replacement."""

    global _ORIGINAL_RESET_REMOVED
    if not _ORIGINAL_RESET_REMOVED:
        return
    bpy.utils.register_class(ui.SPINE2D_OT_ResetSettings)
    _ORIGINAL_RESET_REMOVED = False


def register() -> None:
    """Replace Reset transactionally and register the Rig UI operators."""

    global _ORIGINAL_RESET_REMOVED, _REGISTERED_CLASSES
    if _REGISTERED_CLASSES:
        return

    original_registered = hasattr(ui.SPINE2D_OT_ResetSettings, "bl_rna")
    if original_registered:
        bpy.utils.unregister_class(ui.SPINE2D_OT_ResetSettings)
        _ORIGINAL_RESET_REMOVED = True

    try:
        _REGISTERED_CLASSES = register_classes_transactionally(
            CLASSES,
            register_class=bpy.utils.register_class,
            unregister_class=bpy.utils.unregister_class,
        )
    except Exception as exc:
        logger.exception("Spine2D Rig UI registration failed")
        try:
            _restore_original_reset_operator()
        except Exception:
            logger.exception("Unable to restore base Reset operator during rollback")
        raise RuntimeError("Spine2D Rig UI registration failed") from exc
    logger.debug("Spine2D Rig UI registered")


def unregister() -> None:
    """Unregister Rig UI classes and restore the base Reset operator."""

    global _REGISTERED_CLASSES
    errors: list[BaseException] = []
    if _REGISTERED_CLASSES:
        try:
            unregister_all_best_effort(
                class_cleanup_actions(
                    _REGISTERED_CLASSES,
                    unregister_class=bpy.utils.unregister_class,
                ),
                operation="Spine2D Rig UI unregistration",
            )
        except Exception as exc:
            logger.exception("Unable to unregister Spine2D Rig UI classes")
            errors.append(exc)
        finally:
            _REGISTERED_CLASSES = ()

    try:
        _restore_original_reset_operator()
    except Exception as exc:
        logger.exception("Unable to restore the base Spine2D Reset operator")
        errors.append(exc)

    if errors:
        raise RuntimeError("Spine2D Rig UI unregistration failed") from errors[0]
    logger.debug("Spine2D Rig UI unregistered")


__all__ = [
    "CLASSES",
    "SPINE2D_OT_ResetRigProfile",
    "SPINE2D_OT_ResetSettingsWithProjection",
    "draw_rig_settings",
    "register",
    "unregister",
]
