# pylint: disable=import-error
"""Rig-profile and projection controls shared by the ordered Rewrite UI."""

from __future__ import annotations

import logging
from typing import Set

import bpy

from . import ui
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
    try:
        direction = resolve_a1_projection_direction(
            getattr(
                scene,
                "spine2d_projection_direction",
                A1ProjectionDirection.POSITIVE_Z.value,
            )
        )
    except (TypeError, ValueError):
        layout.label(
            text="Invalid projection direction; Reset restores +Z",
            icon="ERROR",
        )
        return

    if direction is A1ProjectionDirection.ACTIVE_CAMERA:
        layout.label(
            text="Projects UV-segment geometry through the active camera",
            icon="CAMERA_DATA",
        )
        layout.label(
            text="Perspective and Orthographic cameras are supported",
            icon="INFO",
        )
        return

    layout.label(
        text=f"World-axis object-bake projection: {direction.label}",
        icon="ORIENTATION_GLOBAL",
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
        layout.prop(
            scene,
            "spine2d_projection_alpha_threshold",
            text="Projection alpha threshold",
        )
    else:
        layout.label(
            text="Preserves cut regions and generated UV meshes",
            icon="UV",
        )
        _draw_projection_direction(layout, scene)

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
    """Extend the main Reset operator with the Slice 6 projection default."""

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
            return result
        except Exception as exc:
            logger.exception("Unable to reset Spine2D projection direction")
            self.report({"ERROR"}, f"Projection reset error: {exc}")
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
