# pylint: disable=import-error
"""Rig-profile controls shared by the ordered Rewrite UI."""

from __future__ import annotations

import logging
from typing import Set

import bpy

from .domain.spine.rig_profiles import A1RigProfile, resolve_a1_rig_profile
from .domain.baking import A1TextureExportMode
from .infrastructure.blender_registration import (
    class_cleanup_actions,
    register_classes_transactionally,
    unregister_all_best_effort,
)


logger = logging.getLogger(__name__)


def draw_rig_settings(
    layout: bpy.types.UILayout,
    context: bpy.types.Context,
) -> None:
    """Draw rig settings inside the main panel's standard foldout container."""

    scene = context.scene
    layout.label(text="Texture export", icon="TEXTURE")
    layout.prop(scene, "spine2d_texture_export_mode", text="Export mode")
    texture_mode = str(getattr(scene, "spine2d_texture_export_mode", A1TextureExportMode.NORMAL_UV_SEGMENTS.value)).upper()
    if texture_mode == A1TextureExportMode.CAMERA_PROJECTION.value:
        layout.label(text="Active camera render -> one screen-space mesh", icon="CAMERA_DATA")
        layout.prop(scene, "spine2d_projection_alpha_threshold", text="Projection alpha threshold")
    else:
        layout.label(text="Preserves cut regions and generated UV meshes", icon="UV")
    row = layout.row(align=True)
    row.prop(scene, "spine2d_rig_profile", text="Rig profile")
    row.operator(
        "spine2d.reset_rig_profile",
        text="",
        icon="LOOP_BACK",
    )
    profile = resolve_a1_rig_profile(
        getattr(
            scene,
            "spine2d_rig_profile",
            A1RigProfile.TWO_AXIS_ROTATION_SCALE.value,
        )
    )
    description = layout.box()
    if profile is A1RigProfile.THREE_AXIS_ROTATION:
        description.label(
            text="Controls: Rotation X / Y / Z",
            icon="ORIENTATION_GIMBAL",
        )
        description.label(text="Compatibility rig; scale is constrained")
    else:
        description.label(
            text="Controls: Rotation X / Y + Scale",
            icon="FULLSCREEN_ENTER",
        )
        description.label(text="Single export uses a neutral setup pose")
        description.label(text="Multi export preserves scene placement")

    layout.separator()
    row = layout.row(align=True)
    row.label(text="Control icons")
    row.prop(scene, "spine2d_control_icons", text="")
    row = layout.row(align=True)
    row.label(text="Preview animation")
    row.prop(scene, "spine2d_export_preview_animation", text="")


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


CLASSES = (SPINE2D_OT_ResetRigProfile,)


def register() -> None:
    """Register the rig reset operator transactionally."""

    register_classes_transactionally(
        CLASSES,
        register_class=bpy.utils.register_class,
        unregister_class=bpy.utils.unregister_class,
    )
    logger.debug("Spine2D Rig UI registered")


def unregister() -> None:
    """Unregister the rig reset operator best-effort."""

    unregister_all_best_effort(
        class_cleanup_actions(
            CLASSES,
            unregister_class=bpy.utils.unregister_class,
        ),
        operation="Spine2D Rig UI unregistration",
    )
    logger.debug("Spine2D Rig UI unregistered")


__all__ = [
    "CLASSES",
    "SPINE2D_OT_ResetRigProfile",
    "draw_rig_settings",
    "register",
    "unregister",
]
