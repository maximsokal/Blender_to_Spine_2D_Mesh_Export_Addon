# pylint: disable=import-error
"""Dedicated Blender UI category for selectable Spine rig profiles."""

from __future__ import annotations

import logging
from typing import Set

import bpy

from .domain.spine.rig_profiles import A1RigProfile, resolve_a1_rig_profile
from .infrastructure.blender_registration import (
    class_cleanup_actions,
    register_classes_transactionally,
    unregister_all_best_effort,
)


logger = logging.getLogger(__name__)


class SPINE2D_OT_ResetRigProfile(bpy.types.Operator):
    """Restore the compatibility three-axis profile without changing other settings."""

    bl_idname = "spine2d.reset_rig_profile"
    bl_label = "Reset Rig Profile"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context: bpy.types.Context) -> Set[str]:
        try:
            context.scene.spine2d_rig_profile = (
                A1RigProfile.THREE_AXIS_ROTATION.value
            )
            self.report({"INFO"}, "Rig profile reset to 3-Axis Rotation")
            return {"FINISHED"}
        except Exception as exc:
            logger.exception("Unable to reset Spine2D rig profile")
            self.report({"ERROR"}, f"Rig reset error: {exc}")
            return {"CANCELLED"}


class OBJECT_PT_Spine2DRigPanel(bpy.types.Panel):
    """Child panel that keeps rig generation independent from texture settings."""

    bl_label = "Rig"
    bl_idname = "OBJECT_PT_spine2d_rig"
    bl_parent_id = "OBJECT_PT_spine2d_mesh"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Blender to Spine2D Mesh Exporter"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context: bpy.types.Context) -> None:
        layout = self.layout
        scene = context.scene
        try:
            header = layout.row(align=True)
            header.prop(scene, "spine2d_rig_profile", text="Rig profile")
            header.operator(
                "spine2d.reset_rig_profile",
                text="",
                icon="LOOP_BACK",
            )
            profile = resolve_a1_rig_profile(
                getattr(
                    scene,
                    "spine2d_rig_profile",
                    A1RigProfile.THREE_AXIS_ROTATION.value,
                )
            )
            description = layout.box()
            if profile is A1RigProfile.THREE_AXIS_ROTATION:
                description.label(
                    text="Controls: Rotation X / Y / Z",
                    icon="ORIENTATION_GIMBAL",
                )
                description.label(
                    text="Current compatibility rig; scale is constrained"
                )
            else:
                description.label(
                    text="Controls: Rotation X / Y + Scale",
                    icon="FULLSCREEN_ENTER",
                )
                description.label(text="No Rotation Z control is generated")
                description.label(
                    text="Scale affects X owner and all depth planes"
                )

            layout.separator()
            row = layout.row(align=True)
            row.label(text="Control icons")
            row.prop(scene, "spine2d_control_icons", text="")
            row = layout.row(align=True)
            row.label(text="Preview animation")
            row.prop(scene, "spine2d_export_preview_animation", text="")
        except Exception:
            logger.exception("Unable to draw Spine2D Rig panel")
            layout.label(text="Rig UI error (see console)", icon="ERROR")


CLASSES = (
    SPINE2D_OT_ResetRigProfile,
    OBJECT_PT_Spine2DRigPanel,
)


def register() -> None:
    """Register the dedicated rig panel transactionally."""

    register_classes_transactionally(
        CLASSES,
        register_class=bpy.utils.register_class,
        unregister_class=bpy.utils.unregister_class,
    )
    logger.debug("Spine2D Rig UI registered")


def unregister() -> None:
    """Unregister the dedicated rig panel best-effort."""

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
    "OBJECT_PT_Spine2DRigPanel",
    "SPINE2D_OT_ResetRigProfile",
    "register",
    "unregister",
]
