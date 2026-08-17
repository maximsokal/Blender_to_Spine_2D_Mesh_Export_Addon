# pylint: disable=import-error
"""Rig-only controls for the Spine2D exporter child panel."""

from __future__ import annotations

import logging
from typing import Set

import bpy

from .application.a1_shared_pivot import supports_a1_shared_pivot
from .domain.baking import A1TextureExportMode
from .domain.projection import A1ProjectionDirection, resolve_a1_projection_direction
from .domain.spine.rig_profiles import A1RigProfile


logger = logging.getLogger(__name__)


def _selected_mesh_count(context: bpy.types.Context) -> int:
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


def _shared_pivot_available(context: bpy.types.Context) -> bool:
    direction = _resolved_projection_direction(context.scene)
    if direction is None:
        return False
    return supports_a1_shared_pivot(
        A1TextureExportMode.NORMAL_UV_SEGMENTS,
        direction,
        _selected_mesh_count(context),
    )


def draw_rig_settings(
    layout: bpy.types.UILayout,
    context: bpy.types.Context,
) -> None:
    """Draw only controls that are not already owned by the main exporter panel."""

    scene = context.scene
    header = layout.row(align=True)
    header.label(text="2-Axis Rotation + Scale", icon="CON_ROTLIKE")
    header.operator("spine2d.reset_rig_profile", text="Reset")

    layout.label(text="Controls: Rotation X / Y + Scale")

    texture_mode = str(
        getattr(
            scene,
            "spine2d_texture_export_mode",
            A1TextureExportMode.NORMAL_UV_SEGMENTS.value,
        )
    ).strip().upper()
    if (
        texture_mode == A1TextureExportMode.NORMAL_UV_SEGMENTS.value
        and _shared_pivot_available(context)
    ):
        layout.prop(
            scene,
            "spine2d_shared_selection_pivot",
            text="Shared selection pivot",
        )
        if bool(getattr(scene, "spine2d_shared_selection_pivot", True)):
            layout.label(
                text="Pivot: center of selected exported Mesh geometry",
                icon="CON_PIVOT",
            )

    row = layout.row(align=True)
    row.label(text="Preview animation")
    row.prop(scene, "spine2d_export_preview_animation", text="")


class SPINE2D_OT_ResetRigProfile(bpy.types.Operator):
    """Restore public rig/projection settings to production defaults."""

    bl_idname = "spine2d.reset_rig_profile"
    bl_label = "Reset Rig Settings"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context: bpy.types.Context) -> Set[str]:
        try:
            scene = context.scene
            scene.spine2d_rig_profile = A1RigProfile.TWO_AXIS_ROTATION_SCALE.value
            scene.spine2d_projection_direction = A1ProjectionDirection.POSITIVE_Z.value
            scene.spine2d_shared_selection_pivot = True
            scene.spine2d_depth_parallax_horizon_angle = 0.0
            scene.spine2d_export_preview_animation = False
            self.report({"INFO"}, "Rig and projection settings reset")
            return {"FINISHED"}
        except Exception as exc:
            logger.exception("Unable to reset Spine2D rig settings")
            self.report({"ERROR"}, f"Rig reset error: {exc}")
            return {"CANCELLED"}


CLASSES = (SPINE2D_OT_ResetRigProfile,)


def register() -> None:
    """Register the rig reset operator."""

    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister() -> None:
    """Unregister the rig reset operator."""

    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)


__all__ = [
    "CLASSES",
    "SPINE2D_OT_ResetRigProfile",
    "draw_rig_settings",
    "register",
    "unregister",
]
