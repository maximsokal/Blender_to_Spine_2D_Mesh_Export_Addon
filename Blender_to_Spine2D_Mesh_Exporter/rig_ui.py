# pylint: disable=import-error
"""Rig-profile and projection controls for the Spine2D exporter UI."""

from __future__ import annotations

import logging
from typing import Set

import bpy

from .application.a1_shared_pivot import supports_a1_shared_pivot
from .domain.baking import A1TextureExportMode
from .domain.projection import (
    A1ProjectionDirection,
    resolve_a1_projection_direction,
)
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


def _shared_pivot_available(
    context: bpy.types.Context,
    texture_mode: A1TextureExportMode,
) -> bool:
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
                text="Main bone pivot: each object's Object Origin",
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


def _draw_forced_active_camera_projection(layout: bpy.types.UILayout) -> None:
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
    """Draw the production two-axis rig and projection controls."""

    scene = context.scene
    header = layout.row(align=True)
    header.label(text="2-Axis Rotation + Scale", icon="CON_ROTLIKE")
    header.operator("spine2d.reset_rig_profile", text="Reset")

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

    description = layout.box()
    description.label(
        text="Controls: Rotation X / Y + Scale",
        icon="FULLSCREEN_ENTER",
    )
    if texture_mode == A1TextureExportMode.CAMERA_PROJECTION.value:
        description.label(text="Camera Projection keeps compatibility placement")
    elif texture_mode == A1TextureExportMode.DEPTH_CAMERA_PROJECTION.value:
        description.label(text="Depth geometry and rig depth use the active camera")
        description.label(text="Main bone matches projected Object Origin")
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
            description.label(text="Object origins remain unchanged")
        else:
            description.label(text="Main bone matches projected Object Origin")
            description.label(text="Depth uses the selected axis or active camera")

    layout.separator()
    row = layout.row(align=True)
    row.label(text="Control icons")
    row.prop(scene, "spine2d_control_icons", text="")
    row = layout.row(align=True)
    row.label(text="Preview animation")
    row.prop(scene, "spine2d_export_preview_animation", text="")


class SPINE2D_OT_ResetRigProfile(bpy.types.Operator):
    """Restore the public rig/projection settings to production defaults."""

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
