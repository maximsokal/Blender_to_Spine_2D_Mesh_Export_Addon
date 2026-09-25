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


def _texture_mode(scene: bpy.types.Scene) -> A1TextureExportMode:
    raw = str(
        getattr(
            scene,
            "spine2d_texture_export_mode",
            A1TextureExportMode.NORMAL_UV_SEGMENTS.value,
        )
    ).strip().upper()
    try:
        return A1TextureExportMode(raw)
    except ValueError:
        return A1TextureExportMode.NORMAL_UV_SEGMENTS


def _selected_mesh_count(context: bpy.types.Context) -> int:
    """Count Mesh objects accepted by the selected-object export route."""

    return sum(
        1
        for candidate in getattr(context, "selected_objects", ())
        if getattr(candidate, "type", None) == "MESH"
        and getattr(candidate, "data", None) is not None
    )


def _resolved_projection_direction(
    scene: bpy.types.Scene,
) -> A1ProjectionDirection | None:
    """Resolve the persisted projection identifier without hiding invalid state."""

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
    texture_mode: A1TextureExportMode = A1TextureExportMode.NORMAL_UV_SEGMENTS,
) -> bool:
    """Mirror the exporter capability contract for conditional pivot UI visibility."""

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
    """Draw the default-on shared pivot only when the export route supports it."""

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


def draw_rig_settings(
    layout: bpy.types.UILayout,
    context: bpy.types.Context,
) -> None:
    """Draw rig and export-mode/projection controls.

    Export mode and projection choose how the rig is built (camera-space vs. world-axis
    bake, shared vs. per-object pivot, depth-generated bones), so they live here rather
    than in the Paths and Spine 2D version foldout.

    Per-object Connect settings are intentionally not drawn: the public Export Selected
    Objects route always builds standalone rigs and ignores persisted Connect values,
    so exposing the checkbox would present a control that has no effect.
    """

    scene = context.scene
    layout.prop(scene, "spine2d_texture_export_mode", text="Export mode")
    texture_mode = _texture_mode(scene)
    if texture_mode is A1TextureExportMode.CAMERA_PROJECTION:
        layout.label(
            text="Active camera render → flat screen-space mesh",
            icon="CAMERA_DATA",
        )
        layout.prop(
            scene,
            "spine2d_projection_alpha_threshold",
            text="Projection alpha threshold",
        )
    elif texture_mode is A1TextureExportMode.DEPTH_CAMERA_PROJECTION:
        layout.label(
            text="Active camera render → optimized depth-relief mesh",
            icon="CAMERA_DATA",
        )
        layout.label(text="Depth base: Farthest visible point", icon="INFO")
        layout.prop(
            scene,
            "spine2d_projection_alpha_threshold",
            text="Projection alpha threshold",
        )
        layout.separator()
        layout.prop(scene, "spine2d_depth_smoothing", text="Depth smoothing")
        layout.prop(scene, "spine2d_depth_edge_threshold", text="Depth edge threshold")
        layout.prop(
            scene,
            "spine2d_depth_mesh_error_pixels",
            text="Depth mesh error (px)",
        )
        layout.prop(scene, "spine2d_depth_max_points", text="Max depth points")
        layout.label(
            text="One generated vertex bone per retained depth point",
            icon="BONE_DATA",
        )
    else:
        layout.label(
            text="Preserves cut regions and generated UV meshes",
            icon="UV",
        )
        layout.prop(scene, "spine2d_projection_direction", text="Projection direction")
    layout.separator()

    header = layout.row(align=True)
    header.label(text="2-Axis Rotation + Scale", icon="CON_ROTLIKE")
    header.operator("spine2d.reset_rig_profile", text="Reset")

    layout.label(text="Controls: Rotation X / Y + Scale")

    if texture_mode is A1TextureExportMode.NORMAL_UV_SEGMENTS:
        _draw_shared_selection_pivot(layout, context)

    row = layout.row(align=True)
    row.label(text="Control icons")
    row.prop(scene, "spine2d_control_icons", text="")

    row = layout.row(align=True)
    row.label(text="Preview animation")
    row.prop(scene, "spine2d_export_preview_animation", text="")


class SPINE2D_OT_ResetRigProfile(bpy.types.Operator):
    """Restore only the persisted production rig profile default."""

    bl_idname = "spine2d.reset_rig_profile"
    bl_label = "Reset Rig Settings"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context: bpy.types.Context) -> Set[str]:
        try:
            scene = getattr(context, "scene", None)
            if scene is None:
                raise ValueError("context.scene is unavailable")
            scene.spine2d_rig_profile = A1RigProfile.TWO_AXIS_ROTATION_SCALE.value
            self.report({"INFO"}, "Rig profile reset to 2-Axis Rotation + Scale")
            return {"FINISHED"}
        except Exception as exc:
            logger.exception("Unable to reset Spine2D rig profile")
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
