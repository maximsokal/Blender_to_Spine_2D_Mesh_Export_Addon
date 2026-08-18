# pylint: disable=import-error
"""Additional exporter UI sections using ordinary Blender child panels.

The previous implementation temporarily unregistered the main panel and replaced it
with another class during add-on registration. Blender Extensions review explicitly
rejected that lifecycle. The main panel now remains owned by :mod:`ui`; this module
only registers independent child panels in the normal Blender pattern.
"""

from __future__ import annotations

from typing import Any

import bpy

from . import rig_ui
from .blender_adapter import generated_material_ui
from .blender_adapter.normal_uv_modifier_warnings import (
    collect_normal_uv_ignored_modifiers,
    group_ignored_modifiers_by_object,
)
from .domain.baking import A1TextureExportMode


_PARENT_PANEL_ID = "OBJECT_PT_spine2d_mesh"
_MAX_VISIBLE_MODIFIERS_PER_OBJECT = 8


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


def _analysis_mesh_objects(context: bpy.types.Context) -> tuple[bpy.types.Object, ...]:
    """Resolve the Mesh request represented by the current single/multi UI state."""

    selected = tuple(
        candidate
        for candidate in getattr(context, "selected_objects", ())
        if getattr(candidate, "type", None) == "MESH"
        and getattr(candidate, "data", None) is not None
    )
    if len(selected) > 1:
        return selected

    active = getattr(context, "active_object", None)
    if (
        active is not None
        and getattr(active, "type", None) == "MESH"
        and getattr(active, "data", None) is not None
    ):
        return (active,)
    return selected


def _draw_modifier_analysis_warning(
    layout: bpy.types.UILayout,
    context: bpy.types.Context,
) -> None:
    """Warn when Normal / UV export cannot reproduce evaluated modifier geometry.

    The warning is advisory only. It never disables Analyze or Export and therefore
    remains compatible with the explicit, non-blocking readiness policy.
    """

    scene = getattr(context, "scene", None)
    if scene is None:
        return

    descriptors = collect_normal_uv_ignored_modifiers(
        _analysis_mesh_objects(context),
        _texture_mode(scene),
    )
    if not descriptors:
        return

    box = layout.box()
    box.alert = True
    box.label(
        text="Normal / UV Segments ignores active modifiers",
        icon="ERROR",
    )
    box.label(text="Viewport and Spine geometry can look different.")
    box.label(text="Apply or convert modifiers before export.", icon="INFO")

    for object_name, modifiers in group_ignored_modifiers_by_object(descriptors):
        box.separator()
        box.label(text=object_name, icon="MESH_DATA")
        visible = modifiers[:_MAX_VISIBLE_MODIFIERS_PER_OBJECT]
        for modifier in visible:
            states: list[str] = []
            if modifier.show_viewport:
                states.append("viewport")
            if modifier.show_render:
                states.append("render")
            box.label(
                text=(
                    f"{modifier.modifier_name} ({modifier.modifier_type}) — "
                    + "/".join(states)
                ),
                icon="MODIFIER",
            )
        hidden_count = len(modifiers) - len(visible)
        if hidden_count > 0:
            box.label(text=f"... and {hidden_count} more modifier(s)")


class OBJECT_PT_Spine2DRigPanel(bpy.types.Panel):
    """Expose rig/projection controls below the canonical exporter panel."""

    bl_idname = "OBJECT_PT_spine2d_rig"
    bl_label = "Rig"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_parent_id = _PARENT_PANEL_ID
    bl_order = 10

    def draw(self, context: bpy.types.Context) -> None:
        rig_ui.draw_rig_settings(self.layout, context)


class OBJECT_PT_Spine2DGeneratedMaterialsPanel(bpy.types.Panel):
    """Expose generated-material controls without replacing the main panel."""

    bl_idname = "OBJECT_PT_spine2d_generated_materials"
    bl_label = "Generated Materials"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_parent_id = _PARENT_PANEL_ID
    bl_order = 20

    def draw(self, context: bpy.types.Context) -> None:
        generated_material_ui.draw_generated_material_settings(self.layout, context)


class OBJECT_PT_Spine2DDepthParallaxPanel(bpy.types.Panel):
    """Expose depth-only parallax reserve settings when that mode is active."""

    bl_idname = "OBJECT_PT_spine2d_depth_parallax"
    bl_label = "Depth Parallax"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_parent_id = _PARENT_PANEL_ID
    bl_order = 30

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        scene = getattr(context, "scene", None)
        return bool(
            scene is not None
            and _texture_mode(scene) is A1TextureExportMode.DEPTH_CAMERA_PROJECTION
        )

    def draw(self, context: bpy.types.Context) -> None:
        scene = context.scene
        layout = self.layout
        layout.prop(
            scene,
            "spine2d_depth_parallax_horizon_angle",
            text="Parallax Horizon Angle",
        )
        angle = float(
            getattr(scene, "spine2d_depth_parallax_horizon_angle", 0.0) or 0.0
        )
        if angle <= 1.0e-12:
            layout.label(text="0°: front surface only", icon="INFO")
        else:
            layout.label(
                text="Retains connected surfaces beyond the camera horizon",
                icon="MESH_DATA",
            )
            layout.label(
                text="Max Depth Points limits front + reserve geometry",
                icon="BONE_DATA",
            )


class OBJECT_PT_Spine2DAnalysisPanel(bpy.types.Panel):
    """Expose advisory source-vs-export diagnostics without replacing the main panel."""

    bl_idname = "OBJECT_PT_spine2d_analysis"
    bl_label = "Analysis"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_parent_id = _PARENT_PANEL_ID
    bl_order = 40

    def draw(self, context: bpy.types.Context) -> None:
        layout = self.layout
        _draw_modifier_analysis_warning(layout, context)
        layout.label(
            text="Use Analyze in the main exporter panel for readiness diagnostics.",
            icon="VIEWZOOM",
        )


CLASSES = (
    OBJECT_PT_Spine2DRigPanel,
    OBJECT_PT_Spine2DGeneratedMaterialsPanel,
    OBJECT_PT_Spine2DDepthParallaxPanel,
    OBJECT_PT_Spine2DAnalysisPanel,
)

# Compatibility symbol only. Child panels no longer require custom foldout RNA state.
RNA_PROPERTIES: tuple[Any, ...] = ()


def register() -> None:
    """Register child panels in declaration order."""

    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister() -> None:
    """Unregister child panels in reverse declaration order."""

    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)


__all__ = [
    "CLASSES",
    "OBJECT_PT_Spine2DAnalysisPanel",
    "OBJECT_PT_Spine2DDepthParallaxPanel",
    "OBJECT_PT_Spine2DGeneratedMaterialsPanel",
    "OBJECT_PT_Spine2DRigPanel",
    "RNA_PROPERTIES",
    "register",
    "unregister",
]
