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
from .domain.baking import A1TextureExportMode


_PARENT_PANEL_ID = "OBJECT_PT_spine2d_mesh"


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


CLASSES = (
    OBJECT_PT_Spine2DRigPanel,
    OBJECT_PT_Spine2DGeneratedMaterialsPanel,
    OBJECT_PT_Spine2DDepthParallaxPanel,
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
    "OBJECT_PT_Spine2DDepthParallaxPanel",
    "OBJECT_PT_Spine2DGeneratedMaterialsPanel",
    "OBJECT_PT_Spine2DRigPanel",
    "RNA_PROPERTIES",
    "register",
    "unregister",
]
