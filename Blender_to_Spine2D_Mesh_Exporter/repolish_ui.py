# pylint: disable=import-error
"""Small Blender UI owner for the related Re-Polish optimization project."""

from __future__ import annotations

import logging

import bpy

from .infrastructure.blender_registration import (
    class_cleanup_actions,
    register_classes_transactionally,
    unregister_all_best_effort,
)


logger = logging.getLogger(__name__)
REPOLISH_URL = "https://www.re-polish.com/"


class OBJECT_PT_Spine2DRePolishPanel(bpy.types.Panel):
    """Link users to the related Spine animation optimization project."""

    bl_label = "Spine Animation Optimization"
    bl_idname = "OBJECT_PT_spine2d_repolish"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Blender to Spine2D Mesh Exporter"
    bl_parent_id = "OBJECT_PT_spine2d_mesh"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, _context: bpy.types.Context) -> None:
        layout = self.layout
        layout.label(text="Optimize and clean Spine animations:", icon="INFO")
        operator = layout.operator(
            "wm.url_open",
            text="Open Re-Polish",
            icon="URL",
        )
        operator.url = REPOLISH_URL


CLASSES = (OBJECT_PT_Spine2DRePolishPanel,)


def register() -> None:
    """Register the child panel after the main Rewrite UI panel."""

    register_classes_transactionally(
        CLASSES,
        register_class=bpy.utils.register_class,
        unregister_class=bpy.utils.unregister_class,
    )


def unregister() -> None:
    """Unregister the child panel best-effort."""

    unregister_all_best_effort(
        class_cleanup_actions(
            CLASSES,
            unregister_class=bpy.utils.unregister_class,
        ),
        operation="Re-Polish UI unregistration",
    )


__all__ = [
    "CLASSES",
    "OBJECT_PT_Spine2DRePolishPanel",
    "REPOLISH_URL",
    "register",
    "unregister",
]
