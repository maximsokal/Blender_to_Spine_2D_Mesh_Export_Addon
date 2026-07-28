# pylint: disable=import-error
"""Own the exact visual order of Rewrite controls in the main Blender panel."""

from __future__ import annotations

import logging
import os
from typing import Callable

import bpy

from . import rig_ui, ui
from .blender_adapter import generated_material_ui
from .config import get_default_output_dir
from .domain.baking import A1TextureExportMode
from .infrastructure.blender_registration import (
    RnaPropertyRegistration,
    register_rna_properties_transactionally,
    rna_property_cleanup_actions,
    unregister_all_best_effort,
)


logger = logging.getLogger(__name__)

_ORIGINAL_PANEL_REMOVED = False
_ORDERED_PANEL_REGISTERED = False
_REGISTERED_RNA: tuple[RnaPropertyRegistration, ...] = ()


class OBJECT_PT_Spine2DOrderedMeshPanel(bpy.types.Panel):
    """Render every user-facing section with one standard foldout implementation.

    The class deliberately derives directly from ``bpy.types.Panel`` rather than from
    the replaceable base panel Python class. Blender development reloads may reload
    modules in reverse order; direct inheritance prevents this owner from retaining a
    stale, already-unregistered RNA parent class.
    """

    bl_idname = ui.OBJECT_PT_Spine2DMeshPanel.bl_idname
    bl_label = ui.OBJECT_PT_Spine2DMeshPanel.bl_label
    bl_space_type = ui.OBJECT_PT_Spine2DMeshPanel.bl_space_type
    bl_region_type = ui.OBJECT_PT_Spine2DMeshPanel.bl_region_type
    bl_category = ui.OBJECT_PT_Spine2DMeshPanel.bl_category

    def _draw_foldout(
        self,
        layout: bpy.types.UILayout,
        scene: bpy.types.Scene,
        *,
        property_name: str,
        title: str,
        draw_content: Callable[[bpy.types.UILayout], None],
    ) -> None:
        """Delegate the standard foldout appearance to the current UI module."""

        ui.OBJECT_PT_Spine2DMeshPanel._draw_foldout(
            self,
            layout,
            scene,
            property_name=property_name,
            title=title,
            draw_content=draw_content,
        )

    @staticmethod
    def _draw_cut_settings(
        column: bpy.types.UILayout,
        scene: bpy.types.Scene,
    ) -> None:
        ui.OBJECT_PT_Spine2DMeshPanel._draw_cut_settings(column, scene)

    @staticmethod
    def _draw_bake_settings(
        column: bpy.types.UILayout,
        context: bpy.types.Context,
    ) -> None:
        ui.OBJECT_PT_Spine2DMeshPanel._draw_bake_settings(column, context)

    @staticmethod
    def _issue_icon(severity) -> str:
        return ui.OBJECT_PT_Spine2DMeshPanel._issue_icon(severity)

    @staticmethod
    def _state_icon(state) -> str:
        return ui.OBJECT_PT_Spine2DMeshPanel._state_icon(state)

    def _draw_object_readiness(self, layout: bpy.types.UILayout, item) -> None:
        ui.OBJECT_PT_Spine2DMeshPanel._draw_object_readiness(self, layout, item)

    def _draw_readiness(
        self,
        layout: bpy.types.UILayout,
        context: bpy.types.Context,
    ) -> bool:
        return ui.OBJECT_PT_Spine2DMeshPanel._draw_readiness(self, layout, context)

    def _draw_export_settings(
        self,
        column: bpy.types.UILayout,
        context: bpy.types.Context,
    ) -> None:
        """Draw export settings without duplicating controls owned by Rig."""

        scene = context.scene
        column.prop(scene, "spine2d_texture_export_mode", text="Export mode")
        texture_mode = str(
            getattr(
                scene,
                "spine2d_texture_export_mode",
                A1TextureExportMode.NORMAL_UV_SEGMENTS.value,
            )
        ).upper()
        if texture_mode == A1TextureExportMode.CAMERA_PROJECTION.value:
            column.label(
                text="Active camera render → one screen-space mesh",
                icon="CAMERA_DATA",
            )
            column.prop(
                scene,
                "spine2d_projection_alpha_threshold",
                text="Projection alpha threshold",
            )
        else:
            column.label(
                text="Preserves cut regions and generated UV meshes",
                icon="UV",
            )
        column.separator()

        column.prop(scene, "spine2d_texture_size", text="Texture size")
        column.separator()
        column.prop(scene, "spine2d_json_path", text="JSON")
        json_full_path = bpy.path.abspath(scene.spine2d_json_path)
        if not json_full_path or json_full_path == bpy.path.abspath("//"):
            json_full_path = get_default_output_dir()
        column.label(text=json_full_path)
        column.separator()

        column.prop(scene, "spine2d_images_path", text="Images Subfolder")
        images_full_path = os.path.join(json_full_path, scene.spine2d_images_path)
        column.label(text=os.path.normpath(images_full_path))

        selected_meshes = tuple(
            candidate
            for candidate in getattr(context, "selected_objects", ())
            if getattr(candidate, "type", None) == "MESH"
        )
        if len(selected_meshes) > 1:
            column.separator()
            column.label(text="Connect objects:")
            for selected_object in selected_meshes:
                row = column.row(align=True)
                row.label(text=selected_object.name, icon="MESH_DATA")
                row.prop(
                    selected_object.spine2d_connect_settings,
                    "enabled",
                    text="",
                )

    def _draw_readiness_and_export(
        self,
        column: bpy.types.UILayout,
        context: bpy.types.Context,
    ) -> None:
        """Keep analysis and the guarded export action in one final foldout."""

        export_allowed = self._draw_readiness(column, context)
        row = column.row()
        row.enabled = export_allowed
        selected_meshes = tuple(
            candidate
            for candidate in getattr(context, "selected_objects", ())
            if getattr(candidate, "type", None) == "MESH"
        )
        if len(selected_meshes) <= 1:
            row.operator(
                "object.spine2d_single_export",
                text="Export Current Object",
            )
        else:
            row.operator(
                "object.spine2d_multi_export",
                text="Export Selected Objects",
            )

    def draw(self, context: bpy.types.Context) -> None:
        """Draw the exact requested order using one visual foldout style."""

        layout = self.layout
        scene = context.scene
        try:
            if not bpy.data.filepath:
                layout.label(text="Blend file not saved!", icon="ERROR")
                layout.label(text="Please save your .blend first.")
                layout.enabled = False
                return

            header = layout.row(align=True)
            header.label(text="Rewrite settings:")
            header.operator("spine2d.reset_settings", text="Reset")

            # User-facing order is an explicit UI contract.
            self._draw_foldout(
                layout,
                scene,
                property_name="spine2d_show_settings",
                title="Export",
                draw_content=lambda content: self._draw_export_settings(
                    content,
                    context,
                ),
            )
            self._draw_foldout(
                layout,
                scene,
                property_name="spine2d_show_rig_settings",
                title="Rig",
                draw_content=lambda content: rig_ui.draw_rig_settings(
                    content,
                    context,
                ),
            )
            self._draw_foldout(
                layout,
                scene,
                property_name="spine2d_show_generated_material_settings",
                title="Rewrite Generated Materials",
                draw_content=lambda content: (
                    generated_material_ui.draw_generated_material_settings(
                        content,
                        context,
                    )
                ),
            )
            self._draw_foldout(
                layout,
                scene,
                property_name="spine2d_show_cut_settings",
                title="Cut",
                draw_content=lambda content: self._draw_cut_settings(
                    content,
                    scene,
                ),
            )
            self._draw_foldout(
                layout,
                scene,
                property_name="spine2d_show_bake_settings",
                title="Bake",
                draw_content=lambda content: self._draw_bake_settings(
                    content,
                    context,
                ),
            )
            self._draw_foldout(
                layout,
                scene,
                property_name="spine2d_show_readiness",
                title="Export Readiness",
                draw_content=lambda content: self._draw_readiness_and_export(
                    content,
                    context,
                ),
            )
        except Exception:
            logger.exception("Unable to draw ordered Spine2D Rewrite UI")
            layout.label(text="UI error (see console)", icon="ERROR")


RNA_PROPERTIES = (
    RnaPropertyRegistration(
        owner=bpy.types.Scene,
        name="spine2d_show_rig_settings",
        value=bpy.props.BoolProperty(
            name="Show Rig Settings",
            default=False,
            description="Show or hide rig-profile controls",
        ),
    ),
    RnaPropertyRegistration(
        owner=bpy.types.Scene,
        name="spine2d_show_generated_material_settings",
        value=bpy.props.BoolProperty(
            name="Show Generated Material Settings",
            default=False,
            description="Show or hide generated-material controls",
        ),
    ),
    RnaPropertyRegistration(
        owner=bpy.types.Scene,
        name="spine2d_show_readiness",
        value=bpy.props.BoolProperty(
            name="Show Export Readiness",
            default=True,
            description="Show or hide analysis results and export actions",
        ),
    ),
)


def _restore_original_panel() -> None:
    global _ORIGINAL_PANEL_REMOVED
    if not _ORIGINAL_PANEL_REMOVED:
        return
    bpy.utils.register_class(ui.OBJECT_PT_Spine2DMeshPanel)
    _ORIGINAL_PANEL_REMOVED = False


def register() -> None:
    """Replace the base panel transactionally while preserving its operators/RNA."""

    global _ORIGINAL_PANEL_REMOVED, _ORDERED_PANEL_REGISTERED, _REGISTERED_RNA
    if _ORDERED_PANEL_REGISTERED:
        return

    registered_rna = register_rna_properties_transactionally(RNA_PROPERTIES)
    try:
        bpy.utils.unregister_class(ui.OBJECT_PT_Spine2DMeshPanel)
        _ORIGINAL_PANEL_REMOVED = True
        bpy.utils.register_class(OBJECT_PT_Spine2DOrderedMeshPanel)
        _ORDERED_PANEL_REGISTERED = True
        _REGISTERED_RNA = tuple(registered_rna)
    except Exception as exc:
        logger.exception("Ordered Spine2D UI registration failed")
        if _ORDERED_PANEL_REGISTERED:
            try:
                bpy.utils.unregister_class(OBJECT_PT_Spine2DOrderedMeshPanel)
            except Exception:
                logger.exception("Unable to remove partial ordered panel")
            _ORDERED_PANEL_REGISTERED = False
        try:
            _restore_original_panel()
        except Exception:
            logger.exception("Unable to restore base panel during rollback")
        unregister_all_best_effort(
            rna_property_cleanup_actions(tuple(registered_rna)),
            operation="ordered UI RNA registration rollback",
            primary_error=exc,
        )
        _REGISTERED_RNA = ()
        raise
    logger.debug("Ordered Spine2D UI registered")


def unregister() -> None:
    """Remove the ordered panel, restore the base panel, and release RNA state."""

    global _ORDERED_PANEL_REGISTERED, _REGISTERED_RNA
    errors: list[BaseException] = []
    if _ORDERED_PANEL_REGISTERED:
        try:
            bpy.utils.unregister_class(OBJECT_PT_Spine2DOrderedMeshPanel)
        except Exception as exc:
            logger.exception("Unable to unregister ordered Spine2D panel")
            errors.append(exc)
        else:
            _ORDERED_PANEL_REGISTERED = False

    try:
        _restore_original_panel()
    except Exception as exc:
        logger.exception("Unable to restore base Spine2D panel")
        errors.append(exc)

    try:
        unregister_all_best_effort(
            rna_property_cleanup_actions(_REGISTERED_RNA or RNA_PROPERTIES),
            operation="ordered UI RNA unregistration",
            primary_error=errors[0] if errors else None,
        )
    finally:
        _REGISTERED_RNA = ()

    if errors:
        raise RuntimeError("Ordered Spine2D UI unregistration failed") from errors[0]
    logger.debug("Ordered Spine2D UI unregistered")


__all__ = [
    "OBJECT_PT_Spine2DOrderedMeshPanel",
    "RNA_PROPERTIES",
    "register",
    "unregister",
]
