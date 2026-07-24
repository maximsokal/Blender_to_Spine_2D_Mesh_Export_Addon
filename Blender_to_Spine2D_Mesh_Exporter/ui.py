# pylint: disable=import-error
"""Blender 5.2+ Rewrite UI and multi-object export operator."""

from __future__ import annotations

import logging
from math import isfinite
import os
from typing import Callable, Set

import bpy

from .blender_adapter.a1_ui_bridge import export_selected_objects_a1
from .config import get_default_output_dir
from .infrastructure.blender_registration import (
    RnaPropertyRegistration,
    class_cleanup_actions,
    register_classes_transactionally,
    register_rna_properties_transactionally,
    rna_property_cleanup_actions,
    unregister_all_best_effort,
)


logger = logging.getLogger(__name__)


class SPINE2D_OT_ResetSettings(bpy.types.Operator):
    """Reset Rewrite export settings to their defaults."""

    bl_idname = "spine2d.reset_settings"
    bl_label = "Reset Spine2D Settings"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context: bpy.types.Context) -> Set[str]:
        try:
            scene = context.scene
            scene.spine2d_texture_size = 1024
            scene.spine2d_json_path = get_default_output_dir()
            scene.spine2d_images_path = "images/"
            scene.spine2d_control_icons = True
            scene.spine2d_export_preview_animation = True
            scene.spine2d_angle_limit = 30
            scene.spine2d_angular_mode = "SEED_CONE"
            scene.spine2d_local_angle_limit = 30.0
            scene.spine2d_seam_maker_mode = "AUTO"
            scene.spine2d_frames_for_render = 0
            scene.spine2d_bake_frame_start = 0
            self.report({"INFO"}, "Spine2D Rewrite settings have been reset.")
            return {"FINISHED"}
        except Exception as exc:
            logger.exception("Unable to reset Spine2D Rewrite settings")
            self.report({"ERROR"}, f"Reset error: {exc}")
            return {"CANCELLED"}


class OBJECT_OT_Spine2DRefreshInfo(bpy.types.Operator):
    """Recalculate and cache object information displayed by the panel."""

    bl_idname = "object.spine2d_refresh_info"
    bl_label = "Refresh Object Info"

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        active = context.active_object
        return bool(active is not None and active.type == "MESH")

    def execute(self, context: bpy.types.Context) -> Set[str]:
        obj = context.active_object
        if obj is None or obj.type != "MESH" or obj.data is None:
            self.report({"ERROR"}, "A valid active Mesh object is required")
            return {"CANCELLED"}
        try:
            obj["_spine2d_vertex_count"] = len(obj.data.vertices)
            inverted, correct = OBJECT_PT_Spine2DMeshPanel._face_orientation_stats(obj)
            obj["_spine2d_face_stats"] = {
                "inverted": inverted,
                "correct": correct,
            }

            for material in obj.data.materials:
                if material is None:
                    continue
                icon_id = 0
                try:
                    material.preview_ensure()
                    preview = material.preview
                    icon_id = int(getattr(preview, "icon_id", 0) or 0)
                except Exception:
                    logger.exception(
                        "Unable to generate material preview for '%s'",
                        material.name,
                    )
                material["_spine2d_icon_id"] = icon_id

            self.report({"INFO"}, "Object info cache has been updated.")
            return {"FINISHED"}
        except Exception as exc:
            logger.exception("Unable to refresh Spine2D object information")
            self.report({"ERROR"}, f"Refresh error: {exc}")
            return {"CANCELLED"}


class Spine2DConnectSettings(bpy.types.PropertyGroup):
    """Per-object setting for the connected Rewrite multi-object subgroup."""

    enabled: bpy.props.BoolProperty(
        name="Connect",
        description="Attach this object to the shared all_objects rig",
        default=False,
    )


class Spine2DBakeSettings(bpy.types.PropertyGroup):
    """Per-object Rewrite sequence-baking settings."""

    frames_for_render: bpy.props.IntProperty(
        name="Frames for render",
        description="How many frames to render (0 = current frame only)",
        default=0,
        min=0,
    )
    bake_frame_start: bpy.props.IntProperty(
        name="Start frame",
        description="First frame of the sequence",
        default=0,
        min=0,
    )


class OBJECT_PT_Spine2DMeshPanel(bpy.types.Panel):
    """Main Blender 5.2+ Rewrite exporter panel."""

    bl_label = "Blender to Spine2D Mesh Exporter"
    bl_idname = "OBJECT_PT_spine2d_mesh"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Blender to Spine2D Mesh Exporter"

    @staticmethod
    def _scale_applied(obj: bpy.types.Object, tolerance: float = 1e-4) -> bool:
        """Compatibility helper retained for external callers and source tests."""

        return all(abs(float(value) - 1.0) < tolerance for value in obj.scale)

    @staticmethod
    def _world_linear_transform_status(
        obj: bpy.types.Object,
        tolerance: float = 1.0e-12,
    ) -> tuple[bool, bool, float]:
        """Return ``(is_singular, requires_normalization, determinant)``."""

        linear = obj.matrix_world.to_3x3()
        values = tuple(
            float(linear[row][column])
            for row in range(3)
            for column in range(3)
        )
        determinant = float(linear.determinant())
        coefficient_scale = max(1.0, *(abs(value) for value in values))
        threshold = tolerance * coefficient_scale**3
        is_singular = (
            not isfinite(determinant)
            or not isfinite(threshold)
            or abs(determinant) <= threshold
        )
        identity = (
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
        )
        requires_normalization = any(
            actual != expected
            for actual, expected in zip(values, identity, strict=True)
        )
        return is_singular, requires_normalization, determinant

    @staticmethod
    def _face_orientation_stats(obj: bpy.types.Object) -> tuple[int, int]:
        try:
            mesh = obj.data
            matrix_world = obj.matrix_world
            normal_matrix = matrix_world.to_3x3().inverted_safe().transposed()
            origin = matrix_world.translation
            inverted = 0
            correct = 0
            for polygon in mesh.polygons:
                center_world = matrix_world @ polygon.center
                normal_world = (normal_matrix @ polygon.normal).normalized()
                direction = center_world - origin
                if direction.length_squared <= 1e-16:
                    continue
                if direction.normalized().dot(normal_world) < 0.0:
                    inverted += 1
                else:
                    correct += 1
            return inverted, correct
        except Exception:
            logger.exception("Unable to calculate face orientation statistics")
            return 0, 0

    @staticmethod
    def _list_materials(
        box: bpy.types.UILayout,
        obj: bpy.types.Object,
    ) -> None:
        materials = tuple(
            material for material in getattr(obj.data, "materials", ()) if material
        )
        if not materials:
            box.label(text="No materials", icon="ERROR")
            return

        column = box.column(align=True)
        column.label(text=f"Materials ({len(materials)}):")
        for material in materials:
            row = column.row(align=True)
            icon_id = int(material.get("_spine2d_icon_id", 0) or 0)
            if icon_id:
                row.label(text=material.name, icon_value=icon_id)
            else:
                row.label(text=material.name, icon="MATERIAL")

    def _populate_info_box(
        self,
        box: bpy.types.UILayout,
        obj: bpy.types.Object | None,
    ) -> bool:
        if obj is None:
            box.label(text="No active object", icon="ERROR")
            return False
        if obj.type != "MESH" or obj.data is None:
            box.label(text="Active object is not a valid Mesh", icon="ERROR")
            return False

        vertex_count = obj.get("_spine2d_vertex_count")
        if vertex_count is None:
            box.label(text="Vertex count: Press Refresh", icon="QUESTION")
        else:
            box.label(text=f"Vertex count: {vertex_count}", icon="INFO")

        is_singular, requires_normalization, determinant = (
            self._world_linear_transform_status(obj)
        )
        export_allowed = not is_singular
        if is_singular:
            box.label(
                text=(
                    "Object transform is singular; set every scale axis to a "
                    "non-zero value"
                ),
                icon="ERROR",
            )
        elif requires_normalization:
            box.label(
                text="Rotation/scale will be normalized during export",
                icon="INFO",
            )
            if determinant < 0.0:
                box.label(
                    text="Mirrored transform will preserve mirrored winding",
                    icon="INFO",
                )

        self._list_materials(box, obj)
        statistics = obj.get("_spine2d_face_stats")
        if statistics:
            inverted = int(statistics.get("inverted", 0))
            correct = int(statistics.get("correct", 0))
            if inverted == 0:
                box.label(text="All faces oriented correctly", icon="INFO")
            else:
                box.label(
                    text=f"Inverted faces: {inverted} / {inverted + correct}",
                    icon="ERROR",
                )
        else:
            box.label(text="Face orientation: Press Refresh", icon="QUESTION")
        return export_allowed

    def _draw_foldout(
        self,
        layout: bpy.types.UILayout,
        scene: bpy.types.Scene,
        *,
        property_name: str,
        title: str,
        draw_content: Callable[[bpy.types.UILayout], None],
    ) -> None:
        box = layout.box()
        row = box.row()
        expanded = bool(getattr(scene, property_name))
        row.prop(
            scene,
            property_name,
            icon="TRIA_DOWN" if expanded else "TRIA_RIGHT",
            text="",
            icon_only=True,
            emboss=False,
        )
        row.label(text=title)
        if expanded:
            draw_content(box.column(align=True))

    def _draw_export_settings(
        self,
        column: bpy.types.UILayout,
        context: bpy.types.Context,
    ) -> None:
        scene = context.scene
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
        column.separator()

        row = column.row(align=True)
        row.label(text="Control icons")
        row.prop(scene, "spine2d_control_icons", text="")
        row = column.row(align=True)
        row.label(text="Preview animation")
        row.prop(scene, "spine2d_export_preview_animation", text="")

        selected_meshes = tuple(
            candidate
            for candidate in context.selected_objects
            if candidate.type == "MESH"
        )
        if len(selected_meshes) > 1:
            column.separator()
            column.label(text="Connect objects:")
            for selected_object in selected_meshes:
                row = column.row(align=True)
                row.label(text=selected_object.name, icon="MESH_DATA")
                row.prop(selected_object.spine2d_connect_settings, "enabled", text="")

    @staticmethod
    def _draw_cut_settings(
        column: bpy.types.UILayout,
        scene: bpy.types.Scene,
    ) -> None:
        column.prop(scene, "spine2d_seam_maker_mode", text="Seam maker")
        if str(scene.spine2d_seam_maker_mode).upper() == "CUSTOM":
            column.label(
                text="Angular splitting is disabled in Custom seam mode",
                icon="INFO",
            )
            return
        column.separator()
        column.prop(scene, "spine2d_angle_limit", text="Seed angle limit")
        column.prop(scene, "spine2d_angular_mode", text="Angular mode")
        if (
            str(scene.spine2d_angular_mode).upper()
            == "SEED_CONE_AND_LOCAL_DIHEDRAL"
        ):
            column.prop(
                scene,
                "spine2d_local_angle_limit",
                text="Local edge angle limit",
            )

    @staticmethod
    def _draw_bake_settings(
        column: bpy.types.UILayout,
        context: bpy.types.Context,
    ) -> None:
        scene = context.scene
        selected_meshes = tuple(
            candidate
            for candidate in context.selected_objects
            if candidate.type == "MESH"
        )
        if len(selected_meshes) > 1:
            for selected_object in selected_meshes:
                box = column.box()
                box.label(text=selected_object.name, icon="MESH_DATA")
                settings = selected_object.spine2d_bake_settings
                box.prop(settings, "frames_for_render", text="Frames")
                box.prop(settings, "bake_frame_start", text="Start")
                start = max(0, int(settings.bake_frame_start))
                frames = max(0, int(settings.frames_for_render))
                last = start if frames == 0 else start + frames - 1
                box.label(text=f"Last frame: {last}")
            return

        column.prop(scene, "spine2d_frames_for_render", text="Frames for render")
        column.prop(scene, "spine2d_bake_frame_start", text="Start")
        start = max(0, int(scene.spine2d_bake_frame_start))
        frames = max(0, int(scene.spine2d_frames_for_render))
        last = start if frames == 0 else start + frames - 1
        column.label(text=f"Last frame: {last}")
        column.label(text=f"Playback end: {scene.frame_end}")

    def draw(self, context: bpy.types.Context) -> None:
        layout = self.layout
        scene = context.scene
        obj = context.active_object

        try:
            if not bpy.data.filepath:
                layout.label(text="Blend file not saved!", icon="ERROR")
                layout.label(text="Please save your .blend first.")
                layout.enabled = False
                return

            header = layout.row(align=True)
            header.label(text="Rewrite settings:")
            header.operator("spine2d.reset_settings", text="Reset")

            self._draw_foldout(
                layout,
                scene,
                property_name="spine2d_show_settings",
                title="Export",
                draw_content=lambda column: self._draw_export_settings(column, context),
            )
            self._draw_foldout(
                layout,
                scene,
                property_name="spine2d_show_cut_settings",
                title="Cut",
                draw_content=lambda column: self._draw_cut_settings(column, scene),
            )
            self._draw_foldout(
                layout,
                scene,
                property_name="spine2d_show_bake_settings",
                title="Bake",
                draw_content=lambda column: self._draw_bake_settings(column, context),
            )

            layout.separator()
            info_box = layout.box()
            row = info_box.row(align=True)
            row.label(text="Info:")
            row.operator(
                "object.spine2d_refresh_info",
                text="Refresh",
                icon="FILE_REFRESH",
            )
            export_allowed = self._populate_info_box(info_box, obj)

            row = layout.row()
            row.enabled = export_allowed
            selected_meshes = tuple(
                candidate
                for candidate in context.selected_objects
                if candidate.type == "MESH"
            )
            if len(selected_meshes) <= 1:
                row.operator("object.save_uv_as_json", text="Export Current Object")
            else:
                row.operator(
                    "object.spine2d_multi_export",
                    text="Export Selected Objects",
                )
        except Exception:
            logger.exception("Unable to draw the Spine2D Rewrite panel")
            layout.label(text="UI error (see console)", icon="ERROR")


class OBJECT_OT_Spine2DMultiExport(bpy.types.Operator):
    """Export selected Mesh objects through the Rewrite pipeline."""

    bl_idname = "object.spine2d_multi_export"
    bl_label = "Export Selected Objects"
    bl_description = "Export selected Mesh objects with the Rewrite pipeline"

    def _report_result(self, result) -> Set[str]:
        for issue in result.issues:
            self.report(
                {issue.severity.value},
                f"{issue.code}: {issue.message}",
            )
        if result.success:
            destination = result.output_files[0] if result.output_files else "completed"
            self.report({"INFO"}, f"Export finished → {destination}")
            return {"FINISHED"}
        if not result.issues:
            self.report({"ERROR"}, "Rewrite export failed without diagnostics")
        return {"CANCELLED"}

    def execute(self, context: bpy.types.Context) -> Set[str]:
        try:
            result = export_selected_objects_a1(context)
            return self._report_result(result)
        except Exception as exc:
            logger.exception("Rewrite multi-object export failed")
            self.report({"ERROR"}, f"Multi-object export failed: {exc}")
            return {"CANCELLED"}


SCENE_PROPERTIES = (
    (
        "spine2d_show_settings",
        bpy.props.BoolProperty(
            name="Show Settings",
            default=False,
            description="Show or hide Spine2D export settings",
        ),
    ),
    (
        "spine2d_show_cut_settings",
        bpy.props.BoolProperty(
            name="Show Cut Settings",
            default=False,
            description="Show or hide cutting parameters",
        ),
    ),
    (
        "spine2d_show_bake_settings",
        bpy.props.BoolProperty(
            name="Show Bake Settings",
            default=False,
            description="Show or hide baking parameters",
        ),
    ),
    (
        "spine2d_angular_mode",
        bpy.props.EnumProperty(
            name="Angular mode",
            description="Choose seed-normal segmentation or add a local dihedral guard",
            items=(
                (
                    "SEED_CONE",
                    "Seed cone",
                    "Compare every candidate with the segment seed normal",
                ),
                (
                    "SEED_CONE_AND_LOCAL_DIHEDRAL",
                    "Seed cone + local dihedral",
                    "Also reject traversal across locally sharp edges",
                ),
            ),
            default="SEED_CONE",
        ),
    ),
    (
        "spine2d_local_angle_limit",
        bpy.props.FloatProperty(
            name="Local edge angle limit",
            description="Maximum angle across each traversed edge in hybrid mode",
            default=30.0,
            min=0.0,
            max=180.0,
            precision=2,
        ),
    ),
    (
        "spine2d_bake_frame_start",
        bpy.props.IntProperty(
            name="Start frame",
            description="Frame to start sequence baking from",
            default=0,
            min=0,
        ),
    ),
    (
        "spine2d_control_icons",
        bpy.props.BoolProperty(
            name="Control icons",
            description="Export control icons in the final JSON",
            default=True,
        ),
    ),
    (
        "spine2d_export_preview_animation",
        bpy.props.BoolProperty(
            name="Preview animation",
            description="Add a preview animation to the final JSON",
            default=True,
        ),
    ),
)


CLASSES = (
    Spine2DBakeSettings,
    Spine2DConnectSettings,
    SPINE2D_OT_ResetSettings,
    OBJECT_OT_Spine2DRefreshInfo,
    OBJECT_PT_Spine2DMeshPanel,
    OBJECT_OT_Spine2DMultiExport,
)

RNA_PROPERTIES = tuple(
    RnaPropertyRegistration(
        owner=bpy.types.Scene,
        name=name,
        value=prop,
    )
    for name, prop in SCENE_PROPERTIES
) + (
    RnaPropertyRegistration(
        owner=bpy.types.Object,
        name="spine2d_bake_settings",
        value=bpy.props.PointerProperty(type=Spine2DBakeSettings),
    ),
    RnaPropertyRegistration(
        owner=bpy.types.Object,
        name="spine2d_connect_settings",
        value=bpy.props.PointerProperty(type=Spine2DConnectSettings),
    ),
)


def register() -> None:
    """Register all Rewrite UI classes and RNA properties transactionally."""

    registered_classes = register_classes_transactionally(
        CLASSES,
        register_class=bpy.utils.register_class,
        unregister_class=bpy.utils.unregister_class,
    )
    try:
        register_rna_properties_transactionally(RNA_PROPERTIES)
    except Exception as exc:
        logger.exception("Rewrite UI RNA registration failed")
        unregister_all_best_effort(
            class_cleanup_actions(
                registered_classes,
                unregister_class=bpy.utils.unregister_class,
            ),
            operation="Rewrite UI registration rollback",
            primary_error=exc,
        )
        raise
    logger.debug("Rewrite UI registered")


def unregister() -> None:
    """Remove every Rewrite UI property and class before reporting failures."""

    try:
        unregister_all_best_effort(
            (
                *rna_property_cleanup_actions(RNA_PROPERTIES),
                *class_cleanup_actions(
                    CLASSES,
                    unregister_class=bpy.utils.unregister_class,
                ),
            ),
            operation="Rewrite UI unregistration",
        )
    except Exception:
        logger.exception("Rewrite UI unregistration failed")
        raise
    logger.debug("Rewrite UI unregistered")


__all__ = [
    "CLASSES",
    "OBJECT_OT_Spine2DMultiExport",
    "OBJECT_OT_Spine2DRefreshInfo",
    "OBJECT_PT_Spine2DMeshPanel",
    "RNA_PROPERTIES",
    "SPINE2D_OT_ResetSettings",
    "Spine2DBakeSettings",
    "Spine2DConnectSettings",
    "register",
    "unregister",
]
