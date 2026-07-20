# ui.py
# pylint: disable=import-error
"""Blender UI and operators for the Spine2D mesh exporter."""

from __future__ import annotations

import logging
import os
from typing import Set

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
from .multi_object_export import export_selected_objects

logger = logging.getLogger(__name__)
logger.debug("[LOG] Loading ui.py")

MULTI_BACKEND_PROPERTY = "spine2d_multi_export_backend"
DEFAULT_MULTI_BACKEND = "REWRITE"
_MULTI_BACKENDS = frozenset({"REWRITE", "LEGACY"})


def resolve_multi_backend(scene: bpy.types.Scene) -> str:
    """Resolve the multi-object backend and fail closed to Rewrite."""

    raw_value = getattr(scene, MULTI_BACKEND_PROPERTY, DEFAULT_MULTI_BACKEND)
    backend = str(raw_value).strip().upper()
    if backend in _MULTI_BACKENDS:
        return backend
    logger.warning(
        "Unknown multi export backend %r; using %s",
        raw_value,
        DEFAULT_MULTI_BACKEND,
    )
    return DEFAULT_MULTI_BACKEND


class SPINE2D_OT_ResetSettings(bpy.types.Operator):
    """Resets the addon settings to their default values."""

    bl_idname = "spine2d.reset_settings"
    bl_label = "Reset Spine2D Settings"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context: bpy.types.Context) -> Set[str]:
        try:
            scene = context.scene
            scene.spine2d_texture_size = 1024
            scene.spine2d_json_path = get_default_output_dir()
            scene.spine2d_images_path = "./images/"
            scene.spine2d_control_icons = True
            scene.spine2d_export_preview_animation = True
            setattr(scene, MULTI_BACKEND_PROPERTY, DEFAULT_MULTI_BACKEND)
            scene.spine2d_angle_limit = 30
            scene.spine2d_angular_mode = "LEGACY_SEED_CONE"
            scene.spine2d_local_angle_limit = 30.0
            scene.spine2d_seam_maker_mode = "AUTO"
            scene.spine2d_frames_for_render = 0
            scene.spine2d_bake_frame_start = 0
            self.report({"INFO"}, "Spine2D settings have been reset.")
            return {"FINISHED"}
        except Exception as exc:
            logger.exception("[UI] Reset failed")
            self.report({"ERROR"}, f"Reset error: {exc}")
            return {"CANCELLED"}


class OBJECT_OT_Spine2DRefreshInfo(bpy.types.Operator):
    """Recalculates and caches expensive UI data."""

    bl_idname = "object.spine2d_refresh_info"
    bl_label = "Refresh Object Info"

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        return bool(context.active_object and context.active_object.type == "MESH")

    def execute(self, context: bpy.types.Context) -> Set[str]:
        obj = context.active_object
        if obj is None:
            return {"CANCELLED"}

        obj["_spine2d_vertex_count"] = len(obj.data.vertices)
        inverted, correct = OBJECT_PT_Spine2DMeshPanel._face_orientation_stats(obj)
        obj["_spine2d_face_stats"] = {"inverted": inverted, "correct": correct}

        for material in obj.data.materials:
            if material is None:
                continue
            try:
                if hasattr(material, "preview_ensure"):
                    material.preview_ensure()
                preview = getattr(material, "preview", None)
                material["_spine2d_icon_id"] = getattr(preview, "icon_id", 0) or 0
            except Exception:
                logger.exception("[UI] Material preview failed for %s", material.name)
                material["_spine2d_icon_id"] = 0

        self.report({"INFO"}, "Object info cache has been updated.")
        return {"FINISHED"}


class Spine2DConnectSettings(bpy.types.PropertyGroup):
    """Per-object setting for the connected multi-object subgroup."""

    enabled: bpy.props.BoolProperty(
        name="Connect",
        description="Attach this object to the shared all_objects rig",
        default=False,
    )


class Spine2DBakeSettings(bpy.types.PropertyGroup):
    """Per-object sequence-baking settings."""

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
    bl_label = "Blender to Spine2D Mesh Exporter"
    bl_idname = "OBJECT_PT_spine2d_mesh"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Blender to Spine2D Mesh Exporter"

    @staticmethod
    def _scale_applied(obj: bpy.types.Object, tol: float = 1e-4) -> bool:
        return all(abs(value - 1.0) < tol for value in obj.scale)

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
            header.label(text="Settings:")
            header.operator("spine2d.reset_settings", text="Reset")

            self._draw_foldout(
                layout,
                scene,
                prop_name="spine2d_show_settings",
                title="Export",
                draw_content=self._draw_export_settings,
            )
            self._draw_foldout(
                layout,
                scene,
                prop_name="spine2d_show_cut_settings",
                title="Cut",
                draw_content=self._draw_cut_settings,
            )
            self._draw_foldout(
                layout,
                scene,
                prop_name="spine2d_show_bake_settings",
                title="Bake",
                draw_content=self._draw_bake_settings,
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
            selected_meshes = [
                candidate
                for candidate in context.selected_objects
                if candidate.type == "MESH"
            ]
            if len(selected_meshes) <= 1:
                row.operator("object.save_uv_as_json", text="Export Current Object")
            else:
                row.operator(
                    "object.spine2d_multi_export",
                    text="Export Selected Objects",
                )
        except Exception:
            logger.exception("[ERROR in draw panel]")
            layout.label(text="UI error (see console)", icon="ERROR")

    def _populate_info_box(
        self,
        box: bpy.types.UILayout,
        obj: bpy.types.Object,
    ) -> bool:
        if obj is None:
            box.label(text="No active object", icon="ERROR")
            return False
        if obj.type != "MESH":
            box.label(text="Active object is not a Mesh", icon="ERROR")
            return False

        vertex_count = obj.get("_spine2d_vertex_count")
        if vertex_count is None:
            box.label(text="Vertex count: Press Refresh", icon="QUESTION")
        else:
            box.label(text=f"Vertex count: {vertex_count}", icon="INFO")

        export_ok = True
        if not self._scale_applied(obj):
            box.label(
                text="Scale is not applied (Apply > All Transforms)",
                icon="ERROR",
            )
            export_ok = False

        self._list_materials(box, obj)
        stats = obj.get("_spine2d_face_stats")
        if stats:
            inverted = int(stats.get("inverted", 0))
            correct = int(stats.get("correct", 0))
            if inverted == 0:
                box.label(text="All faces oriented correctly", icon="INFO")
            else:
                box.label(
                    text=f"Inverted faces: {inverted} / {inverted + correct}",
                    icon="ERROR",
                )
        else:
            box.label(text="Face orientation: Press Refresh", icon="QUESTION")
        return export_ok

    @staticmethod
    def _list_materials(box: bpy.types.UILayout, obj: bpy.types.Object) -> None:
        materials = [material for material in obj.data.materials if material]
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

    @staticmethod
    def _face_orientation_stats(obj: bpy.types.Object) -> tuple[int, int]:
        try:
            mesh = obj.data
            matrix_world = obj.matrix_world
            matrix_3x3 = matrix_world.to_3x3()
            origin = matrix_world.translation
            inverted = 0
            correct = 0
            for polygon in mesh.polygons:
                center_world = matrix_world @ polygon.center
                normal_world = (matrix_3x3 @ polygon.normal).normalized()
                direction = center_world - origin
                if direction.length == 0.0:
                    continue
                if direction.normalized().dot(normal_world) < 0:
                    inverted += 1
                else:
                    correct += 1
            return inverted, correct
        except Exception:
            logger.exception("[_face_orientation_stats] failed")
            return 0, 0

    def _draw_foldout(
        self,
        layout,
        scene,
        *,
        prop_name: str,
        title: str,
        draw_content,
    ) -> None:
        box = layout.box()
        row = box.row()
        icon = "TRIA_DOWN" if getattr(scene, prop_name) else "TRIA_RIGHT"
        row.prop(scene, prop_name, icon=icon, text="", icon_only=True, emboss=False)
        row.label(text=title)
        if getattr(scene, prop_name):
            draw_content(box.column(align=True), scene)

    def _draw_export_settings(
        self,
        column: bpy.types.UILayout,
        scene: bpy.types.Scene,
    ) -> None:
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

        selected_meshes = [
            candidate
            for candidate in bpy.context.selected_objects
            if candidate.type == "MESH"
        ]
        if len(selected_meshes) > 1:
            column.separator()
            column.prop(
                scene,
                MULTI_BACKEND_PROPERTY,
                text="Multi Export Engine",
            )
            column.label(text="Connect objects:")
            for selected_object in selected_meshes:
                row = column.row(align=True)
                row.label(text=selected_object.name, icon="MESH_DATA")
                row.prop(selected_object.spine2d_connect_settings, "enabled", text="")

    def _draw_cut_settings(
        self,
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

    def _draw_bake_settings(
        self,
        column: bpy.types.UILayout,
        scene: bpy.types.Scene,
    ) -> None:
        selected_meshes = [
            candidate
            for candidate in bpy.context.selected_objects
            if candidate.type == "MESH"
        ]
        if len(selected_meshes) > 1:
            for selected_object in selected_meshes:
                box = column.box()
                box.label(text=selected_object.name, icon="MESH_DATA")
                box.prop(
                    selected_object.spine2d_bake_settings,
                    "frames_for_render",
                    text="Frames",
                )
                box.prop(
                    selected_object.spine2d_bake_settings,
                    "bake_frame_start",
                    text="Start",
                )
                start = max(
                    0,
                    int(selected_object.spine2d_bake_settings.bake_frame_start),
                )
                frames = max(
                    0,
                    int(selected_object.spine2d_bake_settings.frames_for_render),
                )
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


class OBJECT_OT_Spine2DMultiExport(bpy.types.Operator):
    """Exports selected Mesh objects into one Spine JSON."""

    bl_idname = "object.spine2d_multi_export"
    bl_label = "Export Selected Objects"

    def _report_rewrite_result(self, result) -> Set[str]:
        for issue in result.issues:
            severity = issue.severity.value
            self.report({severity}, f"{issue.code}: {issue.message}")
        if result.success:
            destination = result.output_files[0] if result.output_files else "completed"
            self.report({"INFO"}, f"Export finished → {destination}")
            return {"FINISHED"}
        if not result.issues:
            self.report({"ERROR"}, "Rewrite export failed without diagnostics")
        return {"CANCELLED"}

    def execute(self, context: bpy.types.Context) -> Set[str]:
        scene = context.scene
        backend = resolve_multi_backend(scene)
        try:
            if backend == "LEGACY":
                texture_size = max(2, int(scene.spine2d_texture_size))
                output_path = export_selected_objects(
                    texture_size,
                    texture_size,
                    scene.spine2d_json_path,
                )
                if output_path:
                    self.report({"INFO"}, f"Legacy export finished → {output_path}")
                    return {"FINISHED"}
                self.report({"ERROR"}, "Legacy export failed (see console)")
                return {"CANCELLED"}

            result = export_selected_objects_a1(context)
            return self._report_rewrite_result(result)
        except Exception as exc:
            logger.exception("[UI] Multi-export failed with an unhandled exception")
            self.report({"ERROR"}, f"Multi-export failed: {exc}")
            return {"CANCELLED"}


SCENE_PROPERTIES = (
    (
        "spine2d_show_settings",
        bpy.props.BoolProperty(
            name="Show Settings",
            default=False,
            description="Show/hide Spine2D export settings",
        ),
    ),
    (
        "spine2d_show_cut_settings",
        bpy.props.BoolProperty(
            name="Show Cut Settings",
            default=False,
            description="Show/hide cutting parameters",
        ),
    ),
    (
        "spine2d_show_bake_settings",
        bpy.props.BoolProperty(
            name="Show Bake Settings",
            default=False,
            description="Show/hide baking parameters",
        ),
    ),
    (
        "spine2d_angular_mode",
        bpy.props.EnumProperty(
            name="Angular mode",
            description="Choose seed-normal compatibility or add a local dihedral guard",
            items=(
                (
                    "LEGACY_SEED_CONE",
                    "Seed cone (legacy)",
                    "Compare every candidate only with the segment seed normal",
                ),
                (
                    "SEED_CONE_AND_LOCAL_DIHEDRAL",
                    "Seed cone + local dihedral",
                    "Keep the seed cone and reject traversal across locally sharp edges",
                ),
            ),
            default="LEGACY_SEED_CONE",
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
    (
        MULTI_BACKEND_PROPERTY,
        bpy.props.EnumProperty(
            name="Multi Export Engine",
            description="Select the rewritten transactional exporter or the legacy fallback",
            items=(
                (
                    "REWRITE",
                    "Rewrite",
                    "Typed in-memory composition with atomic JSON and texture commit",
                ),
                (
                    "LEGACY",
                    "Legacy",
                    "Previous intermediate-JSON exporter kept as an explicit fallback",
                ),
            ),
            default=DEFAULT_MULTI_BACKEND,
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
    """Register all UI classes and properties as one transaction."""

    registered_classes = register_classes_transactionally(
        CLASSES,
        register_class=bpy.utils.register_class,
        unregister_class=bpy.utils.unregister_class,
    )
    try:
        register_rna_properties_transactionally(RNA_PROPERTIES)
    except Exception as exc:
        logger.exception("[ERROR] UI RNA registration failed")
        unregister_all_best_effort(
            class_cleanup_actions(
                registered_classes,
                unregister_class=bpy.utils.unregister_class,
            ),
            operation="UI registration rollback",
            primary_error=exc,
        )
        raise
    logger.debug("UI: Panel & operators registered.")


def unregister() -> None:
    """Remove every UI property and class before reporting aggregate failures."""

    try:
        unregister_all_best_effort(
            (
                *rna_property_cleanup_actions(RNA_PROPERTIES),
                *class_cleanup_actions(
                    CLASSES,
                    unregister_class=bpy.utils.unregister_class,
                ),
            ),
            operation="UI unregistration",
        )
    except Exception:
        logger.exception("[ERROR] UI unregistration failed")
        raise
    logger.debug("UI: Panel & operators unregistered.")


__all__ = [
    "CLASSES",
    "DEFAULT_MULTI_BACKEND",
    "MULTI_BACKEND_PROPERTY",
    "RNA_PROPERTIES",
    "register",
    "resolve_multi_backend",
    "unregister",
]
