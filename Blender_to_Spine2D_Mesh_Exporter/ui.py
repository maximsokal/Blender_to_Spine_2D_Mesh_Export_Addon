# pylint: disable=import-error
"""Blender 5.2+ Rewrite UI, readiness analysis, and export operators."""

from __future__ import annotations

import logging
from math import isfinite, sqrt
import os
from typing import Callable, Set

import bpy

from .application import A1ReadinessState, IssueSeverity
from .blender_adapter.a1_export_readiness import (
    a1_readiness_depsgraph_update_post,
    analyse_a1_export_readiness,
    clear_a1_export_readiness,
    current_a1_export_readiness,
    require_current_a1_export_readiness,
    store_a1_export_readiness,
)
from .blender_adapter.a1_ui_bridge import (
    export_active_object_a1,
    export_selected_objects_a1,
)
from .config import get_default_output_dir
from .domain.baking import A1TextureExportMode
from .infrastructure.blender_registration import (
    RegistrationCleanupAction,
    RnaPropertyRegistration,
    class_cleanup_actions,
    register_classes_transactionally,
    register_rna_properties_transactionally,
    rna_property_cleanup_actions,
    unregister_all_best_effort,
)


logger = logging.getLogger(__name__)


def _tag_redraw(context: bpy.types.Context) -> None:
    area = getattr(context, "area", None)
    tag_redraw = getattr(area, "tag_redraw", None)
    if callable(tag_redraw):
        tag_redraw()


def _readiness_handler_cleanup_action() -> RegistrationCleanupAction:
    def remove() -> None:
        handlers = bpy.app.handlers.depsgraph_update_post
        while a1_readiness_depsgraph_update_post in handlers:
            handlers.remove(a1_readiness_depsgraph_update_post)

    return RegistrationCleanupAction(
        label="A1 readiness depsgraph handler",
        callback=remove,
    )


class SPINE2D_OT_ResetSettings(bpy.types.Operator):
    """Reset Rewrite export settings to their defaults."""

    bl_idname = "spine2d.reset_settings"
    bl_label = "Reset Spine2D Settings"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context: bpy.types.Context) -> Set[str]:
        try:
            scene = context.scene
            scene.spine2d_texture_export_mode = (
                A1TextureExportMode.NORMAL_UV_SEGMENTS.value
            )
            scene.spine2d_texture_size = 1024
            scene.spine2d_json_path = get_default_output_dir()
            scene.spine2d_images_path = "images/"
            scene.spine2d_control_icons = False
            scene.spine2d_export_preview_animation = False
            scene.spine2d_angle_limit = 30
            scene.spine2d_angular_mode = "SEED_CONE"
            scene.spine2d_local_angle_limit = 30.0
            scene.spine2d_seam_maker_mode = "AUTO"
            scene.spine2d_frames_for_render = 0
            scene.spine2d_bake_frame_start = 0
            clear_a1_export_readiness(scene)
            _tag_redraw(context)
            self.report({"INFO"}, "Spine2D Rewrite settings have been reset.")
            return {"FINISHED"}
        except Exception as exc:
            logger.exception("Unable to reset Spine2D Rewrite settings")
            self.report({"ERROR"}, f"Reset error: {exc}")
            return {"CANCELLED"}


class OBJECT_OT_Spine2DRefreshInfo(bpy.types.Operator):
    """Run production preparation and cache a file-free readiness report."""

    bl_idname = "object.spine2d_refresh_info"
    bl_label = "Analyze Export Readiness"
    bl_description = (
        "Run the Rewrite geometry, UV, material, rig, and composition pipeline "
        "without writing export files"
    )

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        active = getattr(context, "active_object", None)
        if active is not None and getattr(active, "type", None) == "MESH":
            return True
        return any(
            getattr(obj, "type", None) == "MESH"
            for obj in getattr(context, "selected_objects", ())
        )

    def execute(self, context: bpy.types.Context) -> Set[str]:
        try:
            report = analyse_a1_export_readiness(context)
            store_a1_export_readiness(context, report)
            _tag_redraw(context)
            if report.state is A1ReadinessState.BLOCKED:
                self.report(
                    {"ERROR"},
                    f"Export blocked: {report.blocker_count} blocker(s), "
                    f"{report.warning_count} warning(s)",
                )
            elif report.state is A1ReadinessState.WARNING:
                self.report(
                    {"WARNING"},
                    f"Export ready with {report.warning_count} warning(s)",
                )
            else:
                self.report({"INFO"}, "Export readiness analysis passed")
            return {"FINISHED"}
        except Exception as exc:
            logger.exception("Unable to analyze Spine2D export readiness")
            self.report({"ERROR"}, f"Analyze error: {exc}")
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
        first_length = sqrt(values[0] ** 2 + values[3] ** 2 + values[6] ** 2)
        second_length = sqrt(values[1] ** 2 + values[4] ** 2 + values[7] ** 2)
        third_length = sqrt(values[2] ** 2 + values[5] ** 2 + values[8] ** 2)
        scale_product = first_length * second_length * third_length
        relative_determinant = (
            abs(determinant) / scale_product
            if isfinite(scale_product) and scale_product > 0.0
            else 0.0
        )
        is_singular = (
            not isfinite(determinant)
            or not isfinite(relative_determinant)
            or relative_determinant <= tolerance
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
        """Compatibility diagnostic; orientation never acts as an export blocker."""

        try:
            mesh = obj.data
            matrix_world = obj.matrix_world
            linear = matrix_world.to_3x3()
            determinant = float(linear.determinant())
            orientation_sign = -1.0 if determinant < 0.0 else 1.0
            normal_matrix = linear.inverted_safe().transposed()
            origin = matrix_world.translation
            inverted = 0
            correct = 0
            for polygon in mesh.polygons:
                center_world = matrix_world @ polygon.center
                normal_world = (
                    (normal_matrix @ polygon.normal) * orientation_sign
                ).normalized()
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

    @staticmethod
    def _issue_icon(severity: IssueSeverity) -> str:
        if severity is IssueSeverity.ERROR:
            return "CANCEL"
        if severity is IssueSeverity.WARNING:
            return "ERROR"
        return "INFO"

    @staticmethod
    def _state_icon(state: A1ReadinessState) -> str:
        if state is A1ReadinessState.READY:
            return "CHECKMARK"
        if state is A1ReadinessState.WARNING:
            return "ERROR"
        if state is A1ReadinessState.BLOCKED:
            return "CANCEL"
        return "QUESTION"

    def _draw_object_readiness(self, layout: bpy.types.UILayout, item) -> None:
        box = layout.box()
        box.label(
            text=f"{item.object_id}: {item.state.value}",
            icon=self._state_icon(item.state),
        )
        statistics = item.statistics
        source_vertices = int(statistics.get("source_vertices", 0))
        source_faces = int(statistics.get("source_faces", 0))
        exported_vertices = int(statistics.get("exported_attachment_vertices", 0))
        triangles = int(statistics.get("triangles_after_triangulation", 0))
        regions = int(statistics.get("region_count", 0))
        bones = int(statistics.get("final_bone_count", 0))
        attachments = int(statistics.get("attachment_count", 0))
        box.label(text=f"Source: {source_vertices} vertices / {source_faces} faces")
        box.label(text=f"Export: {exported_vertices} vertices / {triangles} triangles")
        box.label(text=f"Rig: {bones} bones / {regions} regions / {attachments} attachments")
        mode = str(statistics.get("texture_export_mode", ""))
        if mode:
            box.label(text=f"Mode: {mode}")
        pipeline = str(statistics.get("texture_pipeline", ""))
        if pipeline:
            frames = int(statistics.get("bake_frame_count", 0))
            materials = int(statistics.get("material_slot_count", 0))
            box.label(text=f"Texture: {pipeline} / {frames} frame(s) / {materials} material(s)")
        topology_parts = []
        for key, label in (
            ("non_manifold_edges", "non-manifold"),
            ("loose_vertices", "loose vertices"),
            ("loose_edges", "loose edges"),
            ("decomposition_cut_count", "cuts"),
        ):
            value = int(statistics.get(key, 0))
            if value:
                topology_parts.append(f"{label}: {value}")
        if topology_parts:
            box.label(text="Topology: " + ", ".join(topology_parts), icon="INFO")
        for issue in item.issues[:6]:
            box.label(
                text=f"{issue.code}: {issue.message}",
                icon=self._issue_icon(issue.severity),
            )
        if len(item.issues) > 6:
            box.label(text=f"... and {len(item.issues) - 6} more issue(s)")

    def _draw_readiness(
        self,
        layout: bpy.types.UILayout,
        context: bpy.types.Context,
    ) -> bool:
        state, report = current_a1_export_readiness(context)
        box = layout.box()
        row = box.row(align=True)
        row.label(text="Export readiness:")
        row.operator(
            "object.spine2d_refresh_info",
            text="Analyze",
            icon="VIEWZOOM",
        )

        if state is A1ReadinessState.NOT_ANALYSED:
            box.label(text="Not analyzed", icon="QUESTION")
            box.label(text="Run Analyze before export")
            return False
        if state is A1ReadinessState.STALE:
            box.label(text="Analysis outdated", icon="FILE_REFRESH")
            box.label(text="Selection, geometry, material, scene, or settings changed")
            return False
        if report is None:
            box.label(text="Analysis cache unavailable", icon="CANCEL")
            return False

        box.label(
            text=(
                f"{state.value}: {report.blocker_count} blocker(s), "
                f"{report.warning_count} warning(s)"
            ),
            icon=self._state_icon(state),
        )
        for issue in report.issues[:6]:
            box.label(
                text=f"{issue.code}: {issue.message}",
                icon=self._issue_icon(issue.severity),
            )
        for item in report.objects:
            self._draw_object_readiness(box, item)
        return report.can_export

    def draw(self, context: bpy.types.Context) -> None:
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
            export_allowed = self._draw_readiness(layout, context)
            row = layout.row()
            row.enabled = export_allowed
            selected_meshes = tuple(
                candidate
                for candidate in context.selected_objects
                if candidate.type == "MESH"
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
        except Exception:
            logger.exception("Unable to draw the Spine2D Rewrite panel")
            layout.label(text="UI error (see console)", icon="ERROR")


class _Spine2DExportOperatorMixin:
    def _require_readiness(self, context: bpy.types.Context) -> bool:
        allowed, message = require_current_a1_export_readiness(context)
        if not allowed:
            self.report({"ERROR"}, message)
        return allowed

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


class OBJECT_OT_Spine2DSingleExport(
    _Spine2DExportOperatorMixin,
    bpy.types.Operator,
):
    """Export the active Mesh through the Rewrite pipeline."""

    bl_idname = "object.spine2d_single_export"
    bl_label = "Export Current Object"
    bl_description = "Export the active Mesh with the Rewrite pipeline"

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        obj = getattr(context, "active_object", None)
        return bool(
            obj is not None
            and getattr(obj, "type", None) == "MESH"
            and getattr(obj, "data", None) is not None
        )

    def execute(self, context: bpy.types.Context) -> Set[str]:
        if not self._require_readiness(context):
            return {"CANCELLED"}
        try:
            return self._report_result(export_active_object_a1(context))
        except Exception as exc:
            logger.exception("Rewrite single-object export failed")
            self.report({"ERROR"}, f"Single-object export failed: {exc}")
            return {"CANCELLED"}


class OBJECT_OT_Spine2DMultiExport(
    _Spine2DExportOperatorMixin,
    bpy.types.Operator,
):
    """Export selected Mesh objects through the Rewrite pipeline."""

    bl_idname = "object.spine2d_multi_export"
    bl_label = "Export Selected Objects"
    bl_description = "Export selected Mesh objects with the Rewrite pipeline"

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        return any(
            getattr(obj, "type", None) == "MESH"
            and getattr(obj, "data", None) is not None
            for obj in getattr(context, "selected_objects", ())
        )

    def execute(self, context: bpy.types.Context) -> Set[str]:
        if not self._require_readiness(context):
            return {"CANCELLED"}
        try:
            return self._report_result(export_selected_objects_a1(context))
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
            default=False,
        ),
    ),
    (
        "spine2d_export_preview_animation",
        bpy.props.BoolProperty(
            name="Preview animation",
            description="Add a preview animation to the final JSON",
            default=False,
        ),
    ),
)


CLASSES = (
    Spine2DBakeSettings,
    Spine2DConnectSettings,
    SPINE2D_OT_ResetSettings,
    OBJECT_OT_Spine2DRefreshInfo,
    OBJECT_PT_Spine2DMeshPanel,
    OBJECT_OT_Spine2DSingleExport,
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
    """Register Rewrite UI, RNA properties, and readiness invalidation."""

    registered_classes = register_classes_transactionally(
        CLASSES,
        register_class=bpy.utils.register_class,
        unregister_class=bpy.utils.unregister_class,
    )
    registered_rna = ()
    handler_added = False
    try:
        registered_rna = register_rna_properties_transactionally(RNA_PROPERTIES)
        handlers = bpy.app.handlers.depsgraph_update_post
        if a1_readiness_depsgraph_update_post not in handlers:
            handlers.append(a1_readiness_depsgraph_update_post)
            handler_added = True
    except Exception as exc:
        logger.exception("Rewrite UI registration failed")
        actions = []
        if handler_added:
            actions.append(_readiness_handler_cleanup_action())
        actions.extend(rna_property_cleanup_actions(registered_rna))
        actions.extend(
            class_cleanup_actions(
                registered_classes,
                unregister_class=bpy.utils.unregister_class,
            )
        )
        unregister_all_best_effort(
            tuple(actions),
            operation="Rewrite UI registration rollback",
            primary_error=exc,
        )
        raise
    logger.debug("Rewrite UI registered")


def unregister() -> None:
    """Remove readiness cache, handler, RNA properties, and UI classes."""

    try:
        unregister_all_best_effort(
            (
                RegistrationCleanupAction(
                    label="A1 readiness cache",
                    callback=clear_a1_export_readiness,
                ),
                _readiness_handler_cleanup_action(),
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
    "OBJECT_OT_Spine2DSingleExport",
    "OBJECT_PT_Spine2DMeshPanel",
    "RNA_PROPERTIES",
    "SPINE2D_OT_ResetSettings",
    "Spine2DBakeSettings",
    "Spine2DConnectSettings",
    "register",
    "unregister",
]
