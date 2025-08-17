# ui.py
# pylint: disable=import-error
"""
Blender to Spine2D Mesh Exporter
Copyright (c) 2025 Maxim Sokolenko

This file is part of Blender to Spine2D Mesh Exporter.

Blender to Spine2D Mesh Exporter is free software: you can redistribute it
and/or modify it under the terms of the GNU General Public License as
published by the Free Software Foundation, either version 3 of the License,
or (at your option) any later version.

Blender to Spine2D Mesh Exporter is distributed in the hope that it will
be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with Blender to Spine2D Mesh Exporter. If not, see <https://www.gnu.org/licenses/>.
This module is responsible for creating the user interface (UI) for the 'Blender_to_Spine2D_Mesh_Exporter' addon within Blender's 3D View. It defines the main panel, operators (buttons), and properties that allow the user to control the export process.

The key functionalities are:
1.  Main UI Panel: It defines the `OBJECT_PT_Spine2DMeshPanel`, which is the primary interface for the addon. This panel is organized into collapsible sections for Export, Cut, and Bake settings, providing a clean and organized user experience.
2.  Export Operators: It creates the operators that trigger the export process. `OBJECT_OT_SaveUVAsJSON` handles the export of a single active object, while `OBJECT_OT_Spine2DMultiExport` is responsible for exporting multiple selected objects.
3.  Settings and Properties: It defines and registers various properties that are displayed in the UI. This includes global settings on the Scene (like texture size, angle limit, and output paths) as well as per-object settings (`Spine2DConnectSettings` and `Spine2DBakeSettings`) for controlling multi-object exports and sequence baking.
4.  Dynamic Information Display: The panel dynamically displays contextual information about the selected object, such as its vertex count, material list, and whether its scale has been applied. It includes a "Refresh" button (`OBJECT_OT_Spine2DRefreshInfo`) to update this information.
5.  User Feedback and Validation: The UI provides immediate feedback to the user. It disables the export button if critical conditions are not met (e.g., the .blend file is not saved, or the object's scale is not applied) and uses icons to indicate errors or warnings.
6.  Reset Functionality: It includes a `SPINE2D_OT_ResetSettings` operator that allows the user to easily reset all addon settings to their default values.

ATTENTION: - The UI code is tightly coupled with the properties defined in `config.py` and the export functions in `main.py` and `multi_object_export.py`. Changing property names (`bl_idname`) or the logic within the `execute` methods of the operators will break the connection between the UI and the addon's core functionality. The `draw()` method can be performance-sensitive, so expensive calculations are offloaded to the "Refresh" operator to avoid UI lag.
Author: Maxim Sokolenko
"""
import bpy
import logging
from typing import Set
import os

logger = logging.getLogger(__name__)
from .config import get_default_output_dir
from .multi_object_export import export_selected_objects

logger.debug("[LOG] Loading ui.py")

# --- Properties will be registered in the register() function ---


class SPINE2D_OT_ResetSettings(bpy.types.Operator):
    """Resets the addon settings to their default values"""

    bl_idname = "spine2d.reset_settings"
    bl_label = "Reset Spine2D Settings"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context: bpy.types.Context) -> Set[str]:
        try:
            scene = context.scene
            # ► export-settings
            scene.spine2d_texture_size = 1024
            scene.spine2d_json_path = get_default_output_dir()
            scene.spine2d_images_path = "./images/"
            scene.spine2d_control_icons = True
            scene.spine2d_export_preview_animation = True
            # ► cut-settings
            scene.spine2d_angle_limit = 30
            scene.spine2d_seam_maker_mode = "AUTO"
            # ► bake-settings
            scene.spine2d_frames_for_render = 0
            scene.spine2d_bake_frame_start = 0
            self.report({"INFO"}, "Spine2D settings have been reset.")
            return {"FINISHED"}
        except Exception as e:
            self.report({"ERROR"}, f"Reset error: {e}")
            return {"CANCELLED"}


class OBJECT_OT_Spine2DRefreshInfo(bpy.types.Operator):
    """Recalculates and caches expensive UI data like face orientation and vertex count"""

    bl_idname = "object.spine2d_refresh_info"
    bl_label = "Refresh Object Info"

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        return context.active_object and context.active_object.type == "MESH"

    def execute(self, context: bpy.types.Context) -> Set[str]:
        obj = context.active_object
        if not obj:
            return {"CANCELLED"}

        # Cache vertex count
        obj["_spine2d_vertex_count"] = len(obj.data.vertices)

        # Cache face orientation stats
        inverted, correct = OBJECT_PT_Spine2DMeshPanel._face_orientation_stats(obj)
        obj["_spine2d_face_stats"] = {"inverted": inverted, "correct": correct}

        # Cache material preview icons
        for mat in obj.data.materials:
            if mat:
                if hasattr(mat, "preview_ensure"):
                    mat.preview_ensure()
                mat["_spine2d_icon_id"] = getattr(mat.preview, "icon_id", 0)

        self.report({"INFO"}, "Object info cache has been updated.")
        return {"FINISHED"}


class Spine2DConnectSettings(bpy.types.PropertyGroup):
    """Per-object setting to connect it to the main skeleton during multi-export."""

    enabled: bpy.props.BoolProperty(
        name="Connect",
        description="Mark this object to be attached to others during multi-export",
        default=False,
    )


class Spine2DBakeSettings(bpy.types.PropertyGroup):
    """Per-object settings for sequence baking."""

    frames_for_render: bpy.props.IntProperty(
        name="Frames for render",
        description="How many frames to render (0 = current frame only)",
        default=0,
        min=0,
    )
    bake_frame_start: bpy.props.IntProperty(
        name="Start frame", description="First frame of the sequence", default=0, min=0
    )


class OBJECT_PT_Spine2DMeshPanel(bpy.types.Panel):
    bl_label = "Blender to Spine2D Mesh Exporter"
    bl_idname = "OBJECT_PT_spine2d_mesh"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Blender to Spine2D Mesh Exporter"

    @staticmethod
    def _scale_applied(obj: bpy.types.Object, tol: float = 1e-4) -> bool:
        """Returns True if the object's local scale is approximately (1,1,1)."""
        return all(abs(v - 1.0) < tol for v in obj.scale)

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
                "object.spine2d_refresh_info", text="Refresh", icon="FILE_REFRESH"
            )

            export_allowed = self._populate_info_box(info_box, obj)

            row = layout.row()
            row.enabled = export_allowed
            sel_meshes = [o for o in context.selected_objects if o.type == "MESH"]
            if len(sel_meshes) <= 1:
                row.operator("object.save_uv_as_json", text="Export Current Object")
            else:
                row.operator(
                    "object.spine2d_multi_export", text="Export Selected Objects"
                )

        except Exception:
            logger.exception("[ERROR in draw panel]")
            layout.label(text="UI error (see console)", icon="ERROR")

    def _populate_info_box(
        self, box: bpy.types.UILayout, obj: bpy.types.Object
    ) -> bool:
        """Fills the info box with data about the object. Returns True if export is allowed."""
        if obj is None:
            box.label(text="No active object", icon="ERROR")
            return False

        if obj.type != "MESH":
            box.label(text="Active object is not a Mesh", icon="ERROR")
            return False

        # Read cached vertex count
        vert_count = obj.get("_spine2d_vertex_count")
        if vert_count is not None:
            box.label(text=f"Vertex count: {vert_count}", icon="INFO")
        else:
            box.label(text="Vertex count: Press Refresh", icon="QUESTION")

        export_ok = True
        if not self._scale_applied(obj):
            box.label(
                text="Scale is not applied (Apply > All Transforms)", icon="ERROR"
            )
            export_ok = False

        self._list_materials(box, obj)

        # Read cached face orientation stats
        stats = obj.get("_spine2d_face_stats")
        if stats:
            inverted, correct = stats.get("inverted", 0), stats.get("correct", 0)
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
        """Lists object materials with cached preview icons."""
        mats = [m for m in obj.data.materials if m]
        if not mats:
            box.label(text="No materials", icon="ERROR")
            return

        col = box.column(align=True)
        col.label(text=f"Materials ({len(mats)}):")

        for mat in mats:
            row = col.row(align=True)
            icon_id = mat.get("_spine2d_icon_id", 0)
            if icon_id:
                row.label(text=mat.name, icon_value=icon_id)
            else:
                row.label(text=mat.name, icon="MATERIAL")

    @staticmethod
    def _face_orientation_stats(obj: bpy.types.Object) -> tuple[int, int]:
        """Calculates face orientation stats. Should be called on demand, not in draw()."""
        try:
            mesh = obj.data
            mw = obj.matrix_world
            mw3 = mw.to_3x3()
            inverted = correct = 0
            origin = mw.translation
            for poly in mesh.polygons:
                center_world = mw @ poly.center
                normal_world = (mw3 @ poly.normal).normalized()
                to_center_dir = (center_world - origin).normalized()
                if to_center_dir.length == 0.0:
                    continue
                if to_center_dir.dot(normal_world) < 0:
                    inverted += 1
                else:
                    correct += 1
            return inverted, correct
        except Exception:
            logger.exception("[_face_orientation_stats] failed")
            return 0, 0

    def _draw_foldout(
        self, layout, scene, *, prop_name: str, title: str, draw_content
    ) -> None:
        """Draws a collapsible box."""
        box = layout.box()
        row = box.row()
        icon = "TRIA_DOWN" if getattr(scene, prop_name) else "TRIA_RIGHT"
        row.prop(scene, prop_name, icon=icon, text="", icon_only=True, emboss=False)
        row.label(text=title)
        if getattr(scene, prop_name):
            col = box.column(align=True)
            draw_content(col, scene)

    def _draw_export_settings(
        self, col: bpy.types.UILayout, scene: bpy.types.Scene
    ) -> None:
        col.prop(scene, "spine2d_texture_size", text="Texture size")
        col.separator()

        # --- JSON Path ---
        col.prop(scene, "spine2d_json_path", text="JSON")
        json_full_path = bpy.path.abspath(scene.spine2d_json_path)
        if not json_full_path or json_full_path == bpy.path.abspath("//"):
            json_full_path = get_default_output_dir()
        col.label(text=json_full_path)
        col.separator()

        # --- Images Path (теперь зависит от JSON) ---
        col.prop(scene, "spine2d_images_path", text="Images Subfolder")
        # Вычисляем и показываем полный путь, но не даем его редактировать
        images_full_path = os.path.join(json_full_path, scene.spine2d_images_path)
        col.label(text=os.path.normpath(images_full_path))
        col.separator()

        row = col.row(align=True)
        row.label(text="Control icons")
        row.prop(scene, "spine2d_control_icons", text="")
        row = col.row(align=True)
        row.label(text="Preview animation")
        row.prop(scene, "spine2d_export_preview_animation", text="")
        sel_meshes = [o for o in bpy.context.selected_objects if o.type == "MESH"]
        if len(sel_meshes) > 1:
            col.separator()
            col.label(text="Connect objects:")
            for ob in sel_meshes:
                row = col.row(align=True)
                row.label(text=ob.name, icon="MESH_DATA")
                row.prop(ob.spine2d_connect_settings, "enabled", text="")

    def _draw_cut_settings(
        self, col: bpy.types.UILayout, scene: bpy.types.Scene
    ) -> None:
        col.prop(scene, "spine2d_angle_limit", text="Angle limit")
        col.separator()
        col.prop(scene, "spine2d_seam_maker_mode", text="Seam maker")

    def _draw_bake_settings(
        self, col: bpy.types.UILayout, scene: bpy.types.Scene
    ) -> None:
        sel_meshes = [o for o in bpy.context.selected_objects if o.type == "MESH"]
        if len(sel_meshes) > 1:
            for ob in sel_meshes:
                box = col.box()
                box.label(text=ob.name, icon="MESH_DATA")
                box.prop(ob.spine2d_bake_settings, "frames_for_render", text="Frames")
                row = box.row(align=True)
                row.prop(ob.spine2d_bake_settings, "bake_frame_start", text="Start")
                start = max(0, int(ob.spine2d_bake_settings.bake_frame_start))
                frames = max(0, int(ob.spine2d_bake_settings.frames_for_render))
                last = start if frames == 0 else start + frames - 1
                box.label(text=f"Last frame: {last}")
        else:
            col.prop(scene, "spine2d_frames_for_render", text="Frames for render")
            row = col.row(align=True)
            row.prop(scene, "spine2d_bake_frame_start", text="Start")
            start = max(0, int(scene.spine2d_bake_frame_start))
            frames = max(0, int(scene.spine2d_frames_for_render))
            last = start if frames == 0 else start + frames - 1
            col.label(text=f"Last frame: {last}")
            col.label(text=f"Playback end: {scene.frame_end}")


class OBJECT_OT_Spine2DMultiExport(bpy.types.Operator):
    """Exports all selected Mesh objects into a single Spine JSON"""

    bl_idname = "object.spine2d_multi_export"
    bl_label = "Export Selected Objects"

    def execute(self, context: bpy.types.Context) -> Set[str]:
        scene = context.scene
        tex_size = int(scene.spine2d_texture_size)
        tex_w = tex_h = max(2, tex_size)
        try:
            out_path = export_selected_objects(tex_w, tex_h, scene.spine2d_json_path)
            if out_path:
                self.report({"INFO"}, f"Export finished → {out_path}")
                return {"FINISHED"}
            self.report({"ERROR"}, "Export failed (see console for details)")
            return {"CANCELLED"}
        except Exception:
            # FIXED: Use logger.exception to include traceback for better debugging
            logger.exception("[UI] Multi-export failed with an unhandled exception")
            self.report({"ERROR"}, "Multi-export failed. Check console for details.")
            return {"CANCELLED"}


# --- Registration ---

SCENE_PROPERTIES = [
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
]

CLASSES = [
    Spine2DBakeSettings,
    Spine2DConnectSettings,
    SPINE2D_OT_ResetSettings,
    OBJECT_OT_Spine2DRefreshInfo,
    OBJECT_PT_Spine2DMeshPanel,
    OBJECT_OT_Spine2DMultiExport,
]


def register() -> None:
    try:
        for cls in CLASSES:
            bpy.utils.register_class(cls)

        for name, prop in SCENE_PROPERTIES:
            setattr(bpy.types.Scene, name, prop)

        bpy.types.Object.spine2d_bake_settings = bpy.props.PointerProperty(
            type=Spine2DBakeSettings
        )
        bpy.types.Object.spine2d_connect_settings = bpy.props.PointerProperty(
            type=Spine2DConnectSettings
        )

        logger.debug("UI: Panel & operators registered.")
    except Exception:
        logger.exception("[ERROR] UI registration failed")


def unregister() -> None:
    try:
        if hasattr(bpy.types.Object, "spine2d_bake_settings"):
            del bpy.types.Object.spine2d_bake_settings
        if hasattr(bpy.types.Object, "spine2d_connect_settings"):
            del bpy.types.Object.spine2d_connect_settings

        for name, _ in SCENE_PROPERTIES:
            if hasattr(bpy.types.Scene, name):
                delattr(bpy.types.Scene, name)

        for cls in reversed(CLASSES):
            bpy.utils.unregister_class(cls)

        logger.debug("UI: Panel & operators unregistered.")
    except Exception:
        logger.exception("[ERROR] UI unregistration failed")
