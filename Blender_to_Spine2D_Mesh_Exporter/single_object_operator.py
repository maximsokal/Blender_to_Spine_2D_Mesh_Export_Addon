"""Registered single-object operator preserving the public Blender operator ID."""

from __future__ import annotations

import logging
from typing import Set

import bpy

from . import config, json_export, main as legacy_main
from .blender_adapter.a1_ui_bridge import export_active_object_a1
from .config import get_texture_size

logger = logging.getLogger(__name__)

SINGLE_BACKEND_PROPERTY = "spine2d_single_export_backend"


class OBJECT_OT_SaveUVAsJSON(bpy.types.Operator):
    """Export the active Mesh through Rewrite or the explicit Legacy backend."""

    bl_idname = "object.save_uv_as_json"
    bl_label = "Export current object"
    bl_description = "Exports the active object to Spine JSON"

    @staticmethod
    def _normalized_texture_size(scene: bpy.types.Scene) -> tuple[int, int]:
        size_or_tuple = get_texture_size(scene)
        if isinstance(size_or_tuple, (tuple, list)) and len(size_or_tuple) == 2:
            width, height = map(int, size_or_tuple)
        else:
            width = height = int(size_or_tuple)
        if width <= 0 or height <= 0:
            raise ValueError(f"Incorrect texture size: {width}×{height}")
        return width, height

    @staticmethod
    def _active_mesh(context: bpy.types.Context) -> bpy.types.Object:
        obj = context.active_object
        if obj is None:
            raise ValueError("There is no active object")
        if obj.type != "MESH":
            raise ValueError(f"The object '{obj.name}' is not a Mesh")
        if obj.data is None:
            raise ValueError(f"The Mesh object '{obj.name}' has no data")
        return obj

    def _report_rewrite_result(self, result) -> Set[str]:
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

    def _execute_legacy(self, context: bpy.types.Context) -> Set[str]:
        obj = self._active_mesh(context)
        width, height = self._normalized_texture_size(context.scene)

        # Preserve the old operator's synchronized global constants. The legacy
        # pipeline imports these values in several modules during add-on startup.
        legacy_main.TEXTURE_WIDTH = width
        legacy_main.TEXTURE_HEIGHT = height
        config.TEXTURE_WIDTH = width
        config.TEXTURE_HEIGHT = height
        json_export.TEXTURE_WIDTH = width
        json_export.TEXTURE_HEIGHT = height

        output_path = legacy_main.save_uv_as_json(
            obj,
            width,
            height,
            output_dir=context.scene.spine2d_json_path,
        )
        if not output_path:
            self.report({"ERROR"}, "Legacy export failed (see console)")
            return {"CANCELLED"}
        self.report({"INFO"}, f"Legacy export finished → {output_path}")
        return {"FINISHED"}

    def execute(self, context: bpy.types.Context) -> Set[str]:
        backend = str(
            getattr(context.scene, SINGLE_BACKEND_PROPERTY, "LEGACY")
        ).upper()
        logger.info("[SaveUVAsJSON] Start %s single-object export", backend)
        try:
            if backend == "LEGACY":
                return self._execute_legacy(context)
            result = export_active_object_a1(context)
            return self._report_rewrite_result(result)
        except Exception as exc:
            logger.exception("[SaveUVAsJSON] Single-object export failed")
            self.report({"ERROR"}, f"Single-object export failed: {exc}")
            return {"CANCELLED"}


class OBJECT_PT_Spine2DSingleExportBackend(bpy.types.Panel):
    """Explicit backend selection for the existing single-object export button."""

    bl_label = "Single Export Engine"
    bl_idname = "OBJECT_PT_spine2d_single_export_backend"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Blender to Spine2D Mesh Exporter"
    bl_parent_id = "OBJECT_PT_spine2d_mesh"
    bl_options = {"DEFAULT_CLOSED"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        active = context.active_object
        selected_meshes = tuple(
            obj for obj in context.selected_objects if obj.type == "MESH"
        )
        return bool(
            active is not None
            and active.type == "MESH"
            and len(selected_meshes) <= 1
        )

    def draw(self, context: bpy.types.Context) -> None:
        column = self.layout.column(align=True)
        column.prop(
            context.scene,
            SINGLE_BACKEND_PROPERTY,
            text="Engine",
        )
        backend = str(getattr(context.scene, SINGLE_BACKEND_PROPERTY, "REWRITE"))
        if backend == "LEGACY":
            column.label(text="Legacy intermediate-JSON pipeline", icon="ERROR")
        else:
            column.label(text="Atomic typed A1 pipeline", icon="CHECKMARK")


CLASSES = (
    OBJECT_OT_SaveUVAsJSON,
    OBJECT_PT_Spine2DSingleExportBackend,
)


def register() -> None:
    setattr(
        bpy.types.Scene,
        SINGLE_BACKEND_PROPERTY,
        bpy.props.EnumProperty(
            name="Single Export Engine",
            description=(
                "Select the rewritten transactional exporter or the explicit "
                "legacy fallback for the active object"
            ),
            items=(
                (
                    "REWRITE",
                    "Rewrite",
                    "Typed A1 export with atomic JSON and texture commit",
                ),
                (
                    "LEGACY",
                    "Legacy",
                    "Previous intermediate-JSON exporter kept for controlled fallback",
                ),
            ),
            default="REWRITE",
        ),
    )
    try:
        for cls in CLASSES:
            bpy.utils.register_class(cls)
    except Exception:
        if hasattr(bpy.types.Scene, SINGLE_BACKEND_PROPERTY):
            delattr(bpy.types.Scene, SINGLE_BACKEND_PROPERTY)
        logger.exception("Failed to register single-object operator UI")
        raise
    logger.debug("Single-object Rewrite/Legacy operator registered")


def unregister() -> None:
    errors: list[Exception] = []
    for cls in reversed(CLASSES):
        try:
            bpy.utils.unregister_class(cls)
        except Exception as exc:
            errors.append(exc)
            logger.exception("Failed to unregister %s", cls.__name__)
    if hasattr(bpy.types.Scene, SINGLE_BACKEND_PROPERTY):
        delattr(bpy.types.Scene, SINGLE_BACKEND_PROPERTY)
    if errors:
        raise RuntimeError(
            f"Single-object operator unregistration failed {len(errors)} time(s)"
        ) from errors[0]
    logger.debug("Single-object Rewrite/Legacy operator unregistered")
