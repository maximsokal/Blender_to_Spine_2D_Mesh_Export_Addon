"""Registered single-object operator preserving the public Blender operator ID."""

from __future__ import annotations

import logging
from typing import Set

import bpy

from . import config, json_export, main as legacy_main
from .blender_adapter.a1_ui_bridge import export_active_object_a1
from .config import get_texture_size

logger = logging.getLogger(__name__)


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
            getattr(context.scene, "spine2d_single_export_backend", "LEGACY")
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


def register() -> None:
    bpy.utils.register_class(OBJECT_OT_SaveUVAsJSON)
    logger.debug("OBJECT_OT_SaveUVAsJSON registered from single_object_operator.py")


def unregister() -> None:
    bpy.utils.unregister_class(OBJECT_OT_SaveUVAsJSON)
    logger.debug("OBJECT_OT_SaveUVAsJSON unregistered from single_object_operator.py")
