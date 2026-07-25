"""Registered Blender 5.2+ single-object Rewrite export operator."""

from __future__ import annotations

import logging
from typing import Set

import bpy

from .blender_adapter.a1_ui_bridge import export_active_object_a1
from .infrastructure.blender_registration import (
    class_cleanup_actions,
    register_classes_transactionally,
    unregister_all_best_effort,
)


logger = logging.getLogger(__name__)


class OBJECT_OT_SaveUVAsJSON(bpy.types.Operator):
    """Export the active Mesh through the Rewrite pipeline."""

    bl_idname = "object.save_uv_as_json"
    bl_label = "Export current object"
    bl_description = "Export the active Mesh to Spine JSON with the Rewrite pipeline"

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        obj = getattr(context, "active_object", None)
        return bool(
            obj is not None
            and getattr(obj, "type", None) == "MESH"
            and getattr(obj, "data", None) is not None
        )

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

    def execute(self, context: bpy.types.Context) -> Set[str]:
        try:
            self._active_mesh(context)
            logger.info("[SaveUVAsJSON] Start Rewrite single-object export")
            result = export_active_object_a1(context)
            return self._report_rewrite_result(result)
        except Exception as exc:
            logger.exception("[SaveUVAsJSON] Single-object Rewrite export failed")
            self.report({"ERROR"}, f"Single-object export failed: {exc}")
            return {"CANCELLED"}


CLASSES = (OBJECT_OT_SaveUVAsJSON,)
RNA_PROPERTIES: tuple[object, ...] = ()


def register() -> None:
    """Register the Rewrite single-object operator transactionally."""

    try:
        register_classes_transactionally(
            CLASSES,
            register_class=bpy.utils.register_class,
            unregister_class=bpy.utils.unregister_class,
        )
    except Exception:
        logger.exception("Failed to register single-object Rewrite operator")
        raise
    logger.debug("Single-object Rewrite operator registered")


def unregister() -> None:
    """Remove every owned operator class even when one cleanup fails."""

    try:
        unregister_all_best_effort(
            class_cleanup_actions(
                CLASSES,
                unregister_class=bpy.utils.unregister_class,
            ),
            operation="single-object operator unregistration",
        )
    except Exception:
        logger.exception("Single-object Rewrite operator unregistration failed")
        raise
    logger.debug("Single-object Rewrite operator unregistered")


__all__ = [
    "CLASSES",
    "OBJECT_OT_SaveUVAsJSON",
    "RNA_PROPERTIES",
    "register",
    "unregister",
]
