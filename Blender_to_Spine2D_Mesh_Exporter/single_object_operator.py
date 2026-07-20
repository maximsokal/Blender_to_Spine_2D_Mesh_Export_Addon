"""Registered single-object operator preserving the public Blender operator ID."""

from __future__ import annotations

import logging
from typing import Set

import bpy

from . import config
from .blender_adapter.a1_ui_bridge import export_active_object_a1
from .config import get_texture_size
from .infrastructure.blender_registration import (
    RnaPropertyRegistration,
    class_cleanup_actions,
    register_classes_transactionally,
    register_rna_properties_transactionally,
    rna_property_cleanup_actions,
    unregister_all_best_effort,
)
from .legacy_loader import load_legacy_single_backend


logger = logging.getLogger(__name__)
SINGLE_BACKEND_PROPERTY = "spine2d_single_export_backend"
DEFAULT_SINGLE_BACKEND = "REWRITE"
_SINGLE_BACKENDS = frozenset({"REWRITE", "LEGACY"})


def resolve_single_backend(scene: bpy.types.Scene) -> str:
    """Resolve the single-object backend and fail closed to Rewrite."""

    raw_value = getattr(scene, SINGLE_BACKEND_PROPERTY, DEFAULT_SINGLE_BACKEND)
    backend = str(raw_value).strip().upper()
    if backend in _SINGLE_BACKENDS:
        return backend
    logger.warning(
        "Unknown single export backend %r; using %s",
        raw_value,
        DEFAULT_SINGLE_BACKEND,
    )
    return DEFAULT_SINGLE_BACKEND


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
        backend = load_legacy_single_backend()

        # Preserve the old synchronized globals only after explicit Legacy selection.
        backend.main.TEXTURE_WIDTH = width
        backend.main.TEXTURE_HEIGHT = height
        config.TEXTURE_WIDTH = width
        config.TEXTURE_HEIGHT = height
        backend.json_export.TEXTURE_WIDTH = width
        backend.json_export.TEXTURE_HEIGHT = height

        output_path = backend.main.save_uv_as_json(
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
        backend = resolve_single_backend(context.scene)
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
        backend = resolve_single_backend(context.scene)
        if backend == "LEGACY":
            column.label(text="Legacy intermediate-JSON pipeline", icon="ERROR")
        else:
            column.label(text="Atomic typed A1 pipeline", icon="CHECKMARK")


CLASSES = (
    OBJECT_OT_SaveUVAsJSON,
    OBJECT_PT_Spine2DSingleExportBackend,
)

RNA_PROPERTIES = (
    RnaPropertyRegistration(
        owner=bpy.types.Scene,
        name=SINGLE_BACKEND_PROPERTY,
        value=bpy.props.EnumProperty(
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
            default=DEFAULT_SINGLE_BACKEND,
        ),
    ),
)


def register() -> None:
    """Register classes and Scene backend property as one transaction."""

    registered_classes = register_classes_transactionally(
        CLASSES,
        register_class=bpy.utils.register_class,
        unregister_class=bpy.utils.unregister_class,
    )
    try:
        register_rna_properties_transactionally(RNA_PROPERTIES)
    except Exception as exc:
        logger.exception("Failed to register single-object operator RNA")
        unregister_all_best_effort(
            class_cleanup_actions(
                registered_classes,
                unregister_class=bpy.utils.unregister_class,
            ),
            operation="single-object registration rollback",
            primary_error=exc,
        )
        raise
    logger.debug("Single-object Rewrite/Legacy operator registered")


def unregister() -> None:
    """Remove every owned property and class even when one cleanup fails."""

    try:
        unregister_all_best_effort(
            (
                *rna_property_cleanup_actions(RNA_PROPERTIES),
                *class_cleanup_actions(
                    CLASSES,
                    unregister_class=bpy.utils.unregister_class,
                ),
            ),
            operation="single-object operator unregistration",
        )
    except Exception:
        logger.exception("Single-object operator unregistration failed")
        raise
    logger.debug("Single-object Rewrite/Legacy operator unregistered")


__all__ = [
    "CLASSES",
    "DEFAULT_SINGLE_BACKEND",
    "OBJECT_OT_SaveUVAsJSON",
    "OBJECT_PT_Spine2DSingleExportBackend",
    "RNA_PROPERTIES",
    "SINGLE_BACKEND_PROPERTY",
    "register",
    "resolve_single_backend",
    "unregister",
]
