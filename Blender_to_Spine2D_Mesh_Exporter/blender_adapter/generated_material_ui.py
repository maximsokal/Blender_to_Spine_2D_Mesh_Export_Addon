"""Rewrite-only generated material controls and Scene RNA registration."""

from __future__ import annotations

import logging

import bpy

from ..domain.baking.generated_materials import (
    A1GeneratedMaterialPattern,
    A1MaterialSourcePolicy,
)
from ..infrastructure.blender_registration import (
    RnaPropertyRegistration,
    class_cleanup_actions,
    register_classes_transactionally,
    register_rna_properties_transactionally,
    rna_property_cleanup_actions,
    unregister_all_best_effort,
)


logger = logging.getLogger(__name__)

MATERIAL_SOURCE_POLICY_PROPERTY = "spine2d_material_source_policy"
GENERATED_MATERIAL_PATTERN_PROPERTY = "spine2d_generated_material_pattern"
GENERATED_GRAY_COLOR_PROPERTY = "spine2d_generated_gray_color"


class SPINE2D_OT_ResetGeneratedMaterials(bpy.types.Operator):
    """Reset Rewrite generated-material controls without touching source materials."""

    bl_idname = "spine2d.reset_generated_materials"
    bl_label = "Reset Generated Materials"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context: bpy.types.Context) -> set[str]:
        scene = context.scene
        setattr(
            scene,
            MATERIAL_SOURCE_POLICY_PROPERTY,
            A1MaterialSourcePolicy.REQUIRE_SOURCE.value,
        )
        setattr(
            scene,
            GENERATED_MATERIAL_PATTERN_PROPERTY,
            A1GeneratedMaterialPattern.SOLID_GRAY.value,
        )
        setattr(scene, GENERATED_GRAY_COLOR_PROPERTY, (0.5, 0.5, 0.5))
        self.report({"INFO"}, "Generated material settings have been reset.")
        return {"FINISHED"}


class OBJECT_PT_Spine2DGeneratedMaterials(bpy.types.Panel):
    """Configure source-material fallback for the Rewrite exporter only."""

    bl_label = "Rewrite Generated Materials"
    bl_idname = "OBJECT_PT_spine2d_generated_materials"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Blender to Spine2D Mesh Exporter"
    bl_parent_id = "OBJECT_PT_spine2d_mesh"
    bl_options = {"DEFAULT_CLOSED"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        active = context.active_object
        return bool(active is not None and active.type == "MESH")

    def draw(self, context: bpy.types.Context) -> None:
        scene = context.scene
        column = self.layout.column(align=True)
        header = column.row(align=True)
        header.label(text="Rewrite material policy")
        header.operator(
            "spine2d.reset_generated_materials",
            text="Reset",
        )
        column.prop(
            scene,
            MATERIAL_SOURCE_POLICY_PROPERTY,
            text="Material source",
        )
        policy = str(
            getattr(
                scene,
                MATERIAL_SOURCE_POLICY_PROPERTY,
                A1MaterialSourcePolicy.REQUIRE_SOURCE.value,
            )
        ).upper()
        generated_column = column.column(align=True)
        generated_column.enabled = (
            policy != A1MaterialSourcePolicy.REQUIRE_SOURCE.value
        )
        generated_column.prop(
            scene,
            GENERATED_MATERIAL_PATTERN_PROPERTY,
            text="Generated pattern",
        )
        pattern = str(
            getattr(
                scene,
                GENERATED_MATERIAL_PATTERN_PROPERTY,
                A1GeneratedMaterialPattern.SOLID_GRAY.value,
            )
        ).upper()
        if pattern == A1GeneratedMaterialPattern.SOLID_GRAY.value:
            generated_column.prop(
                scene,
                GENERATED_GRAY_COLOR_PROPERTY,
                text="Gray color",
            )

        active = context.active_object
        material_count = len(
            tuple(
                material
                for material in getattr(getattr(active, "data", None), "materials", ())
                if material is not None
            )
        )
        if policy == A1MaterialSourcePolicy.REQUIRE_SOURCE.value:
            column.label(
                text="Missing materials stop Rewrite export",
                icon="ERROR" if material_count == 0 else "INFO",
            )
        elif policy == A1MaterialSourcePolicy.GENERATE_IF_MISSING.value:
            column.label(
                text=(
                    "Generated fallback will be used"
                    if material_count == 0
                    else "Fallback activates when used material data is missing"
                ),
                icon="CHECKMARK",
            )
        else:
            column.label(
                text="Source materials are ignored for Rewrite export",
                icon="INFO",
            )
        column.label(text="These controls are ignored by Legacy export", icon="INFO")


CLASSES = (
    SPINE2D_OT_ResetGeneratedMaterials,
    OBJECT_PT_Spine2DGeneratedMaterials,
)

RNA_PROPERTIES = (
    RnaPropertyRegistration(
        owner=bpy.types.Scene,
        name=MATERIAL_SOURCE_POLICY_PROPERTY,
        value=bpy.props.EnumProperty(
            name="Material Source",
            description=(
                "Require source materials, generate only when a used material is "
                "missing, or force a generated diagnostic material"
            ),
            items=(
                (
                    A1MaterialSourcePolicy.REQUIRE_SOURCE.value,
                    "Require Source",
                    "Keep strict Rewrite material validation",
                ),
                (
                    A1MaterialSourcePolicy.GENERATE_IF_MISSING.value,
                    "Generate If Missing",
                    "Generate one temporary material when used material data is missing",
                ),
                (
                    A1MaterialSourcePolicy.FORCE_GENERATED.value,
                    "Force Generated",
                    "Ignore source materials and use the selected diagnostic pattern",
                ),
            ),
            default=A1MaterialSourcePolicy.REQUIRE_SOURCE.value,
        ),
    ),
    RnaPropertyRegistration(
        owner=bpy.types.Scene,
        name=GENERATED_MATERIAL_PATTERN_PROPERTY,
        value=bpy.props.EnumProperty(
            name="Generated Pattern",
            description="Choose the temporary Rewrite diagnostic coloring",
            items=(
                (
                    A1GeneratedMaterialPattern.SOLID_GRAY.value,
                    "Solid Gray",
                    "Use one opaque gray color",
                ),
                (
                    A1GeneratedMaterialPattern.REGION_COLORS.value,
                    "One Region — One Color",
                    "Give every final exported region a deterministic color",
                ),
                (
                    A1GeneratedMaterialPattern.EXPORTED_FACE_COLORS.value,
                    "One Polygon — One Color",
                    "Give every final triangulated exported polygon a deterministic color",
                ),
            ),
            default=A1GeneratedMaterialPattern.SOLID_GRAY.value,
        ),
    ),
    RnaPropertyRegistration(
        owner=bpy.types.Scene,
        name=GENERATED_GRAY_COLOR_PROPERTY,
        value=bpy.props.FloatVectorProperty(
            name="Generated Gray",
            description="Opaque display RGB used by Solid Gray",
            subtype="COLOR_GAMMA",
            size=3,
            default=(0.5, 0.5, 0.5),
            min=0.0,
            max=1.0,
        ),
    ),
)


def register() -> None:
    """Register generated-material Scene properties and the child panel."""

    registered_classes = register_classes_transactionally(
        CLASSES,
        register_class=bpy.utils.register_class,
        unregister_class=bpy.utils.unregister_class,
    )
    try:
        register_rna_properties_transactionally(RNA_PROPERTIES)
    except Exception as exc:
        logger.exception("Generated material RNA registration failed")
        unregister_all_best_effort(
            class_cleanup_actions(
                registered_classes,
                unregister_class=bpy.utils.unregister_class,
            ),
            operation="generated material registration rollback",
            primary_error=exc,
        )
        raise


def unregister() -> None:
    """Remove every generated-material property and panel."""

    unregister_all_best_effort(
        (
            *rna_property_cleanup_actions(RNA_PROPERTIES),
            *class_cleanup_actions(
                CLASSES,
                unregister_class=bpy.utils.unregister_class,
            ),
        ),
        operation="generated material UI unregistration",
    )


__all__ = [
    "CLASSES",
    "GENERATED_GRAY_COLOR_PROPERTY",
    "GENERATED_MATERIAL_PATTERN_PROPERTY",
    "MATERIAL_SOURCE_POLICY_PROPERTY",
    "OBJECT_PT_Spine2DGeneratedMaterials",
    "RNA_PROPERTIES",
    "SPINE2D_OT_ResetGeneratedMaterials",
    "register",
    "unregister",
]
