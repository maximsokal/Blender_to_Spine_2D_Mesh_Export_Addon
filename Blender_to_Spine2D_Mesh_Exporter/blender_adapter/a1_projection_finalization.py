"""Rebuild prepared B4 Spine attachments after render-derived layout analysis."""

from __future__ import annotations

from dataclasses import replace

from ..application import assemble_a1_camera_projection_document
from ..domain.baking import CameraProjectionPlan, resolve_projection_output_policy
from ..domain.baking.projection_layout import CameraProjectionLayout
from .a1_preparation_contracts import (
    PreparedA1Object,
    build_skeleton_metadata,
    freeze_statistics,
)


def finalize_prepared_camera_projection(
    prepared: PreparedA1Object,
    layout: CameraProjectionLayout | None,
) -> PreparedA1Object:
    """Return a prepared object whose document matches the staged cropped render."""

    if not isinstance(prepared, PreparedA1Object):
        raise TypeError("prepared must be PreparedA1Object")
    camera_projection = isinstance(prepared.bake_plan, CameraProjectionPlan)
    if not camera_projection:
        if layout is not None:
            raise ValueError("object-bake preparation cannot accept a projection layout")
        return prepared
    if not isinstance(layout, CameraProjectionLayout):
        raise TypeError("camera projection preparation requires CameraProjectionLayout")

    plan = prepared.bake_plan
    assert isinstance(plan, CameraProjectionPlan)
    output_policy = resolve_projection_output_policy(
        prepared.settings.bake_execution.projection_output_policy,
        plan.settings.texture_format,
    )
    document_assembly = assemble_a1_camera_projection_document(
        prepared.rig,
        prepared.z_groups,
        plan,
        prepared.document_assembly.settings,
        layout=layout,
        skeleton_metadata=build_skeleton_metadata(prepared.settings),
    )
    document = document_assembly.document
    offset_x, offset_y = layout.offset_pixels
    statistics = freeze_statistics(
        prepared.statistics,
        {
            "projection_full_width": layout.full_width,
            "projection_full_height": layout.full_height,
            "projection_crop_min_x": layout.crop.minimum_x,
            "projection_crop_min_y": layout.crop.minimum_y,
            "projection_crop_max_x": layout.crop.maximum_x,
            "projection_crop_max_y": layout.crop.maximum_y,
            "projection_crop_width": layout.cropped_width,
            "projection_crop_height": layout.cropped_height,
            "projection_offset_x": offset_x,
            "projection_offset_y": offset_y,
            "projection_hull_vertex_count": len(layout.hull),
            "projection_contour_vertex_count": len(layout.hull),
            "projection_source_contour_vertex_count": layout.source_contour_vertex_count,
            "projection_contour_mode": layout.contour_mode.value,
            "projection_contour_concave": int(layout.concave),
            "projection_outer_component_count": layout.outer_component_count,
            "projection_contour_fallback_reason": layout.contour_fallback_reason or "",
            "projection_contour_simplify_tolerance_pixels": layout.simplify_tolerance_pixels,
            "projection_union_visible_pixels": layout.visible_pixel_count,
            "projection_alpha_threshold": layout.alpha_threshold,
            "projection_padding_pixels": layout.padding_pixels,
            "projection_coverage_mode": layout.coverage_mode.value,
            "projection_coverage_core_alpha_threshold": layout.coverage_core_alpha_threshold,
            "projection_coverage_raw_nonzero_pixels": layout.coverage_raw_nonzero_pixel_count,
            "projection_coverage_strong_pixels": layout.coverage_strong_pixel_count,
            "projection_coverage_components_before_cleanup": layout.coverage_component_count_before_cleanup,
            "projection_coverage_components_after_cleanup": layout.coverage_component_count_after_cleanup,
            "projection_coverage_removed_component_pixels": layout.coverage_removed_component_pixel_count,
            "projection_coverage_filled_hole_pixels": layout.coverage_filled_hole_pixel_count,
            "projection_coverage_used_weak_only_fallback": int(
                layout.coverage_used_weak_only_fallback
            ),
            "projection_output_texture_format": output_policy.texture_format.value,
            "projection_output_dynamic_range": output_policy.dynamic_range.value,
            "projection_output_tone_mapping": output_policy.tone_mapping.value,
            "projection_output_alpha_representation": output_policy.alpha_representation.value,
            "projection_output_color_depth": output_policy.color_depth,
            "projection_output_float_buffer": int(output_policy.float_buffer),
            "final_bone_count": len(document.bones),
            "slot_count": len(document.slots),
            "attachment_count": sum(
                len(attachments)
                for skin in document.skins
                for attachments in skin.attachments.values()
            ),
        },
    )
    return replace(
        prepared,
        document_assembly=document_assembly,
        statistics=statistics,
    )


__all__ = ["finalize_prepared_camera_projection"]
