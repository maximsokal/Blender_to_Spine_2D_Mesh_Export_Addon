"""Rebuild prepared B4 Spine attachments after render-derived layout analysis."""

from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType

from ..application import assemble_a1_camera_projection_document
from ..domain.baking import CameraProjectionPlan
from ..domain.baking.projection_layout import CameraProjectionLayout
from .a1_object_preparation import PreparedA1Object


def _skeleton_metadata(prepared: PreparedA1Object) -> dict[str, object]:
    settings = prepared.settings
    return {
        "hash": "hash_value_placeholder",
        "spine": settings.export.spine_version,
        "x": 0,
        "y": 0,
        "width": settings.export.texture_width,
        "height": settings.export.texture_height,
        "images": "",
        "audio": "./audio",
    }


def finalize_prepared_camera_projection(
    prepared: PreparedA1Object,
    layout: CameraProjectionLayout | None,
) -> PreparedA1Object:
    """Return a prepared object whose document matches the staged cropped render.

    Object-bake preparations pass through unchanged and must not receive a layout. Camera
    preparations require the exact layout returned by the render executor. Reassembly is pure
    application work: no Blender datablocks, operators or source material state are touched.
    """

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
    document_assembly = assemble_a1_camera_projection_document(
        prepared.rig,
        prepared.z_groups,
        plan,
        prepared.document_assembly.settings,
        layout=layout,
        skeleton_metadata=_skeleton_metadata(prepared),
    )
    document = document_assembly.document
    statistics = dict(prepared.statistics)
    offset_x, offset_y = layout.offset_pixels
    statistics.update(
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
            "projection_source_contour_vertex_count": (
                layout.source_contour_vertex_count
            ),
            "projection_contour_mode": layout.contour_mode.value,
            "projection_contour_concave": layout.concave,
            "projection_outer_component_count": layout.outer_component_count,
            "projection_contour_fallback_reason": (
                layout.contour_fallback_reason or ""
            ),
            "projection_contour_simplify_tolerance_pixels": (
                layout.simplify_tolerance_pixels
            ),
            "projection_union_visible_pixels": layout.visible_pixel_count,
            "projection_alpha_threshold": layout.alpha_threshold,
            "projection_padding_pixels": layout.padding_pixels,
            "final_bone_count": len(document.bones),
            "slot_count": len(document.slots),
            "attachment_count": sum(
                len(attachments)
                for skin in document.skins
                for attachments in skin.attachments.values()
            ),
        }
    )
    return replace(
        prepared,
        document_assembly=document_assembly,
        statistics=MappingProxyType(statistics),
    )


__all__ = ["finalize_prepared_camera_projection"]
