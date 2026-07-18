"""Shared final-document and grouped-camera statistics for A1 output routes."""

from __future__ import annotations

from typing import Tuple

from ..application import GroupedCameraOverlayResult
from ..domain.spine import SpineDocument
from .a1_object_preparation import PreparedA1Object, StatisticsValue
from .grouped_camera_projection_executor import GroupedCameraProjectionStageResult
from .grouped_camera_projection_policy import GroupedCameraProjectionRequest


def record_final_document_statistics(
    target: dict[str, StatisticsValue],
    document: SpineDocument,
    finalized_objects: Tuple[PreparedA1Object, ...],
    *,
    grouped_enabled: bool,
) -> None:
    """Record statistics common to multi and mixed final documents."""

    if not isinstance(target, dict):
        raise TypeError("target must be dict")
    if not isinstance(document, SpineDocument):
        raise TypeError("document must be SpineDocument")
    if not isinstance(finalized_objects, tuple) or not finalized_objects:
        raise ValueError("finalized_objects must be a non-empty tuple")
    if not all(isinstance(item, PreparedA1Object) for item in finalized_objects):
        raise TypeError("finalized_objects must contain PreparedA1Object values")
    if not isinstance(grouped_enabled, bool):
        raise TypeError("grouped_enabled must be bool")

    target.update(
        {
            "final_bone_count": len(document.bones),
            "final_slot_count": len(document.slots),
            "final_skin_count": len(document.skins),
            "final_constraint_count": len(document.ik) + len(document.transform),
            "projection_cropped_component_count": sum(
                1
                for item in finalized_objects
                if "projection_crop_width" in item.statistics
            ),
            "grouped_b4_enabled": int(grouped_enabled),
        }
    )


def record_grouped_camera_statistics(
    target: dict[str, StatisticsValue],
    request: GroupedCameraProjectionRequest,
    staged: GroupedCameraProjectionStageResult,
    overlay: GroupedCameraOverlayResult,
) -> None:
    """Record grouped-camera output statistics after overlay application."""

    if not isinstance(target, dict):
        raise TypeError("target must be dict")
    if not isinstance(request, GroupedCameraProjectionRequest):
        raise TypeError("request must be GroupedCameraProjectionRequest")
    if not isinstance(staged, GroupedCameraProjectionStageResult):
        raise TypeError("staged must be GroupedCameraProjectionStageResult")
    if not isinstance(overlay, GroupedCameraOverlayResult):
        raise TypeError("overlay must be GroupedCameraOverlayResult")
    target.update(
        {
            "grouped_b4_source_count": len(request.plan.source_object_ids),
            "grouped_b4_frame_count": len(request.plan.frame_tasks),
            "grouped_b4_crop_width": staged.layout.cropped_width,
            "grouped_b4_crop_height": staged.layout.cropped_height,
            "grouped_b4_contour_vertex_count": len(staged.layout.hull),
            "grouped_b4_hidden_slot_count": len(overlay.hidden_slot_names),
        }
    )


__all__ = [
    "record_final_document_statistics",
    "record_grouped_camera_statistics",
]
