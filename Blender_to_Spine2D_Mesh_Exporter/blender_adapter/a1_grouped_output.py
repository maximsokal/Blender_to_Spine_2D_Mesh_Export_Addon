"""Apply a staged grouped-camera result to one typed Spine document."""

from __future__ import annotations

from ..application import (
    GroupedCameraOverlayResult,
    apply_grouped_camera_overlay,
)
from ..domain.spine import SpineDocument
from .grouped_camera_projection_output import (
    GroupedCameraProjectionStageResult,
)
from .grouped_camera_projection_policy import (
    GroupedCameraProjectionRequest,
)


def apply_staged_grouped_camera_overlay(
    document: SpineDocument,
    request: GroupedCameraProjectionRequest,
    staged: GroupedCameraProjectionStageResult,
) -> GroupedCameraOverlayResult:
    """Validate grouped render ownership and append its root-bound overlay."""

    if not isinstance(document, SpineDocument):
        raise TypeError("document must be SpineDocument")
    if not isinstance(request, GroupedCameraProjectionRequest):
        raise TypeError("request must be GroupedCameraProjectionRequest")
    if not isinstance(staged, GroupedCameraProjectionStageResult):
        raise TypeError("staged must be GroupedCameraProjectionStageResult")
    if staged.source_object_ids != request.plan.source_object_ids:
        raise ValueError("grouped stage source IDs do not match grouped request")
    return apply_grouped_camera_overlay(
        document,
        request.plan,
        staged.layout,
        visual_slot_names=request.visual_slot_names,
        image_relative_directory=request.image_relative_directory,
        slot_name=request.slot_name,
        attachment_name=request.attachment_name,
    )


__all__ = ["apply_staged_grouped_camera_overlay"]
