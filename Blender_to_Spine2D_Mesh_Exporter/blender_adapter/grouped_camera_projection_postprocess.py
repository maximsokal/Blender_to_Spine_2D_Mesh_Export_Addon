"""Adapt grouped B4 runtime values to the shared projection postprocess engine."""

from __future__ import annotations

import logging
from typing import Tuple

from ..domain.baking.projection_layout import CameraProjectionLayout
from ..infrastructure import AtomicOutputReservation
from .camera_projection_error import CameraProjectionExecutionError
from .camera_projection_postprocess import (
    ProjectionPostprocessRequest,
    process_projection_outputs,
)
from .grouped_camera_projection_validation import (
    GroupedCameraProjectionRuntime,
    validate_grouped_camera_projection_reservations,
)


logger = logging.getLogger(__name__)


def build_grouped_projection_postprocess_request(
    runtime: GroupedCameraProjectionRuntime,
) -> ProjectionPostprocessRequest:
    """Build one shared postprocess request from a validated grouped runtime."""

    if not isinstance(runtime, GroupedCameraProjectionRuntime):
        raise TypeError("runtime must be GroupedCameraProjectionRuntime")
    return ProjectionPostprocessRequest(
        owner_id=runtime.plan.group_id,
        bpy_module=runtime.bpy_module,
        image_plan=runtime.plan,
        settings=runtime.plan.settings,
        frame_tasks=runtime.plan.frame_tasks,
        execution_settings=runtime.execution_settings,
        output_policy=runtime.output_policy,
    )


def log_grouped_camera_projection_layout(
    runtime: GroupedCameraProjectionRuntime,
    layout: CameraProjectionLayout,
) -> None:
    """Log grouped ownership information not present in the shared layout log."""

    if not isinstance(runtime, GroupedCameraProjectionRuntime):
        raise TypeError("runtime must be GroupedCameraProjectionRuntime")
    if not isinstance(layout, CameraProjectionLayout):
        raise TypeError("layout must be CameraProjectionLayout")
    logger.info(
        "Grouped B4 layout '%s': sources=%d source_ids=%s camera='%s' "
        "crop=%dx%d contour=%s vertices=%d components=%d coverage=%s "
        "final_visible=%d dynamic_range=%s tone_mapping=%s alpha=%s",
        runtime.plan.group_id,
        len(runtime.plan.source_object_ids),
        runtime.plan.source_object_ids,
        runtime.plan.camera_object_id,
        layout.cropped_width,
        layout.cropped_height,
        layout.contour_mode.value,
        len(layout.hull),
        layout.outer_component_count,
        layout.coverage_mode.value,
        layout.visible_pixel_count,
        runtime.output_policy.dynamic_range.value,
        runtime.output_policy.tone_mapping.value,
        runtime.output_policy.alpha_representation.value,
    )


def process_grouped_camera_projection_outputs(
    runtime: GroupedCameraProjectionRuntime,
    reservations: Tuple[AtomicOutputReservation, ...],
) -> CameraProjectionLayout:
    """Build grouped coverage/layout only after reversible rendering has returned."""

    if not isinstance(runtime, GroupedCameraProjectionRuntime):
        raise TypeError("runtime must be GroupedCameraProjectionRuntime")
    resolved = validate_grouped_camera_projection_reservations(
        runtime.plan,
        reservations,
    )
    request = build_grouped_projection_postprocess_request(runtime)
    layout = process_projection_outputs(
        request,
        resolved,
        apply_crop=True,
    )
    log_grouped_camera_projection_layout(runtime, layout)
    return layout


__all__ = [
    "CameraProjectionExecutionError",
    "build_grouped_projection_postprocess_request",
    "log_grouped_camera_projection_layout",
    "process_grouped_camera_projection_outputs",
]
