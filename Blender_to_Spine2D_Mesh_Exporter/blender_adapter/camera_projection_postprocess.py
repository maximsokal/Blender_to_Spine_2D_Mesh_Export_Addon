"""Build B4 sequence coverage/layout and rewrite staged images after rendering."""

from __future__ import annotations

import logging
from typing import Tuple

from ..domain.baking.projection_layout import (
    CameraProjectionLayout,
    CameraProjectionLayoutError,
    ProjectionAlphaUnionAccumulator,
    build_full_frame_layout,
)
from ..infrastructure import AtomicOutputReservation
from .camera_projection_error import CameraProjectionExecutionError
from .camera_projection_image import (
    read_staged_alpha_coverage,
    rewrite_staged_image_with_crop,
)
from .camera_projection_validation import (
    CameraProjectionRuntime,
    validate_camera_projection_reservations,
)


logger = logging.getLogger(__name__)


def build_projection_union_accumulator(
    runtime: CameraProjectionRuntime,
) -> ProjectionAlphaUnionAccumulator:
    """Create the fixed-size sequence union from one validated execution policy."""

    if not isinstance(runtime, CameraProjectionRuntime):
        raise TypeError("runtime must be CameraProjectionRuntime")

    execution_settings = runtime.execution_settings
    alpha_threshold = float(
        execution_settings.projection_alpha_threshold
    )
    return ProjectionAlphaUnionAccumulator(
        width=runtime.plan.settings.width,
        height=runtime.plan.settings.height,
        alpha_threshold=alpha_threshold,
        padding_pixels=runtime.plan.settings.margin_pixels,
        contour_mode=execution_settings.projection_contour_mode,
        simplify_tolerance_pixels=float(
            execution_settings.projection_contour_simplify_tolerance_pixels
        ),
        coverage_policy=execution_settings.projection_coverage_policy,
    )


def log_camera_projection_layout(
    runtime: CameraProjectionRuntime,
    layout: CameraProjectionLayout,
    accumulator: ProjectionAlphaUnionAccumulator,
) -> None:
    """Log the complete B4 crop, contour, coverage, and output-policy contract."""

    logger.info(
        "B4 union layout '%s': full=%dx%d crop=(%d,%d)-(%d,%d) "
        "size=%dx%d contour=%s vertices=%d source_vertices=%d "
        "outer_components=%d contour_fallback=%r coverage=%s "
        "raw_nonzero=%d strong=%d final_visible=%d components=%d->%d "
        "removed=%d filled_holes=%d weak_only=%s frames=%d union_bytes=%d "
        "fringe_threshold=%.8f core_threshold=%.8f simplify_tolerance=%.4f "
        "dynamic_range=%s tone_mapping=%s alpha=%s color_depth=%s",
        runtime.plan.source_object_id,
        layout.full_width,
        layout.full_height,
        layout.crop.minimum_x,
        layout.crop.minimum_y,
        layout.crop.maximum_x,
        layout.crop.maximum_y,
        layout.cropped_width,
        layout.cropped_height,
        layout.contour_mode.value,
        len(layout.hull),
        layout.source_contour_vertex_count,
        layout.outer_component_count,
        layout.contour_fallback_reason,
        layout.coverage_mode.value,
        layout.coverage_raw_nonzero_pixel_count,
        layout.coverage_strong_pixel_count,
        layout.visible_pixel_count,
        layout.coverage_component_count_before_cleanup,
        layout.coverage_component_count_after_cleanup,
        layout.coverage_removed_component_pixel_count,
        layout.coverage_filled_hole_pixel_count,
        layout.coverage_used_weak_only_fallback,
        accumulator.frame_count,
        accumulator.allocated_mask_bytes,
        layout.alpha_threshold,
        layout.coverage_core_alpha_threshold,
        layout.simplify_tolerance_pixels,
        runtime.output_policy.dynamic_range.value,
        runtime.output_policy.tone_mapping.value,
        runtime.output_policy.alpha_representation.value,
        runtime.output_policy.color_depth,
    )


def process_camera_projection_outputs(
    runtime: CameraProjectionRuntime,
    reservations: Tuple[AtomicOutputReservation, ...],
    *,
    apply_crop: bool,
) -> CameraProjectionLayout:
    """Decode completed renders and derive one stable layout outside Scene state."""

    if not isinstance(runtime, CameraProjectionRuntime):
        raise TypeError("runtime must be CameraProjectionRuntime")
    if not isinstance(apply_crop, bool):
        raise TypeError("apply_crop must be bool")

    resolved = validate_camera_projection_reservations(
        runtime.plan,
        reservations,
    )

    if not apply_crop:
        return build_full_frame_layout(
            runtime.plan.settings.width,
            runtime.plan.settings.height,
            frame_count=len(runtime.plan.frame_tasks),
        )

    accumulator = build_projection_union_accumulator(runtime)
    for task, reservation in zip(
        runtime.plan.frame_tasks,
        resolved,
        strict=True,
    ):
        coverage = read_staged_alpha_coverage(
            runtime.bpy_module,
            reservation.staged_path,
            width=runtime.plan.settings.width,
            height=runtime.plan.settings.height,
        )
        newly_visible = accumulator.add_coverage(
            coverage,
            frame_index=task.task_index,
        )
        del coverage
        logger.debug(
            "Merged B4 projection frame %d into alpha coverage union: "
            "new_nonzero=%d raw_nonzero_total=%d",
            task.task_index,
            newly_visible,
            accumulator.visible_pixel_count,
        )

    try:
        layout = accumulator.build_layout()
    except CameraProjectionLayoutError as exc:
        raise CameraProjectionExecutionError(str(exc)) from exc

    for reservation in resolved:
        rewrite_staged_image_with_crop(
            runtime.bpy_module,
            runtime.plan,
            reservation,
            layout,
            runtime.output_policy,
        )

    log_camera_projection_layout(runtime, layout, accumulator)
    return layout


__all__ = [
    "CameraProjectionExecutionError",
    "build_projection_union_accumulator",
    "log_camera_projection_layout",
    "process_camera_projection_outputs",
]
