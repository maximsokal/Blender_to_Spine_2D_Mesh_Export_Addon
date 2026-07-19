"""Shared single/grouped B4 coverage, layout, and staged-image postprocessing."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Iterable, Tuple, TypeAlias

from ..domain.baking import (
    BakeExecutionSettings,
    BakeFrameTask,
    BakeSettings,
    CameraProjectionPlan,
    GroupedCameraProjectionPlan,
    ResolvedProjectionOutputPolicy,
)
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
from .camera_projection_validation import CameraProjectionRuntime


logger = logging.getLogger(__name__)


ProjectionImagePlan: TypeAlias = (
    CameraProjectionPlan | GroupedCameraProjectionPlan
)


@dataclass(frozen=True, slots=True)
class ProjectionPostprocessRequest:
    """Blender-state-independent data required for one B4 output postprocess."""

    owner_id: str
    bpy_module: Any
    image_plan: ProjectionImagePlan
    settings: BakeSettings
    frame_tasks: Tuple[BakeFrameTask, ...]
    execution_settings: BakeExecutionSettings
    output_policy: ResolvedProjectionOutputPolicy

    def __post_init__(self) -> None:
        if not isinstance(self.owner_id, str) or not self.owner_id.strip():
            raise ValueError("owner_id must be a non-empty string")
        if self.bpy_module is None:
            raise ValueError("bpy_module cannot be None")
        if not isinstance(
            self.image_plan,
            (CameraProjectionPlan, GroupedCameraProjectionPlan),
        ):
            raise TypeError(
                "image_plan must be CameraProjectionPlan or "
                "GroupedCameraProjectionPlan"
            )
        if not isinstance(self.settings, BakeSettings):
            raise TypeError("settings must be BakeSettings")
        if self.settings != self.image_plan.settings:
            raise ValueError("settings must match image_plan.settings")
        if (
            not isinstance(self.frame_tasks, tuple)
            or not self.frame_tasks
            or not all(
                isinstance(task, BakeFrameTask)
                for task in self.frame_tasks
            )
        ):
            raise ValueError(
                "frame_tasks must be a non-empty tuple of BakeFrameTask"
            )
        if self.frame_tasks != self.image_plan.frame_tasks:
            raise ValueError("frame_tasks must match image_plan.frame_tasks")
        if not isinstance(
            self.execution_settings,
            BakeExecutionSettings,
        ):
            raise TypeError(
                "execution_settings must be BakeExecutionSettings"
            )
        if not isinstance(
            self.output_policy,
            ResolvedProjectionOutputPolicy,
        ):
            raise TypeError(
                "output_policy must be ResolvedProjectionOutputPolicy"
            )
        if (
            self.output_policy.texture_format
            is not self.settings.texture_format
        ):
            raise CameraProjectionExecutionError(
                "resolved output policy texture format does not match "
                "projection settings"
            )

        resolved_paths: list[Path] = []
        for expected_index, task in enumerate(self.frame_tasks):
            if task.task_index != expected_index:
                raise CameraProjectionExecutionError(
                    "projection postprocess frame indices must be contiguous; "
                    f"expected={expected_index}, got={task.task_index}"
                )
            resolved_paths.append(
                task.output_path.expanduser().resolve(strict=False)
            )
        if len(resolved_paths) != len(set(resolved_paths)):
            raise CameraProjectionExecutionError(
                "projection postprocess contains duplicate output paths"
            )


def validate_projection_postprocess_reservations(
    request: ProjectionPostprocessRequest,
    reservations: Iterable[AtomicOutputReservation],
) -> Tuple[AtomicOutputReservation, ...]:
    """Require one correctly ordered staged path for every postprocess frame."""

    if not isinstance(request, ProjectionPostprocessRequest):
        raise TypeError("request must be ProjectionPostprocessRequest")

    resolved = tuple(reservations)
    if len(resolved) != len(request.frame_tasks):
        raise CameraProjectionExecutionError(
            f"Expected {len(request.frame_tasks)} projection reservations, "
            f"got {len(resolved)}"
        )

    for task, reservation in zip(
        request.frame_tasks,
        resolved,
        strict=True,
    ):
        if not isinstance(reservation, AtomicOutputReservation):
            raise TypeError(
                "reservations must contain AtomicOutputReservation values"
            )
        expected = task.output_path.expanduser().resolve(strict=False)
        if reservation.final_path != expected:
            raise CameraProjectionExecutionError(
                f"Projection task {task.task_index} expected '{expected}', "
                f"got '{reservation.final_path}'"
            )

    return resolved


def build_camera_projection_postprocess_request(
    runtime: CameraProjectionRuntime,
) -> ProjectionPostprocessRequest:
    """Adapt the validated single-object B4 runtime to the shared engine."""

    if not isinstance(runtime, CameraProjectionRuntime):
        raise TypeError("runtime must be CameraProjectionRuntime")
    return ProjectionPostprocessRequest(
        owner_id=runtime.plan.source_object_id,
        bpy_module=runtime.bpy_module,
        image_plan=runtime.plan,
        settings=runtime.plan.settings,
        frame_tasks=runtime.plan.frame_tasks,
        execution_settings=runtime.execution_settings,
        output_policy=runtime.output_policy,
    )


def _coerce_postprocess_request(
    value: ProjectionPostprocessRequest | CameraProjectionRuntime,
) -> ProjectionPostprocessRequest:
    if isinstance(value, ProjectionPostprocessRequest):
        return value
    if isinstance(value, CameraProjectionRuntime):
        return build_camera_projection_postprocess_request(value)
    raise TypeError(
        "value must be ProjectionPostprocessRequest or CameraProjectionRuntime"
    )


def build_projection_union_accumulator(
    value: ProjectionPostprocessRequest | CameraProjectionRuntime,
) -> ProjectionAlphaUnionAccumulator:
    """Create one fixed-size sequence union from a validated B4 policy."""

    request = _coerce_postprocess_request(value)
    execution_settings = request.execution_settings
    alpha_threshold = float(
        execution_settings.projection_alpha_threshold
    )
    return ProjectionAlphaUnionAccumulator(
        width=request.settings.width,
        height=request.settings.height,
        alpha_threshold=alpha_threshold,
        padding_pixels=request.settings.margin_pixels,
        contour_mode=execution_settings.projection_contour_mode,
        simplify_tolerance_pixels=float(
            execution_settings.projection_contour_simplify_tolerance_pixels
        ),
        coverage_policy=execution_settings.projection_coverage_policy,
    )


def log_projection_layout(
    request: ProjectionPostprocessRequest,
    layout: CameraProjectionLayout,
    accumulator: ProjectionAlphaUnionAccumulator,
) -> None:
    """Log the complete shared B4 crop, coverage, contour, and output policy."""

    if not isinstance(request, ProjectionPostprocessRequest):
        raise TypeError("request must be ProjectionPostprocessRequest")
    logger.info(
        "B4 union layout '%s': full=%dx%d crop=(%d,%d)-(%d,%d) "
        "size=%dx%d contour=%s vertices=%d source_vertices=%d "
        "outer_components=%d contour_fallback=%r coverage=%s "
        "raw_nonzero=%d strong=%d final_visible=%d components=%d->%d "
        "removed=%d filled_holes=%d weak_only=%s frames=%d union_bytes=%d "
        "fringe_threshold=%.8f core_threshold=%.8f simplify_tolerance=%.4f "
        "dynamic_range=%s tone_mapping=%s alpha=%s color_depth=%s",
        request.owner_id,
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
        request.output_policy.dynamic_range.value,
        request.output_policy.tone_mapping.value,
        request.output_policy.alpha_representation.value,
        request.output_policy.color_depth,
    )


def process_projection_outputs(
    request: ProjectionPostprocessRequest,
    reservations: Tuple[AtomicOutputReservation, ...],
    *,
    apply_crop: bool,
) -> CameraProjectionLayout:
    """Decode completed renders and postprocess them outside Blender state scope."""

    if not isinstance(request, ProjectionPostprocessRequest):
        raise TypeError("request must be ProjectionPostprocessRequest")
    if not isinstance(apply_crop, bool):
        raise TypeError("apply_crop must be bool")

    resolved = validate_projection_postprocess_reservations(
        request,
        reservations,
    )
    if not apply_crop:
        return build_full_frame_layout(
            request.settings.width,
            request.settings.height,
            frame_count=len(request.frame_tasks),
        )

    accumulator = build_projection_union_accumulator(request)
    for task, reservation in zip(
        request.frame_tasks,
        resolved,
        strict=True,
    ):
        coverage = read_staged_alpha_coverage(
            request.bpy_module,
            reservation.staged_path,
            width=request.settings.width,
            height=request.settings.height,
        )
        newly_visible = accumulator.add_coverage(
            coverage,
            frame_index=task.task_index,
        )
        del coverage
        logger.debug(
            "Merged B4 projection '%s' frame %d into alpha coverage union: "
            "new_nonzero=%d raw_nonzero_total=%d",
            request.owner_id,
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
            request.bpy_module,
            request.image_plan,
            reservation,
            layout,
            request.output_policy,
        )

    log_projection_layout(request, layout, accumulator)
    return layout


def log_camera_projection_layout(
    runtime: CameraProjectionRuntime,
    layout: CameraProjectionLayout,
    accumulator: ProjectionAlphaUnionAccumulator,
) -> None:
    """Compatibility wrapper for historical single-B4 diagnostics."""

    request = build_camera_projection_postprocess_request(runtime)
    log_projection_layout(request, layout, accumulator)


def process_camera_projection_outputs(
    runtime: CameraProjectionRuntime,
    reservations: Tuple[AtomicOutputReservation, ...],
    *,
    apply_crop: bool,
) -> CameraProjectionLayout:
    """Preserve the single-B4 wrapper over the shared postprocess engine."""

    request = build_camera_projection_postprocess_request(runtime)
    return process_projection_outputs(
        request,
        reservations,
        apply_crop=apply_crop,
    )


__all__ = [
    "CameraProjectionExecutionError",
    "ProjectionImagePlan",
    "ProjectionPostprocessRequest",
    "build_camera_projection_postprocess_request",
    "build_projection_union_accumulator",
    "log_camera_projection_layout",
    "log_projection_layout",
    "process_camera_projection_outputs",
    "process_projection_outputs",
    "validate_projection_postprocess_reservations",
]
