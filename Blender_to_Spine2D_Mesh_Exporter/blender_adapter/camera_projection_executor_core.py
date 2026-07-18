"""Atomic B4 rendering, coverage union, contour layout, and crop orchestration."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Iterable, Tuple

from ..domain.baking import (
    BakeArtifact,
    BakeExecutionResult,
    BakeExecutionSettings,
    CameraProjectionPlan,
    resolve_projection_output_policy,
)
from ..domain.baking.projection_layout import (
    CameraProjectionLayout,
    CameraProjectionLayoutError,
    ProjectionAlphaUnionAccumulator,
    build_full_frame_layout,
)
from ..infrastructure import (
    AtomicFileTransaction,
    AtomicOutputReservation,
    atomic_file_transaction,
)
from .camera_projection_image import (
    read_staged_alpha_coverage,
    rewrite_staged_image_with_crop,
)
from .camera_projection_state import (
    CameraProjectionExecutionError,
    call_public_render_operator,
    configure_camera_visibility,
    configure_scene_for_camera_projection,
    preserve_camera_projection_state,
    require_reservations,
    set_timeline_frame,
    validate_projection_runtime,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CameraProjectionStageResult:
    reservations: Tuple[AtomicOutputReservation, ...]
    layout: CameraProjectionLayout

    def __post_init__(self) -> None:
        if not isinstance(self.reservations, tuple) or not self.reservations:
            raise ValueError("reservations must be a non-empty tuple")
        if not all(isinstance(item, AtomicOutputReservation) for item in self.reservations):
            raise TypeError("reservations must contain AtomicOutputReservation values")
        if not isinstance(self.layout, CameraProjectionLayout):
            raise TypeError("layout must be CameraProjectionLayout")
        if len(self.reservations) != self.layout.frame_count:
            raise ValueError("reservation count must match layout.frame_count")


def _render_to_reservations(
    source_obj: Any,
    plan: CameraProjectionPlan,
    execution_settings: BakeExecutionSettings,
    reservations: Tuple[AtomicOutputReservation, ...],
    *,
    context: Any | None,
    scene: Any | None,
    apply_crop: bool,
) -> CameraProjectionLayout:
    bpy_module, resolved_context, resolved_scene = validate_projection_runtime(
        source_obj,
        plan,
        context=context,
        scene=scene,
    )
    resolved = require_reservations(plan, reservations)
    output_policy = resolve_projection_output_policy(
        execution_settings.projection_output_policy,
        plan.settings.texture_format,
    )
    alpha_threshold = float(execution_settings.projection_alpha_threshold)
    union_accumulator = (
        ProjectionAlphaUnionAccumulator(
            width=plan.settings.width,
            height=plan.settings.height,
            alpha_threshold=alpha_threshold,
            padding_pixels=plan.settings.margin_pixels,
            contour_mode=execution_settings.projection_contour_mode,
            simplify_tolerance_pixels=float(
                execution_settings.projection_contour_simplify_tolerance_pixels
            ),
            coverage_policy=execution_settings.projection_coverage_policy,
        )
        if apply_crop
        else None
    )

    with preserve_camera_projection_state(resolved_scene):
        configure_camera_visibility(
            source_obj,
            resolved_scene,
            isolate=plan.isolate_source_to_camera,
        )
        for task, reservation in zip(plan.frame_tasks, resolved):
            set_timeline_frame(resolved_scene, resolved_context, task.timeline_frame)
            configure_scene_for_camera_projection(
                resolved_scene,
                plan,
                execution_settings,
                reservation.staged_path,
            )
            logger.info(
                "Rendering B4 projection '%s' frame %d/%d camera='%s' "
                "dynamic_range=%s tone_mapping=%s alpha=%s",
                plan.source_object_id,
                task.task_index + 1,
                len(plan.frame_tasks),
                plan.camera_object_id,
                output_policy.dynamic_range.value,
                output_policy.tone_mapping.value,
                output_policy.alpha_representation.value,
            )
            call_public_render_operator(bpy_module)
            if (
                not reservation.staged_path.is_file()
                or reservation.staged_path.stat().st_size <= 0
            ):
                raise CameraProjectionExecutionError(
                    f"Projection staged output is missing or empty: {reservation.staged_path}"
                )
            if union_accumulator is not None:
                coverage = read_staged_alpha_coverage(
                    bpy_module,
                    reservation.staged_path,
                    width=plan.settings.width,
                    height=plan.settings.height,
                )
                newly_visible = union_accumulator.add_coverage(
                    coverage,
                    frame_index=task.task_index,
                )
                del coverage
                logger.debug(
                    "Merged B4 projection frame %d into alpha coverage union: "
                    "new_nonzero=%d raw_nonzero_total=%d",
                    task.task_index,
                    newly_visible,
                    union_accumulator.visible_pixel_count,
                )

        if union_accumulator is None:
            return build_full_frame_layout(
                plan.settings.width,
                plan.settings.height,
                frame_count=len(plan.frame_tasks),
            )
        try:
            layout = union_accumulator.build_layout()
        except CameraProjectionLayoutError as exc:
            raise CameraProjectionExecutionError(str(exc)) from exc
        for reservation in resolved:
            rewrite_staged_image_with_crop(
                bpy_module,
                plan,
                reservation,
                layout,
                output_policy,
            )
        logger.info(
            "B4 union layout '%s': full=%dx%d crop=(%d,%d)-(%d,%d) "
            "size=%dx%d contour=%s vertices=%d source_vertices=%d "
            "outer_components=%d contour_fallback=%r coverage=%s "
            "raw_nonzero=%d strong=%d final_visible=%d components=%d->%d "
            "removed=%d filled_holes=%d weak_only=%s frames=%d union_bytes=%d "
            "fringe_threshold=%.8f core_threshold=%.8f simplify_tolerance=%.4f "
            "dynamic_range=%s tone_mapping=%s alpha=%s color_depth=%s",
            plan.source_object_id,
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
            union_accumulator.frame_count,
            union_accumulator.allocated_mask_bytes,
            layout.alpha_threshold,
            layout.coverage_core_alpha_threshold,
            layout.simplify_tolerance_pixels,
            output_policy.dynamic_range.value,
            output_policy.tone_mapping.value,
            output_policy.alpha_representation.value,
            output_policy.color_depth,
        )
        return layout


def _reserve(
    plan: CameraProjectionPlan,
    transaction: AtomicFileTransaction,
) -> Tuple[AtomicOutputReservation, ...]:
    return tuple(transaction.reserve(task.output_path) for task in plan.frame_tasks)


def stage_camera_projection_outputs_detailed(
    source_obj: Any,
    plan: CameraProjectionPlan,
    output_transaction: AtomicFileTransaction,
    execution_settings: BakeExecutionSettings | None = None,
    *,
    context: Any | None = None,
    scene: Any | None = None,
) -> CameraProjectionStageResult:
    if not isinstance(output_transaction, AtomicFileTransaction):
        raise TypeError("output_transaction must be an AtomicFileTransaction")
    resolved_settings = execution_settings or BakeExecutionSettings()
    if not isinstance(resolved_settings, BakeExecutionSettings):
        raise TypeError("execution_settings must be BakeExecutionSettings or None")
    try:
        reservations = _reserve(plan, output_transaction)
        layout = _render_to_reservations(
            source_obj,
            plan,
            resolved_settings,
            reservations,
            context=context,
            scene=scene,
            apply_crop=True,
        )
        return CameraProjectionStageResult(reservations, layout)
    except CameraProjectionExecutionError:
        raise
    except Exception as exc:
        logger.exception("Unexpected B4 projection failure for '%s'", plan.source_object_id)
        raise CameraProjectionExecutionError(
            f"Camera projection failed for '{plan.source_object_id}': {exc}"
        ) from exc


def stage_camera_projection_outputs(
    source_obj: Any,
    plan: CameraProjectionPlan,
    output_transaction: AtomicFileTransaction,
    execution_settings: BakeExecutionSettings | None = None,
    *,
    context: Any | None = None,
    scene: Any | None = None,
) -> Tuple[AtomicOutputReservation, ...]:
    """Keep full-frame output for callers that serialized projection JSON before staging."""

    if not isinstance(output_transaction, AtomicFileTransaction):
        raise TypeError("output_transaction must be an AtomicFileTransaction")
    resolved_settings = execution_settings or BakeExecutionSettings()
    if not isinstance(resolved_settings, BakeExecutionSettings):
        raise TypeError("execution_settings must be BakeExecutionSettings or None")
    reservations = _reserve(plan, output_transaction)
    _render_to_reservations(
        source_obj,
        plan,
        resolved_settings,
        reservations,
        context=context,
        scene=scene,
        apply_crop=False,
    )
    return reservations


def _build_execution_result(
    plan: CameraProjectionPlan,
    committed_paths: Iterable[Path],
    layout: CameraProjectionLayout,
) -> BakeExecutionResult:
    resolved = tuple(
        Path(path).expanduser().resolve(strict=False) for path in committed_paths
    )
    expected = tuple(
        task.output_path.expanduser().resolve(strict=False) for task in plan.frame_tasks
    )
    if resolved != expected:
        raise CameraProjectionExecutionError(
            f"Committed projection paths do not match plan; expected={expected}, got={resolved}"
        )
    artifacts = tuple(
        BakeArtifact(
            task_index=task.task_index,
            timeline_frame=task.timeline_frame,
            image_name=task.image_name,
            output_path=path,
            width=layout.cropped_width,
            height=layout.cropped_height,
        )
        for task, path in zip(plan.frame_tasks, resolved)
    )
    return BakeExecutionResult(plan=plan, artifacts=artifacts)


def execute_camera_projection_plan(
    source_obj: Any,
    plan: CameraProjectionPlan,
    execution_settings: BakeExecutionSettings | None = None,
    *,
    context: Any | None = None,
    scene: Any | None = None,
) -> BakeExecutionResult:
    with atomic_file_transaction() as transaction:
        staged = stage_camera_projection_outputs_detailed(
            source_obj,
            plan,
            transaction,
            execution_settings,
            context=context,
            scene=scene,
        )
        committed = transaction.commit()
    return _build_execution_result(plan, committed, staged.layout)
