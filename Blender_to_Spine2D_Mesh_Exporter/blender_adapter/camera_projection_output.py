"""Own B4 output reservation, postprocess staging, atomic commit, and results."""

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
)
from ..domain.baking.projection_layout import CameraProjectionLayout
from ..infrastructure import (
    AtomicFileTransaction,
    AtomicOutputReservation,
    atomic_file_transaction,
)
from .camera_projection_error import CameraProjectionExecutionError
from .camera_projection_execution import render_camera_projection_frames
from .camera_projection_postprocess import process_camera_projection_outputs
from .camera_projection_validation import (
    CameraProjectionRuntime,
    validate_camera_projection_request,
    validate_camera_projection_reservations,
)


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CameraProjectionStageResult:
    """Staged B4 paths plus the exact render-derived projection layout."""

    reservations: Tuple[AtomicOutputReservation, ...]
    layout: CameraProjectionLayout

    def __post_init__(self) -> None:
        if not isinstance(self.reservations, tuple) or not self.reservations:
            raise ValueError("reservations must be a non-empty tuple")
        if not all(
            isinstance(item, AtomicOutputReservation)
            for item in self.reservations
        ):
            raise TypeError(
                "reservations must contain AtomicOutputReservation values"
            )
        if not isinstance(self.layout, CameraProjectionLayout):
            raise TypeError("layout must be CameraProjectionLayout")
        if len(self.reservations) != self.layout.frame_count:
            raise ValueError(
                "reservation count must match layout.frame_count"
            )


def _plan_identifier(plan: object) -> str:
    value = getattr(plan, "source_object_id", None)
    return str(value) if value is not None else "<unvalidated-camera-plan>"


def _require_transaction(value: Any) -> AtomicFileTransaction:
    if not isinstance(value, AtomicFileTransaction):
        raise TypeError(
            "output_transaction must be an AtomicFileTransaction"
        )
    return value


def _reserve_camera_projection_outputs(
    plan: CameraProjectionPlan,
    transaction: AtomicFileTransaction,
) -> Tuple[AtomicOutputReservation, ...]:
    """Reserve every validated B4 frame path in immutable task order."""

    if not isinstance(plan, CameraProjectionPlan):
        raise TypeError("plan must be CameraProjectionPlan")
    if not isinstance(transaction, AtomicFileTransaction):
        raise TypeError("transaction must be AtomicFileTransaction")

    reservations = tuple(
        transaction.reserve(task.output_path)
        for task in plan.frame_tasks
    )
    return validate_camera_projection_reservations(
        plan,
        reservations,
    )


def _stage_validated_camera_projection(
    runtime: CameraProjectionRuntime,
    transaction: AtomicFileTransaction,
    *,
    apply_crop: bool,
) -> CameraProjectionStageResult:
    """Reserve, render, restore state, then optionally derive and apply crop."""

    if not isinstance(runtime, CameraProjectionRuntime):
        raise TypeError("runtime must be CameraProjectionRuntime")
    if not isinstance(apply_crop, bool):
        raise TypeError("apply_crop must be bool")

    reservations = _reserve_camera_projection_outputs(
        runtime.plan,
        transaction,
    )
    rendered = render_camera_projection_frames(
        runtime,
        reservations,
    )
    layout = process_camera_projection_outputs(
        runtime,
        rendered,
        apply_crop=apply_crop,
    )
    return CameraProjectionStageResult(
        reservations=rendered,
        layout=layout,
    )


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
    """Compatibility execution for already reserved historical callers."""

    runtime = validate_camera_projection_request(
        source_obj,
        plan,
        execution_settings,
        context=context,
        scene=scene,
    )
    resolved = validate_camera_projection_reservations(
        runtime.plan,
        reservations,
    )
    render_camera_projection_frames(runtime, resolved)
    return process_camera_projection_outputs(
        runtime,
        resolved,
        apply_crop=apply_crop,
    )


def stage_camera_projection_outputs_detailed(
    source_obj: Any,
    plan: CameraProjectionPlan,
    output_transaction: AtomicFileTransaction,
    execution_settings: BakeExecutionSettings | None = None,
    *,
    context: Any | None = None,
    scene: Any | None = None,
) -> CameraProjectionStageResult:
    """Validate, stage full renders, then build and apply one stable B4 crop."""

    transaction = _require_transaction(output_transaction)
    try:
        runtime = validate_camera_projection_request(
            source_obj,
            plan,
            execution_settings,
            context=context,
            scene=scene,
        )
        return _stage_validated_camera_projection(
            runtime,
            transaction,
            apply_crop=True,
        )
    except CameraProjectionExecutionError:
        raise
    except Exception as exc:
        plan_id = _plan_identifier(plan)
        logger.exception(
            "Unexpected B4 projection failure for '%s'",
            plan_id,
        )
        raise CameraProjectionExecutionError(
            f"Camera projection failed for '{plan_id}': {exc}"
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
    """Stage compatibility full-frame output without coverage decode or crop."""

    transaction = _require_transaction(output_transaction)
    try:
        runtime = validate_camera_projection_request(
            source_obj,
            plan,
            execution_settings,
            context=context,
            scene=scene,
        )
        reservations = _reserve_camera_projection_outputs(
            runtime.plan,
            transaction,
        )
        return render_camera_projection_frames(
            runtime,
            reservations,
        )
    except CameraProjectionExecutionError:
        raise
    except Exception as exc:
        plan_id = _plan_identifier(plan)
        logger.exception(
            "Unexpected full-frame B4 projection failure for '%s'",
            plan_id,
        )
        raise CameraProjectionExecutionError(
            f"Camera projection failed for '{plan_id}': {exc}"
        ) from exc


def build_camera_projection_execution_result(
    plan: CameraProjectionPlan,
    committed_paths: Iterable[Path],
    layout: CameraProjectionLayout,
) -> BakeExecutionResult:
    """Build artifacts only when commit and frame-task order match exactly."""

    if not isinstance(plan, CameraProjectionPlan):
        raise TypeError("plan must be CameraProjectionPlan")
    if not isinstance(layout, CameraProjectionLayout):
        raise TypeError("layout must be CameraProjectionLayout")
    if layout.frame_count != len(plan.frame_tasks):
        raise CameraProjectionExecutionError(
            "projection layout frame_count does not match CameraProjectionPlan"
        )

    resolved = tuple(
        Path(path).expanduser().resolve(strict=False)
        for path in committed_paths
    )
    expected = tuple(
        task.output_path.expanduser().resolve(strict=False)
        for task in plan.frame_tasks
    )
    if resolved != expected:
        raise CameraProjectionExecutionError(
            "Committed projection paths do not match frame task order; "
            f"expected={expected}, got={resolved}"
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
        for task, path in zip(
            plan.frame_tasks,
            resolved,
            strict=True,
        )
    )
    return BakeExecutionResult(
        plan=plan,
        artifacts=artifacts,
    )


def execute_camera_projection_plan(
    source_obj: Any,
    plan: CameraProjectionPlan,
    execution_settings: BakeExecutionSettings | None = None,
    *,
    context: Any | None = None,
    scene: Any | None = None,
) -> BakeExecutionResult:
    """Validate before transaction creation, stage, and commit exactly once."""

    try:
        runtime = validate_camera_projection_request(
            source_obj,
            plan,
            execution_settings,
            context=context,
            scene=scene,
        )
        with atomic_file_transaction(
            operation_name="camera-projection"
        ) as transaction:
            staged = _stage_validated_camera_projection(
                runtime,
                transaction,
                apply_crop=True,
            )
            expected_commit_order = tuple(
                reservation.final_path
                for reservation in staged.reservations
            )
            committed_paths = tuple(transaction.commit())

        if committed_paths != expected_commit_order:
            raise CameraProjectionExecutionError(
                "Atomic transaction changed B4 output order; "
                f"expected={expected_commit_order}, got={committed_paths}"
            )

        result = build_camera_projection_execution_result(
            runtime.plan,
            committed_paths,
            staged.layout,
        )
        logger.info(
            "Committed %d B4 projection files for '%s'",
            len(result.artifacts),
            runtime.plan.source_object_id,
        )
        return result
    except CameraProjectionExecutionError:
        raise
    except Exception as exc:
        plan_id = _plan_identifier(plan)
        logger.exception(
            "Unable to commit B4 projection outputs for '%s'",
            plan_id,
        )
        raise CameraProjectionExecutionError(
            f"Unable to commit camera projection for '{plan_id}': {exc}"
        ) from exc


# Historical private names retained without duplicate implementation.
_reserve = _reserve_camera_projection_outputs
_build_execution_result = build_camera_projection_execution_result


__all__ = [
    "CameraProjectionExecutionError",
    "CameraProjectionStageResult",
    "_build_execution_result",
    "_render_to_reservations",
    "_reserve",
    "build_camera_projection_execution_result",
    "execute_camera_projection_plan",
    "stage_camera_projection_outputs",
    "stage_camera_projection_outputs_detailed",
]
