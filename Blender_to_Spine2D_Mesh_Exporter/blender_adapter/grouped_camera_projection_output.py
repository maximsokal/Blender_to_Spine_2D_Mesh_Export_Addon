"""Own grouped Blender 5.2 projection reservation, staging, and layout."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Tuple

from ..domain.baking import (
    BakeExecutionSettings,
    CameraProjectionLayout,
    GroupedCameraProjectionPlan,
)
from ..infrastructure import (
    AtomicFileTransaction,
    AtomicOutputReservation,
)
from .camera_projection_error import CameraProjectionExecutionError
from .grouped_camera_projection_execution import (
    render_grouped_camera_projection_frames,
)
from .grouped_camera_projection_postprocess import (
    process_grouped_camera_projection_outputs,
)
from .grouped_camera_projection_validation import (
    GroupedCameraProjectionRuntime,
    validate_grouped_camera_projection_request,
    validate_grouped_camera_projection_reservations,
)


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class GroupedCameraProjectionStageResult:
    """Caller-owned grouped reservations plus one exact shared layout."""

    reservations: Tuple[AtomicOutputReservation, ...]
    layout: CameraProjectionLayout
    source_object_ids: Tuple[str, ...]

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
        if (
            not isinstance(self.source_object_ids, tuple)
            or len(self.source_object_ids) < 2
            or not all(
                isinstance(value, str) and value.strip()
                for value in self.source_object_ids
            )
        ):
            raise ValueError(
                "source_object_ids must contain at least two names"
            )


def _plan_identifier(plan: object) -> str:
    value = getattr(plan, "group_id", None)
    return str(value) if value is not None else "<unvalidated-grouped-plan>"


def require_grouped_transaction(
    value: Any,
) -> AtomicFileTransaction:
    if not isinstance(value, AtomicFileTransaction):
        raise TypeError(
            "output_transaction must be an AtomicFileTransaction"
        )
    return value


def reserve_grouped_camera_projection_outputs(
    plan: GroupedCameraProjectionPlan,
    transaction: AtomicFileTransaction,
) -> Tuple[AtomicOutputReservation, ...]:
    """Reserve validated grouped frame paths in immutable task order."""

    if not isinstance(plan, GroupedCameraProjectionPlan):
        raise TypeError("plan must be GroupedCameraProjectionPlan")
    if not isinstance(transaction, AtomicFileTransaction):
        raise TypeError("transaction must be AtomicFileTransaction")

    reservations = tuple(
        transaction.reserve(task.output_path)
        for task in plan.frame_tasks
    )
    return validate_grouped_camera_projection_reservations(
        plan,
        reservations,
    )


def stage_validated_grouped_camera_projection(
    runtime: GroupedCameraProjectionRuntime,
    transaction: AtomicFileTransaction,
) -> GroupedCameraProjectionStageResult:
    """Reserve, render, restore state, then process grouped staged images."""

    if not isinstance(runtime, GroupedCameraProjectionRuntime):
        raise TypeError("runtime must be GroupedCameraProjectionRuntime")
    if not isinstance(transaction, AtomicFileTransaction):
        raise TypeError("transaction must be AtomicFileTransaction")

    reservations = reserve_grouped_camera_projection_outputs(
        runtime.plan,
        transaction,
    )
    rendered = render_grouped_camera_projection_frames(
        runtime,
        reservations,
    )
    layout = process_grouped_camera_projection_outputs(
        runtime,
        rendered,
    )
    return GroupedCameraProjectionStageResult(
        reservations=rendered,
        layout=layout,
        source_object_ids=runtime.plan.source_object_ids,
    )


def stage_grouped_camera_projection_outputs(
    source_objects: Tuple[Any, ...],
    plan: GroupedCameraProjectionPlan,
    output_transaction: AtomicFileTransaction,
    execution_settings: BakeExecutionSettings,
    *,
    context: Any | None = None,
    scene: Any | None = None,
) -> GroupedCameraProjectionStageResult:
    """Validate and stage into the caller-owned atomic transaction."""

    runtime = validate_grouped_camera_projection_request(
        source_objects,
        plan,
        execution_settings,
        context=context,
        scene=scene,
    )
    transaction = require_grouped_transaction(output_transaction)
    try:
        return stage_validated_grouped_camera_projection(
            runtime,
            transaction,
        )
    except CameraProjectionExecutionError:
        raise
    except Exception as exc:
        plan_id = _plan_identifier(plan)
        logger.exception(
            "Unexpected grouped projection failure for '%s'",
            plan_id,
        )
        raise CameraProjectionExecutionError(
            f"Grouped camera projection failed for '{plan_id}': {exc}"
        ) from exc


__all__ = [
    "CameraProjectionExecutionError",
    "GroupedCameraProjectionStageResult",
    "require_grouped_transaction",
    "reserve_grouped_camera_projection_outputs",
    "stage_grouped_camera_projection_outputs",
    "stage_validated_grouped_camera_projection",
]
