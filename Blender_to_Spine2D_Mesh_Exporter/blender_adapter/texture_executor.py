"""Dispatch immutable texture plans to Blender 5.2 execution owners.

Every entry point first builds one typed ``TextureExecutionRequest`` so invalid
domain values fail before filesystem reservations or Blender Scene mutation.
The detailed staging API retains camera-projection layout metadata required by
post-render Spine document finalization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

from ..application import A1ExportProgressCallback
from ..domain.baking import (
    BakeExecutionResult,
    BakeExecutionSettings,
    BakePlan,
    CameraProjectionPlan,
)
from ..domain.baking.projection_layout import CameraProjectionLayout
from ..domain.geometry import MeshSnapshot
from ..infrastructure import (
    AtomicFileTransaction,
    AtomicOutputReservation,
)
from .camera_projection_output import (
    execute_camera_projection_plan,
    stage_camera_projection_outputs_detailed,
)
from .semantic_bake_output import (
    execute_bake_plan as execute_object_bake_plan,
    stage_bake_plan_outputs as stage_object_bake_outputs,
)


@dataclass(frozen=True, slots=True)
class TextureExecutionRequest:
    """One validated texture execution request shared by all dispatch routes."""

    source_object: Any
    target_snapshot: MeshSnapshot
    plan: BakePlan
    execution_settings: BakeExecutionSettings

    def __post_init__(self) -> None:
        if self.source_object is None:
            raise ValueError("source_object cannot be None")
        if not isinstance(self.target_snapshot, MeshSnapshot):
            raise TypeError("target_snapshot must be MeshSnapshot")
        if not isinstance(self.plan, BakePlan):
            raise TypeError("plan must be BakePlan")
        if not isinstance(self.execution_settings, BakeExecutionSettings):
            raise TypeError("execution_settings must be BakeExecutionSettings")
        if self.target_snapshot.source_object_id != self.plan.source_object_id:
            raise ValueError(
                "target_snapshot.source_object_id must match plan.source_object_id"
            )

    @classmethod
    def capture(
        cls,
        source_object: Any,
        target_snapshot: MeshSnapshot,
        plan: BakePlan,
        execution_settings: BakeExecutionSettings | None = None,
    ) -> "TextureExecutionRequest":
        resolved_settings = (
            BakeExecutionSettings()
            if execution_settings is None
            else execution_settings
        )
        return cls(
            source_object=source_object,
            target_snapshot=target_snapshot,
            plan=plan,
            execution_settings=resolved_settings,
        )


@dataclass(frozen=True, slots=True)
class TextureStageResult:
    reservations: Tuple[AtomicOutputReservation, ...]
    projection_layout: CameraProjectionLayout | None = None

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
        if self.projection_layout is not None and not isinstance(
            self.projection_layout,
            CameraProjectionLayout,
        ):
            raise TypeError(
                "projection_layout must be CameraProjectionLayout or None"
            )
        if (
            self.projection_layout is not None
            and self.projection_layout.frame_count != len(self.reservations)
        ):
            raise ValueError(
                "projection layout frame count must match reservations"
            )


def _require_transaction(value: Any) -> AtomicFileTransaction:
    if not isinstance(value, AtomicFileTransaction):
        raise TypeError("output_transaction must be AtomicFileTransaction")
    return value


def stage_texture_plan_outputs(
    source_obj: Any,
    target_snapshot: MeshSnapshot,
    plan: BakePlan,
    output_transaction: AtomicFileTransaction,
    execution_settings: BakeExecutionSettings | None = None,
    *,
    context: Any | None = None,
    scene: Any | None = None,
    progress_callback: A1ExportProgressCallback | None = None,
) -> TextureStageResult:
    """Stage one texture plan and retain camera-projection layout metadata."""

    request = TextureExecutionRequest.capture(
        source_obj,
        target_snapshot,
        plan,
        execution_settings,
    )
    transaction = _require_transaction(output_transaction)
    if isinstance(request.plan, CameraProjectionPlan):
        staged = stage_camera_projection_outputs_detailed(
            request.source_object,
            request.plan,
            transaction,
            request.execution_settings,
            context=context,
            scene=scene,
            progress_callback=progress_callback,
        )
        return TextureStageResult(staged.reservations, staged.layout)
    reservations = stage_object_bake_outputs(
        request.source_object,
        request.target_snapshot,
        request.plan,
        transaction,
        request.execution_settings,
        context=context,
        scene=scene,
        progress_callback=progress_callback,
    )
    return TextureStageResult(tuple(reservations))


def execute_bake_plan(
    source_obj: Any,
    target_snapshot: MeshSnapshot,
    plan: BakePlan,
    execution_settings: BakeExecutionSettings | None = None,
    *,
    context: Any | None = None,
    scene: Any | None = None,
    progress_callback: A1ExportProgressCallback | None = None,
) -> BakeExecutionResult:
    """Execute one validated texture plan and atomically commit its outputs."""

    request = TextureExecutionRequest.capture(
        source_obj,
        target_snapshot,
        plan,
        execution_settings,
    )
    if isinstance(request.plan, CameraProjectionPlan):
        return execute_camera_projection_plan(
            request.source_object,
            request.plan,
            request.execution_settings,
            context=context,
            scene=scene,
            progress_callback=progress_callback,
        )
    return execute_object_bake_plan(
        request.source_object,
        request.target_snapshot,
        request.plan,
        request.execution_settings,
        context=context,
        scene=scene,
        progress_callback=progress_callback,
    )


__all__ = [
    "TextureExecutionRequest",
    "TextureStageResult",
    "execute_bake_plan",
    "stage_texture_plan_outputs",
]
