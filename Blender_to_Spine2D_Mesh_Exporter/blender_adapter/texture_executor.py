"""Dispatch immutable texture plans to object baking or camera projection.

This module owns no Blender operator access. The detailed API returns render-derived layout
metadata for orchestration that finalizes JSON after staging. The historical reservations-only
API keeps B4 full-frame so existing multi-object code cannot commit cropped images beside a
pre-serialized full-frame document.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

from ..domain.baking import CameraProjectionPlan
from ..domain.baking.projection_layout import CameraProjectionLayout
from ..infrastructure import AtomicOutputReservation
from .camera_projection_executor import (
    execute_camera_projection_plan,
    stage_camera_projection_outputs,
    stage_camera_projection_outputs_detailed,
)
from .semantic_bake_executor import (
    execute_bake_plan as execute_object_bake_plan,
    stage_bake_plan_outputs as stage_object_bake_outputs,
)


@dataclass(frozen=True, slots=True)
class TextureStageResult:
    reservations: Tuple[AtomicOutputReservation, ...]
    projection_layout: CameraProjectionLayout | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.reservations, tuple) or not self.reservations:
            raise ValueError("reservations must be a non-empty tuple")
        if not all(isinstance(item, AtomicOutputReservation) for item in self.reservations):
            raise TypeError("reservations must contain AtomicOutputReservation values")
        if self.projection_layout is not None and not isinstance(
            self.projection_layout,
            CameraProjectionLayout,
        ):
            raise TypeError("projection_layout must be CameraProjectionLayout or None")
        if (
            self.projection_layout is not None
            and self.projection_layout.frame_count != len(self.reservations)
        ):
            raise ValueError("projection layout frame count must match reservations")


def stage_texture_plan_outputs(
    source_obj: Any,
    target_snapshot: Any,
    plan: Any,
    output_transaction: Any,
    execution_settings: Any = None,
    *,
    context: Any | None = None,
    scene: Any | None = None,
) -> TextureStageResult:
    """Stage one plan and retain the exact B4 crop/hull layout when applicable."""

    if isinstance(plan, CameraProjectionPlan):
        staged = stage_camera_projection_outputs_detailed(
            source_obj,
            plan,
            output_transaction,
            execution_settings,
            context=context,
            scene=scene,
        )
        return TextureStageResult(staged.reservations, staged.layout)
    reservations = stage_object_bake_outputs(
        source_obj,
        target_snapshot,
        plan,
        output_transaction,
        execution_settings,
        context=context,
        scene=scene,
    )
    return TextureStageResult(tuple(reservations))


def stage_bake_plan_outputs(
    source_obj: Any,
    target_snapshot: Any,
    plan: Any,
    output_transaction: Any,
    execution_settings: Any = None,
    *,
    context: Any | None = None,
    scene: Any | None = None,
):
    """Compatibility staging for callers that do not post-finalize projection JSON."""

    if isinstance(plan, CameraProjectionPlan):
        return stage_camera_projection_outputs(
            source_obj,
            plan,
            output_transaction,
            execution_settings,
            context=context,
            scene=scene,
        )
    return stage_object_bake_outputs(
        source_obj,
        target_snapshot,
        plan,
        output_transaction,
        execution_settings,
        context=context,
        scene=scene,
    )


def execute_bake_plan(
    source_obj: Any,
    target_snapshot: Any,
    plan: Any,
    execution_settings: Any = None,
    *,
    context: Any | None = None,
    scene: Any | None = None,
):
    """Execute one texture plan and atomically commit its outputs."""

    if isinstance(plan, CameraProjectionPlan):
        return execute_camera_projection_plan(
            source_obj,
            plan,
            execution_settings,
            context=context,
            scene=scene,
        )
    return execute_object_bake_plan(
        source_obj,
        target_snapshot,
        plan,
        execution_settings,
        context=context,
        scene=scene,
    )


__all__ = [
    "TextureStageResult",
    "execute_bake_plan",
    "stage_bake_plan_outputs",
    "stage_texture_plan_outputs",
]
