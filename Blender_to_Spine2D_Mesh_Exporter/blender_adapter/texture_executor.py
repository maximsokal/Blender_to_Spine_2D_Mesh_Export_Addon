"""Dispatch immutable texture plans to object baking or camera projection.

This module owns no Blender operator access.  The stable ``bake_executor`` facade keeps
only the two failure-injection hooks for the real object-bake and render operators.
"""

from __future__ import annotations

from typing import Any

from ..domain.baking import CameraProjectionPlan
from .camera_projection_executor import (
    execute_camera_projection_plan,
    stage_camera_projection_outputs,
)
from .semantic_bake_executor import (
    execute_bake_plan as execute_object_bake_plan,
    stage_bake_plan_outputs as stage_object_bake_outputs,
)


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
    """Stage one texture plan without committing its caller-owned transaction."""

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


__all__ = ["execute_bake_plan", "stage_bake_plan_outputs"]
