"""Stable public facade for object baking and B4 camera projection execution."""

from __future__ import annotations

from typing import Any

from ..domain.baking import CameraProjectionPlan
from . import bake_executor_core as _core

BakeExecutionError = _core.BakeExecutionError


def _call_bake_operator(bpy_module: Any, bake_type: str) -> None:
    """Compatibility hook and the only public route to Blender's bake operator."""

    _core._call_bake_operator(bpy_module, bake_type)


def _call_render_operator(bpy_module: Any) -> None:
    """Compatibility hook and the only B4 route to Blender's render operator."""

    operator = bpy_module.ops.render.render
    poll = getattr(operator, "poll", None)
    if callable(poll) and not poll():
        raise BakeExecutionError("bpy.ops.render.render.poll() returned False")
    try:
        result = operator(write_still=True)
    except Exception as exc:
        raise BakeExecutionError("bpy.ops.render.render(write_still=True) failed") from exc
    try:
        finished = "FINISHED" in result
    except Exception as exc:
        raise BakeExecutionError(
            f"bpy.ops.render.render returned an invalid result: {result!r}"
        ) from exc
    if not finished:
        raise BakeExecutionError(
            f"bpy.ops.render.render did not finish: {result!r}"
        )


from .camera_projection_executor import (  # noqa: E402
    CameraProjectionExecutionError,
    execute_camera_projection_plan,
    stage_camera_projection_outputs,
)
from .semantic_bake_executor import (  # noqa: E402
    build_bake_execution_result,
    execute_bake_plan as _execute_object_bake_plan,
    stage_bake_plan_outputs as _stage_object_bake_outputs,
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
    """Stage either an object-bake plan or a camera projection plan."""

    if isinstance(plan, CameraProjectionPlan):
        return stage_camera_projection_outputs(
            source_obj,
            plan,
            output_transaction,
            execution_settings,
            context=context,
            scene=scene,
        )
    return _stage_object_bake_outputs(
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
    """Execute either texture pipeline and atomically commit its outputs."""

    if isinstance(plan, CameraProjectionPlan):
        return execute_camera_projection_plan(
            source_obj,
            plan,
            execution_settings,
            context=context,
            scene=scene,
        )
    return _execute_object_bake_plan(
        source_obj,
        target_snapshot,
        plan,
        execution_settings,
        context=context,
        scene=scene,
    )


__all__ = [
    "BakeExecutionError",
    "CameraProjectionExecutionError",
    "build_bake_execution_result",
    "execute_bake_plan",
    "stage_bake_plan_outputs",
]
