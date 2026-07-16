"""Stable public facade for object baking and B4 camera projection execution.

Actual dispatch lives in :mod:`texture_executor`. This module intentionally contains only
the two real Blender operator hooks so tests can inject deterministic failures without
spreading ``bpy.ops`` access through the pipeline.
"""

from __future__ import annotations

from typing import Any

from . import bake_executor_core as _core

BakeExecutionError = _core.BakeExecutionError


def _call_bake_operator(bpy_module: Any, bake_type: str) -> None:
    """Compatibility hook and the only public route to Blender's bake operator."""

    _core._call_bake_operator(bpy_module, bake_type)


def _call_render_operator(bpy_module: Any) -> None:
    """Compatibility hook and the only route to Blender's still-render operator."""

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


from .camera_projection_executor import CameraProjectionExecutionError  # noqa: E402
from .semantic_bake_executor import build_bake_execution_result  # noqa: E402
from .texture_executor import execute_bake_plan, stage_bake_plan_outputs  # noqa: E402

__all__ = [
    "BakeExecutionError",
    "CameraProjectionExecutionError",
    "build_bake_execution_result",
    "execute_bake_plan",
    "stage_bake_plan_outputs",
]
