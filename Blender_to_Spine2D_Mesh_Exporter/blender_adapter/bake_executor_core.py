"""Compatibility facade for retired object-bake core helpers.

The semantic object-bake pipeline now lives in dedicated validation, image-I/O,
execution, and output modules. This module keeps historical private import paths
stable and remains the sole owner of direct ``bpy.ops.object.bake`` access.
"""

from __future__ import annotations

from typing import Any

from .bake_execution_error import BakeExecutionError
from .semantic_bake_image_io import (
    _activate_uv_layer,
    _create_bake_image,
    _remove_image,
    _save_bake_image,
    _set_timeline_frame,
)
from .semantic_bake_output import (
    build_bake_execution_result,
    execute_bake_plan,
    stage_bake_plan_outputs,
)
from .semantic_bake_validation import (
    _load_bpy,
    _validate_execution_input,
    validate_semantic_bake_reservations as _require_reservations,
)


def _call_bake_operator(bpy_module: Any, bake_type: str) -> None:
    """Call Blender's object-bake operator through one injectable boundary."""

    operator = bpy_module.ops.object.bake
    poll = getattr(operator, "poll", None)
    if callable(poll) and not poll():
        raise BakeExecutionError("bpy.ops.object.bake.poll() returned False")
    try:
        result = operator(type=bake_type)
    except Exception as exc:
        raise BakeExecutionError(
            f"bpy.ops.object.bake(type='{bake_type}') failed"
        ) from exc
    try:
        finished = "FINISHED" in result
    except Exception as exc:
        raise BakeExecutionError(
            f"bpy.ops.object.bake returned an invalid result: {result!r}"
        ) from exc
    if not finished:
        raise BakeExecutionError(
            f"bpy.ops.object.bake did not finish: {result!r}"
        )


__all__ = [
    "BakeExecutionError",
    "_activate_uv_layer",
    "_call_bake_operator",
    "_create_bake_image",
    "_load_bpy",
    "_remove_image",
    "_require_reservations",
    "_save_bake_image",
    "_set_timeline_frame",
    "_validate_execution_input",
    "build_bake_execution_result",
    "execute_bake_plan",
    "stage_bake_plan_outputs",
]
