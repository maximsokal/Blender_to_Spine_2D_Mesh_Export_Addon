"""Stable public facade for object baking and B4 camera projection execution.

Actual dispatch lives in :mod:`texture_executor`. This module intentionally contains only
the two real Blender operator hooks so tests can inject deterministic failures without
spreading ``bpy.ops`` access through the pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import bake_executor_core as _core

BakeExecutionError = _core.BakeExecutionError


def _call_bake_operator(bpy_module: Any, bake_type: str) -> None:
    """Compatibility hook and the only public route to Blender's bake operator."""

    _core._call_bake_operator(bpy_module, bake_type)


def _call_render_operator(bpy_module: Any) -> None:
    """Render into ``Render Result`` and save it without discarding float pixels.

    Blender 4.4 background mode may expose ``Render Result`` as a zero-sized image after
    ``write_still=True`` even though the file was written. B4 needs the in-memory float RGBA
    to derive one sequence alpha union. The operator therefore renders without writing and
    the resulting Image datablock is saved through ``Image.save_render`` to the already
    configured atomic staged path.
    """

    operator = bpy_module.ops.render.render
    poll = getattr(operator, "poll", None)
    if callable(poll) and not poll():
        raise BakeExecutionError("bpy.ops.render.render.poll() returned False")
    try:
        result = operator(write_still=False)
    except Exception as exc:
        raise BakeExecutionError("bpy.ops.render.render(write_still=False) failed") from exc
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

    scene = getattr(getattr(bpy_module, "context", None), "scene", None)
    if scene is None:
        raise BakeExecutionError("Blender context has no Scene after render")
    render_result = bpy_module.data.images.get("Render Result")
    if render_result is None:
        raise BakeExecutionError("Blender did not create the Render Result image")
    try:
        width, height = (int(value) for value in render_result.size[:2])
    except Exception as exc:
        raise BakeExecutionError("Unable to inspect Render Result dimensions") from exc
    if width <= 0 or height <= 0:
        raise BakeExecutionError(
            f"Render Result has invalid dimensions after render: {(width, height)}"
        )

    filepath = Path(str(getattr(scene.render, "filepath", "") or ""))
    if not str(filepath):
        raise BakeExecutionError("Scene render filepath is empty")
    try:
        render_result.save_render(str(filepath), scene=scene)
    except Exception as exc:
        raise BakeExecutionError(
            f"Unable to save Render Result to staged path '{filepath}'"
        ) from exc


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
