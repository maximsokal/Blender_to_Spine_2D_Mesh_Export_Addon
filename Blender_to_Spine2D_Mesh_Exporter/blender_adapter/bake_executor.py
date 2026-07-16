"""Stable public facade for semantic texture bake execution.

Low-level Blender primitives live in :mod:`bake_executor_core`; strategy execution,
material preparation, and RGBA composition live in :mod:`semantic_bake_executor`.
The public import path is preserved for production code and failure-injection tests.
"""

from __future__ import annotations

from typing import Any

from . import bake_executor_core as _core

BakeExecutionError = _core.BakeExecutionError


def _call_bake_operator(bpy_module: Any, bake_type: str) -> None:
    """Compatibility hook and the only public route to Blender's bake operator."""

    _core._call_bake_operator(bpy_module, bake_type)


from .semantic_bake_executor import (  # noqa: E402
    build_bake_execution_result,
    execute_bake_plan,
    stage_bake_plan_outputs,
)

__all__ = [
    "BakeExecutionError",
    "build_bake_execution_result",
    "execute_bake_plan",
    "stage_bake_plan_outputs",
]
