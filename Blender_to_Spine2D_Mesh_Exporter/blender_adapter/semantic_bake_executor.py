"""Compatibility facade for semantic object-bake validation, execution, and output."""

from .semantic_bake_output import (
    BakeExecutionError,
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
