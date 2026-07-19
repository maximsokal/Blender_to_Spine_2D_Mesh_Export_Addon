"""Compatibility facade for the decomposed B4 camera projection pipeline.

Physical ownership lives in validation, reversible execution, postprocessing, and
atomic output modules. Historical private imports remain available without a
second implementation.
"""

from .camera_projection_error import CameraProjectionExecutionError
from .camera_projection_output import (
    CameraProjectionStageResult,
    _build_execution_result,
    _render_to_reservations,
    _reserve,
    build_camera_projection_execution_result,
    execute_camera_projection_plan,
    stage_camera_projection_outputs,
    stage_camera_projection_outputs_detailed,
)


__all__ = [
    "CameraProjectionExecutionError",
    "CameraProjectionStageResult",
    "_build_execution_result",
    "_render_to_reservations",
    "_reserve",
    "build_camera_projection_execution_result",
    "execute_camera_projection_plan",
    "stage_camera_projection_outputs",
    "stage_camera_projection_outputs_detailed",
]
