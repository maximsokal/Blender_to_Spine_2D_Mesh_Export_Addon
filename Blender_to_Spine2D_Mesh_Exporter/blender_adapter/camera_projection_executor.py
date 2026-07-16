"""Stable public facade for B4 camera render projection execution."""

from .camera_projection_executor_core import (
    CameraProjectionStageResult,
    execute_camera_projection_plan,
    stage_camera_projection_outputs,
    stage_camera_projection_outputs_detailed,
)
from .camera_projection_state import (
    CameraProjectionExecutionError,
    configure_scene_for_camera_projection,
    preserve_camera_projection_state,
)

__all__ = [
    "CameraProjectionExecutionError",
    "CameraProjectionStageResult",
    "configure_scene_for_camera_projection",
    "execute_camera_projection_plan",
    "preserve_camera_projection_state",
    "stage_camera_projection_outputs",
    "stage_camera_projection_outputs_detailed",
]
