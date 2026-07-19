"""Render validated B4 frame tasks into caller-owned staged reservations."""

from __future__ import annotations

import logging
from typing import Any, Tuple

from ..infrastructure import AtomicOutputReservation
from .camera_projection_error import CameraProjectionExecutionError
from .camera_projection_state import (
    configure_camera_visibility,
    configure_scene_for_camera_projection,
    preserve_camera_projection_state,
    set_timeline_frame,
)
from .camera_projection_validation import (
    CameraProjectionRuntime,
    validate_camera_projection_reservations,
)


logger = logging.getLogger(__name__)


def call_public_render_operator(bpy_module: Any) -> None:
    """Route rendering through the stable public failure-injection hook."""

    from . import bake_executor as public_executor

    public_executor._call_render_operator(bpy_module)


def _require_nonempty_staged_output(reservation: AtomicOutputReservation) -> None:
    try:
        exists = reservation.staged_path.is_file()
        size = reservation.staged_path.stat().st_size if exists else 0
    except Exception as exc:
        raise CameraProjectionExecutionError(
            "Unable to inspect staged projection output "
            f"'{reservation.staged_path}'"
        ) from exc

    if not exists or size <= 0:
        raise CameraProjectionExecutionError(
            "Projection staged output is missing or empty: "
            f"{reservation.staged_path}"
        )


def render_camera_projection_frames(
    runtime: CameraProjectionRuntime,
    reservations: Tuple[AtomicOutputReservation, ...],
) -> Tuple[AtomicOutputReservation, ...]:
    """Render every full-frame B4 task and restore Scene state before returning."""

    if not isinstance(runtime, CameraProjectionRuntime):
        raise TypeError("runtime must be CameraProjectionRuntime")

    resolved = validate_camera_projection_reservations(
        runtime.plan,
        reservations,
    )

    with preserve_camera_projection_state(runtime.scene):
        configure_camera_visibility(
            runtime.source_object,
            runtime.scene,
            isolate=runtime.plan.isolate_source_to_camera,
        )

        for task, reservation in zip(
            runtime.plan.frame_tasks,
            resolved,
            strict=True,
        ):
            set_timeline_frame(
                runtime.scene,
                runtime.context,
                task.timeline_frame,
            )
            configure_scene_for_camera_projection(
                runtime.scene,
                runtime.plan,
                runtime.execution_settings,
                reservation.staged_path,
            )
            logger.info(
                "Rendering B4 projection '%s' frame %d/%d camera='%s' "
                "dynamic_range=%s tone_mapping=%s alpha=%s",
                runtime.plan.source_object_id,
                task.task_index + 1,
                len(runtime.plan.frame_tasks),
                runtime.plan.camera_object_id,
                runtime.output_policy.dynamic_range.value,
                runtime.output_policy.tone_mapping.value,
                runtime.output_policy.alpha_representation.value,
            )
            call_public_render_operator(runtime.bpy_module)
            _require_nonempty_staged_output(reservation)

    return resolved


__all__ = [
    "CameraProjectionExecutionError",
    "call_public_render_operator",
    "render_camera_projection_frames",
]
