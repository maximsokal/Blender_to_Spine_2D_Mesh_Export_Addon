"""Render validated grouped Blender 5.2 projection tasks reversibly."""

from __future__ import annotations

import logging
from typing import Tuple

from ..infrastructure import AtomicOutputReservation
from .camera_projection_error import CameraProjectionExecutionError
from .camera_projection_execution import _call_render_operator
from .camera_projection_state import (
    configure_scene_for_camera_projection,
    preserve_camera_projection_state,
    set_timeline_frame,
)
from .grouped_camera_projection_validation import (
    GroupedCameraProjectionRuntime,
    validate_grouped_camera_projection_reservations,
)
from .grouped_camera_projection_visibility import (
    configure_group_camera_visibility,
)


logger = logging.getLogger(__name__)


def require_nonempty_grouped_staged_output(
    reservation: AtomicOutputReservation,
) -> None:
    """Fail when Blender did not produce one usable grouped staged image."""

    if not isinstance(reservation, AtomicOutputReservation):
        raise TypeError("reservation must be AtomicOutputReservation")
    try:
        exists = reservation.staged_path.is_file()
        size = reservation.staged_path.stat().st_size if exists else 0
    except Exception as exc:
        raise CameraProjectionExecutionError(
            "Unable to inspect grouped staged projection output "
            f"'{reservation.staged_path}'"
        ) from exc

    if not exists or size <= 0:
        raise CameraProjectionExecutionError(
            "Grouped projection staged output is missing or empty: "
            f"{reservation.staged_path}"
        )


def render_grouped_camera_projection_frames(
    runtime: GroupedCameraProjectionRuntime,
    reservations: Tuple[AtomicOutputReservation, ...],
) -> Tuple[AtomicOutputReservation, ...]:
    """Render all grouped frames and restore Scene state before returning."""

    if not isinstance(runtime, GroupedCameraProjectionRuntime):
        raise TypeError("runtime must be GroupedCameraProjectionRuntime")

    resolved = validate_grouped_camera_projection_reservations(
        runtime.plan,
        reservations,
    )

    with preserve_camera_projection_state(runtime.scene):
        configure_group_camera_visibility(
            runtime.source_objects,
            runtime.scene,
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
                runtime.plan.representative_plan,
                runtime.execution_settings,
                reservation.staged_path,
            )
            logger.info(
                "Rendering grouped projection '%s' frame %d/%d camera='%s' "
                "sources=%s dynamic_range=%s tone_mapping=%s alpha=%s",
                runtime.plan.group_id,
                task.task_index + 1,
                len(runtime.plan.frame_tasks),
                runtime.plan.camera_object_id,
                runtime.plan.source_object_ids,
                runtime.output_policy.dynamic_range.value,
                runtime.output_policy.tone_mapping.value,
                runtime.output_policy.alpha_representation.value,
            )
            _call_render_operator(runtime.bpy_module)
            require_nonempty_grouped_staged_output(reservation)

    return resolved


__all__ = [
    "CameraProjectionExecutionError",
    "render_grouped_camera_projection_frames",
    "require_nonempty_grouped_staged_output",
]
