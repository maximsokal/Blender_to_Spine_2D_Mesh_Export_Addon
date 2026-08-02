"""Render validated Blender 5.2 camera-projection tasks into staged files."""

from __future__ import annotations

import logging
from typing import Any, Tuple

from ..application import A1ExportProgressCallback, emit_a1_frame_progress
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


def _call_render_operator(bpy_module: Any) -> None:
    """Invoke Blender 5.2 still rendering through the physical execution owner."""

    if bpy_module is None:
        raise CameraProjectionExecutionError("bpy_module cannot be None")
    try:
        operator = bpy_module.ops.render.render
    except Exception as exc:
        raise CameraProjectionExecutionError(
            "bpy.ops.render.render is unavailable"
        ) from exc
    poll = getattr(operator, "poll", None)
    if callable(poll):
        try:
            available = bool(poll())
        except Exception as exc:
            raise CameraProjectionExecutionError(
                "bpy.ops.render.render.poll() failed"
            ) from exc
        if not available:
            raise CameraProjectionExecutionError(
                "bpy.ops.render.render.poll() returned False"
            )
    try:
        result = operator(write_still=True)
    except Exception as exc:
        raise CameraProjectionExecutionError(
            "bpy.ops.render.render(write_still=True) failed"
        ) from exc
    try:
        finished = "FINISHED" in result
    except Exception as exc:
        raise CameraProjectionExecutionError(
            f"bpy.ops.render.render returned an invalid result: {result!r}"
        ) from exc
    if not finished:
        raise CameraProjectionExecutionError(
            f"bpy.ops.render.render did not finish: {result!r}"
        )


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
    *,
    progress_callback: A1ExportProgressCallback | None = None,
) -> Tuple[AtomicOutputReservation, ...]:
    """Render every full-frame task and restore Scene state before returning."""

    if not isinstance(runtime, CameraProjectionRuntime):
        raise TypeError("runtime must be CameraProjectionRuntime")

    resolved = validate_camera_projection_reservations(
        runtime.plan,
        reservations,
    )
    frame_count = len(runtime.plan.frame_tasks)

    with preserve_camera_projection_state(runtime.scene):
        configure_camera_visibility(
            runtime.source_object,
            runtime.scene,
            isolate=runtime.plan.isolate_source_to_camera,
            influence_policy=runtime.execution_settings.camera_influence_policy,
        )

        for frame_index, (task, reservation) in enumerate(
            zip(runtime.plan.frame_tasks, resolved, strict=True),
            start=1,
        ):
            emit_a1_frame_progress(
                progress_callback,
                stage="CAMERA_RENDER_FRAME",
                action="Rendering",
                frame_index=frame_index,
                frame_count=frame_count,
                completed=False,
                object_id=runtime.plan.source_object_id,
            )
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
            policy = runtime.execution_settings.camera_influence_policy
            logger.info(
                "Rendering camera projection '%s' frame %d/%d camera='%s' "
                "dynamic_range=%s tone_mapping=%s alpha=%s shadows=%s "
                "reflection_transmission=%s world=%s",
                runtime.plan.source_object_id,
                frame_index,
                frame_count,
                runtime.plan.camera_object_id,
                runtime.output_policy.dynamic_range.value,
                runtime.output_policy.tone_mapping.value,
                runtime.output_policy.alpha_representation.value,
                policy.include_scene_shadows,
                policy.include_scene_reflection_transmission,
                policy.world_affects_lighting_reflections,
            )
            _call_render_operator(runtime.bpy_module)
            _require_nonempty_staged_output(reservation)
            emit_a1_frame_progress(
                progress_callback,
                stage="CAMERA_RENDER_FRAME",
                action="Rendered",
                frame_index=frame_index,
                frame_count=frame_count,
                completed=True,
                object_id=runtime.plan.source_object_id,
            )

    return resolved


__all__ = [
    "CameraProjectionExecutionError",
    "_call_render_operator",
    "render_camera_projection_frames",
]
