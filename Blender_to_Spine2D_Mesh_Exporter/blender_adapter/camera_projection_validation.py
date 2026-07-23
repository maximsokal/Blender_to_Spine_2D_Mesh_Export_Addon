"""Validate Blender 5.2 camera-projection requests before any mutation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Tuple

from ..domain.baking import (
    BakeExecutionSettings,
    CameraProjectionPlan,
    ResolvedProjectionOutputPolicy,
    resolve_projection_output_policy,
)
from ..infrastructure import AtomicOutputReservation
from .camera_projection_error import CameraProjectionExecutionError
from .render_engine_contract import (
    RenderEngineContract,
    render_engine_contract,
    render_engine_contract_from_execution,
)
from .scene_bake_runtime import validate_runtime_scene_context
from .view_layer_contract import validate_source_view_layer_for_camera_projection


@dataclass(frozen=True, slots=True)
class CameraProjectionRuntime:
    """Fully validated immutable values required by camera projection."""

    source_object: Any
    plan: CameraProjectionPlan
    execution_settings: BakeExecutionSettings
    bpy_module: Any
    context: Any
    scene: Any
    renderer: RenderEngineContract
    output_policy: ResolvedProjectionOutputPolicy

    def __post_init__(self) -> None:
        if self.source_object is None:
            raise ValueError("source_object cannot be None")
        if not isinstance(self.plan, CameraProjectionPlan):
            raise TypeError("plan must be CameraProjectionPlan")
        if not isinstance(self.execution_settings, BakeExecutionSettings):
            raise TypeError("execution_settings must be BakeExecutionSettings")
        if self.bpy_module is None or self.context is None or self.scene is None:
            raise ValueError("bpy_module, context, and scene cannot be None")
        if not isinstance(self.renderer, RenderEngineContract):
            raise TypeError("renderer must be RenderEngineContract")
        if not isinstance(
            self.output_policy,
            ResolvedProjectionOutputPolicy,
        ):
            raise TypeError(
                "output_policy must be ResolvedProjectionOutputPolicy"
            )


def _load_bpy() -> Any:
    try:
        import bpy
    except Exception as exc:
        raise CameraProjectionExecutionError(
            "Blender bpy module is unavailable"
        ) from exc
    return bpy


def _validate_projection_plan_tasks(plan: CameraProjectionPlan) -> None:
    if not plan.frame_tasks:
        raise CameraProjectionExecutionError(
            "CameraProjectionPlan contains no frame tasks"
        )

    resolved_paths: list[Path] = []
    for expected_index, task in enumerate(plan.frame_tasks):
        if task.task_index != expected_index:
            raise CameraProjectionExecutionError(
                "CameraProjectionPlan frame task indices must be contiguous; "
                f"expected={expected_index}, got={task.task_index}"
            )
        if not isinstance(task.output_path, Path):
            raise TypeError("CameraProjectionPlan output paths must be pathlib.Path")
        resolved_paths.append(
            task.output_path.expanduser().resolve(strict=False)
        )

    if len(resolved_paths) != len(set(resolved_paths)):
        raise CameraProjectionExecutionError(
            "CameraProjectionPlan contains duplicate output paths"
        )


def _resolve_projection_runtime_context(
    source_obj: Any,
    plan: CameraProjectionPlan,
    *,
    context: Any | None,
    scene: Any | None,
) -> tuple[Any, Any, Any]:
    """Resolve and validate the exact Blender Context and Scene for one request."""

    if source_obj is None or getattr(source_obj, "type", None) != "MESH":
        raise CameraProjectionExecutionError(
            "source_obj must be a Blender MESH object"
        )
    if not isinstance(plan, CameraProjectionPlan):
        raise TypeError("plan must be CameraProjectionPlan")
    if str(getattr(source_obj, "name", "")) != plan.source_object_id:
        raise CameraProjectionExecutionError(
            "source object identity does not match CameraProjectionPlan"
        )

    _validate_projection_plan_tasks(plan)

    bpy_module = _load_bpy()
    resolved_context = bpy_module.context if context is None else context
    if resolved_context is None:
        raise CameraProjectionExecutionError(
            "A Blender Context is required"
        )

    resolved_scene = (
        getattr(resolved_context, "scene", None)
        if scene is None
        else scene
    )
    if resolved_scene is None:
        raise CameraProjectionExecutionError(
            "A Blender Scene is required"
        )
    if getattr(resolved_scene, "camera", None) is None:
        raise CameraProjectionExecutionError("Scene has no active camera")

    validate_source_view_layer_for_camera_projection(
        source_obj,
        getattr(resolved_context, "view_layer", None),
    )
    validate_runtime_scene_context(
        source_obj,
        plan.object_context,
        plan.scene_context,
        scene=resolved_scene,
        context=resolved_context,
    )
    return bpy_module, resolved_context, resolved_scene


def validate_camera_projection_request(
    source_obj: Any,
    plan: CameraProjectionPlan,
    execution_settings: BakeExecutionSettings | None = None,
    *,
    context: Any | None = None,
    scene: Any | None = None,
) -> CameraProjectionRuntime:
    """Resolve and validate the complete request before output reservation."""

    if source_obj is None:
        raise ValueError("source_obj cannot be None")
    if not isinstance(plan, CameraProjectionPlan):
        raise TypeError("plan must be CameraProjectionPlan")

    resolved_settings = (
        BakeExecutionSettings()
        if execution_settings is None
        else execution_settings
    )
    if not isinstance(resolved_settings, BakeExecutionSettings):
        raise TypeError(
            "execution_settings must be BakeExecutionSettings or None"
        )

    renderer = render_engine_contract_from_execution(resolved_settings)
    if plan.scene_context is None:
        raise CameraProjectionExecutionError(
            "CameraProjectionPlan is missing SceneBakeContext"
        )
    planned_renderer = render_engine_contract(
        plan.scene_context.render_engine
    )
    if renderer != planned_renderer:
        raise CameraProjectionExecutionError(
            "camera projection execution engine differs from the analyzed renderer; "
            f"planned={planned_renderer.blender_engine}, "
            f"execution={renderer.blender_engine}"
        )

    output_policy = resolve_projection_output_policy(
        resolved_settings.projection_output_policy,
        plan.settings.texture_format,
    )
    bpy_module, resolved_context, resolved_scene = (
        _resolve_projection_runtime_context(
            source_obj,
            plan,
            context=context,
            scene=scene,
        )
    )

    return CameraProjectionRuntime(
        source_object=source_obj,
        plan=plan,
        execution_settings=resolved_settings,
        bpy_module=bpy_module,
        context=resolved_context,
        scene=resolved_scene,
        renderer=renderer,
        output_policy=output_policy,
    )


def validate_camera_projection_reservations(
    plan: CameraProjectionPlan,
    reservations: Iterable[AtomicOutputReservation],
) -> Tuple[AtomicOutputReservation, ...]:
    """Require one correctly ordered reservation for every projection task."""

    if not isinstance(plan, CameraProjectionPlan):
        raise TypeError("plan must be CameraProjectionPlan")

    resolved = tuple(reservations)
    if len(resolved) != len(plan.frame_tasks):
        raise CameraProjectionExecutionError(
            f"Expected {len(plan.frame_tasks)} projection reservations, "
            f"got {len(resolved)}"
        )

    for task, reservation in zip(
        plan.frame_tasks,
        resolved,
        strict=True,
    ):
        if not isinstance(reservation, AtomicOutputReservation):
            raise TypeError(
                "reservations must contain AtomicOutputReservation values"
            )
        expected = task.output_path.expanduser().resolve(strict=False)
        if reservation.final_path != expected:
            raise CameraProjectionExecutionError(
                f"Projection task {task.task_index} expected '{expected}', "
                f"got '{reservation.final_path}'"
            )

    return resolved


__all__ = [
    "CameraProjectionExecutionError",
    "CameraProjectionRuntime",
    "validate_camera_projection_request",
    "validate_camera_projection_reservations",
]
