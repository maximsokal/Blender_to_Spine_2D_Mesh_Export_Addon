"""Validate grouped Blender 5.2 projection requests before mutation."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Iterable, Tuple

from ..domain.baking import (
    BakeExecutionSettings,
    CameraProjectionPlan,
    GroupedCameraProjectionPlan,
    ResolvedProjectionOutputPolicy,
    resolve_projection_output_policy,
)
from ..infrastructure import AtomicOutputReservation
from .camera_projection_error import CameraProjectionExecutionError
from .camera_projection_validation import (
    CameraProjectionRuntime,
    validate_camera_projection_request,
)
from .render_engine_contract import RenderEngineContract


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class GroupedCameraProjectionRuntime:
    """Fully validated runtime values for one depth-correct grouped render."""

    source_objects: Tuple[Any, ...]
    plan: GroupedCameraProjectionPlan
    execution_settings: BakeExecutionSettings
    bpy_module: Any
    context: Any
    scene: Any
    renderer: RenderEngineContract
    output_policy: ResolvedProjectionOutputPolicy

    def __post_init__(self) -> None:
        if (
            not isinstance(self.source_objects, tuple)
            or len(self.source_objects) < 2
            or any(item is None for item in self.source_objects)
        ):
            raise ValueError(
                "source_objects must contain at least two Blender objects"
            )
        if not isinstance(self.plan, GroupedCameraProjectionPlan):
            raise TypeError("plan must be GroupedCameraProjectionPlan")
        if len(self.source_objects) != len(self.plan.source_plans):
            raise ValueError("source_objects must match grouped source plans")
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


def object_name(obj: Any) -> str:
    """Return one stable non-empty Blender object identifier."""

    value = str(
        getattr(obj, "name_full", None)
        or getattr(obj, "name", None)
        or ""
    ).strip()
    if not value:
        raise CameraProjectionExecutionError(
            "grouped projection source has an empty name"
        )
    return value


def rna_identity(value: Any) -> tuple[str, object]:
    """Prefer Blender RNA pointer identity over transient Python wrappers."""

    pointer = getattr(value, "as_pointer", None)
    if callable(pointer):
        try:
            resolved = int(pointer())
            if resolved:
                return ("RNA_POINTER", resolved)
        except Exception:
            logger.debug(
                "Unable to read Blender RNA pointer",
                exc_info=True,
            )

    name = str(
        getattr(value, "name_full", None)
        or getattr(value, "name", None)
        or ""
    ).strip()
    if name:
        return ("RNA_NAME", name)
    return ("PYTHON_ID", id(value))


def _settings_signature(plan: GroupedCameraProjectionPlan) -> tuple[object, ...]:
    settings = plan.settings
    return (
        settings.width,
        settings.height,
        settings.output_directory.expanduser().resolve(strict=False),
        settings.texture_format,
        settings.margin_pixels,
        settings.sequence_start_frame,
        settings.sequence_frame_count,
        settings.sequence_frame_digits,
    )


def _source_settings_signature(plan: CameraProjectionPlan) -> tuple[object, ...]:
    settings = plan.settings
    return (
        settings.width,
        settings.height,
        settings.output_directory.expanduser().resolve(strict=False),
        settings.texture_format,
        settings.margin_pixels,
        settings.sequence_start_frame,
        settings.sequence_frame_count,
        settings.sequence_frame_digits,
    )


def _frame_signature(plan: CameraProjectionPlan) -> tuple[int | None, ...]:
    return tuple(task.timeline_frame for task in plan.frame_tasks)


def _validate_grouped_plan_contract(
    plan: GroupedCameraProjectionPlan,
) -> None:
    """Revalidate builder assumptions for manually constructed grouped plans."""

    if not isinstance(plan, GroupedCameraProjectionPlan):
        raise TypeError("plan must be GroupedCameraProjectionPlan")
    if not plan.frame_tasks:
        raise CameraProjectionExecutionError(
            "GroupedCameraProjectionPlan contains no frame tasks"
        )

    resolved_paths: list[Path] = []
    for expected_index, task in enumerate(plan.frame_tasks):
        if task.task_index != expected_index:
            raise CameraProjectionExecutionError(
                "grouped frame task indices must be contiguous; "
                f"expected={expected_index}, got={task.task_index}"
            )
        if not isinstance(task.output_path, Path):
            raise TypeError("grouped output paths must be pathlib.Path")
        resolved_paths.append(
            task.output_path.expanduser().resolve(strict=False)
        )
    if len(resolved_paths) != len(set(resolved_paths)):
        raise CameraProjectionExecutionError(
            "GroupedCameraProjectionPlan contains duplicate output paths"
        )

    representative = plan.representative_plan
    expected_settings = _source_settings_signature(representative)
    if _settings_signature(plan) != expected_settings:
        raise CameraProjectionExecutionError(
            "grouped settings differ from representative CameraProjectionPlan"
        )

    expected_frames = _frame_signature(representative)
    actual_frames = tuple(task.timeline_frame for task in plan.frame_tasks)
    if actual_frames != expected_frames:
        raise CameraProjectionExecutionError(
            "grouped frame tasks differ from representative camera plan"
        )

    representative_scene = representative.scene_context
    source_paths: set[Path] = set()
    for index, source_plan in enumerate(plan.source_plans):
        if _source_settings_signature(source_plan) != expected_settings:
            raise CameraProjectionExecutionError(
                f"grouped source plan {index} uses incompatible BakeSettings"
            )
        if _frame_signature(source_plan) != expected_frames:
            raise CameraProjectionExecutionError(
                f"grouped source plan {index} uses incompatible frame tasks"
            )
        if source_plan.scene_context != representative_scene:
            raise CameraProjectionExecutionError(
                f"grouped source plan {index} uses incompatible SceneBakeContext"
            )
        if (
            source_plan.transparent_background
            != plan.transparent_background
        ):
            raise CameraProjectionExecutionError(
                f"grouped source plan {index} uses incompatible transparency"
            )
        source_paths.update(
            task.output_path.expanduser().resolve(strict=False)
            for task in source_plan.frame_tasks
        )

    collisions = tuple(path for path in resolved_paths if path in source_paths)
    if collisions:
        raise CameraProjectionExecutionError(
            f"grouped output paths collide with source outputs: {collisions}"
        )


def validate_grouped_camera_projection_request(
    source_objects: Tuple[Any, ...],
    plan: GroupedCameraProjectionPlan,
    execution_settings: BakeExecutionSettings,
    *,
    context: Any | None = None,
    scene: Any | None = None,
) -> GroupedCameraProjectionRuntime:
    """Validate the complete grouped request before output reservation."""

    if not isinstance(plan, GroupedCameraProjectionPlan):
        raise TypeError("plan must be GroupedCameraProjectionPlan")
    if (
        not isinstance(source_objects, tuple)
        or len(source_objects) != len(plan.source_plans)
        or len(source_objects) < 2
    ):
        raise ValueError("source_objects must match grouped source plans")
    if not isinstance(execution_settings, BakeExecutionSettings):
        raise TypeError("execution_settings must be BakeExecutionSettings")

    _validate_grouped_plan_contract(plan)

    source_identities = tuple(rna_identity(obj) for obj in source_objects)
    if len(source_identities) != len(set(source_identities)):
        raise CameraProjectionExecutionError(
            "grouped projection source_objects contain duplicate Blender objects"
        )

    names = tuple(object_name(obj) for obj in source_objects)
    if names != plan.source_object_ids:
        raise CameraProjectionExecutionError(
            "grouped projection source object order differs from the immutable plan; "
            f"expected={plan.source_object_ids}, actual={names}"
        )

    grouped_output_policy = resolve_projection_output_policy(
        execution_settings.projection_output_policy,
        plan.settings.texture_format,
    )

    resolved_bpy = None
    resolved_context = context
    resolved_scene = scene
    expected_scene_identity = None
    expected_renderer = None

    for source_obj, source_plan in zip(
        source_objects,
        plan.source_plans,
        strict=True,
    ):
        source_runtime: CameraProjectionRuntime = (
            validate_camera_projection_request(
                source_obj,
                source_plan,
                execution_settings,
                context=resolved_context,
                scene=resolved_scene,
            )
        )
        current_scene_identity = rna_identity(source_runtime.scene)
        if resolved_bpy is None:
            resolved_bpy = source_runtime.bpy_module
            resolved_context = source_runtime.context
            resolved_scene = source_runtime.scene
            expected_scene_identity = current_scene_identity
            expected_renderer = source_runtime.renderer
        else:
            if source_runtime.bpy_module is not resolved_bpy:
                raise CameraProjectionExecutionError(
                    "grouped projection sources resolved through different bpy modules"
                )
            if current_scene_identity != expected_scene_identity:
                raise CameraProjectionExecutionError(
                    "grouped projection sources resolved to different Blender scenes"
                )
            if source_runtime.renderer != expected_renderer:
                raise CameraProjectionExecutionError(
                    "grouped projection sources resolved to different render engines"
                )

        if source_runtime.output_policy != grouped_output_policy:
            raise CameraProjectionExecutionError(
                "grouped projection output policy differs from a source camera plan"
            )

    if (
        resolved_bpy is None
        or resolved_context is None
        or resolved_scene is None
        or expected_renderer is None
    ):
        raise CameraProjectionExecutionError(
            "grouped projection runtime did not resolve Blender state"
        )

    return GroupedCameraProjectionRuntime(
        source_objects=source_objects,
        plan=plan,
        execution_settings=execution_settings,
        bpy_module=resolved_bpy,
        context=resolved_context,
        scene=resolved_scene,
        renderer=expected_renderer,
        output_policy=grouped_output_policy,
    )


def validate_grouped_camera_projection_reservations(
    plan: GroupedCameraProjectionPlan,
    reservations: Iterable[AtomicOutputReservation],
) -> Tuple[AtomicOutputReservation, ...]:
    """Require one correctly ordered reservation per grouped frame task."""

    if not isinstance(plan, GroupedCameraProjectionPlan):
        raise TypeError("plan must be GroupedCameraProjectionPlan")

    resolved = tuple(reservations)
    if len(resolved) != len(plan.frame_tasks):
        raise CameraProjectionExecutionError(
            f"Expected {len(plan.frame_tasks)} grouped reservations, "
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
                f"Grouped task {task.task_index} expected '{expected}', "
                f"got '{reservation.final_path}'"
            )

    return resolved


__all__ = [
    "CameraProjectionExecutionError",
    "GroupedCameraProjectionRuntime",
    "object_name",
    "rna_identity",
    "validate_grouped_camera_projection_request",
    "validate_grouped_camera_projection_reservations",
]
