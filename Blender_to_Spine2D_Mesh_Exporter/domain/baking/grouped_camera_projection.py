"""Immutable planning for one depth-correct camera render of connected B4 objects."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Tuple

from .camera_projection import CameraProjectionPlan
from .model import BakeFrameTask, BakeSettings, sanitize_filename_stem


class GroupedCameraProjectionPlanError(ValueError):
    """Raised when connected B4 plans cannot share one camera render contract."""


@dataclass(frozen=True, slots=True)
class GroupedCameraProjectionPlan:
    """One staged image sequence containing all connected B4 sources together."""

    group_id: str
    source_object_ids: Tuple[str, ...]
    source_plans: Tuple[CameraProjectionPlan, ...]
    settings: BakeSettings
    frame_tasks: Tuple[BakeFrameTask, ...]
    transparent_background: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.group_id, str) or not self.group_id.strip():
            raise ValueError("group_id must be a non-empty string")
        if (
            not isinstance(self.source_object_ids, tuple)
            or len(self.source_object_ids) < 2
            or not all(
                isinstance(value, str) and value.strip()
                for value in self.source_object_ids
            )
        ):
            raise ValueError(
                "source_object_ids must contain at least two non-empty strings"
            )
        if len(self.source_object_ids) != len(set(self.source_object_ids)):
            raise ValueError("source_object_ids must be unique")
        if (
            not isinstance(self.source_plans, tuple)
            or len(self.source_plans) != len(self.source_object_ids)
            or not all(
                isinstance(plan, CameraProjectionPlan) for plan in self.source_plans
            )
        ):
            raise TypeError(
                "source_plans must match source_object_ids and contain CameraProjectionPlan"
            )
        if tuple(plan.source_object_id for plan in self.source_plans) != (
            self.source_object_ids
        ):
            raise ValueError("source_plans do not match source_object_ids")
        if not isinstance(self.settings, BakeSettings):
            raise TypeError("settings must be BakeSettings")
        if not isinstance(self.frame_tasks, tuple) or not self.frame_tasks:
            raise ValueError("frame_tasks must be a non-empty tuple")
        if not all(isinstance(task, BakeFrameTask) for task in self.frame_tasks):
            raise TypeError("frame_tasks must contain BakeFrameTask values")
        if tuple(task.task_index for task in self.frame_tasks) != tuple(
            range(len(self.frame_tasks))
        ):
            raise ValueError("frame task indices must be ordered and dense")
        if not isinstance(self.transparent_background, bool):
            raise TypeError("transparent_background must be bool")

    @property
    def representative_plan(self) -> CameraProjectionPlan:
        return self.source_plans[0]

    @property
    def camera_object_id(self) -> str:
        return self.representative_plan.camera_object_id

    @property
    def sequence(self) -> bool:
        return any(task.timeline_frame is not None for task in self.frame_tasks)

    @property
    def representative_task(self) -> BakeFrameTask:
        return self.frame_tasks[0]


def _frame_signature(plan: CameraProjectionPlan) -> tuple[int | None, ...]:
    return tuple(task.timeline_frame for task in plan.frame_tasks)


def _settings_signature(settings: BakeSettings) -> tuple[object, ...]:
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


def _build_group_frame_tasks(settings: BakeSettings) -> Tuple[BakeFrameTask, ...]:
    stem = sanitize_filename_stem(settings.output_stem)
    extension = settings.texture_format.extension
    if settings.sequence_frame_count == 0:
        image_name = f"{stem}_Baked"
        return (
            BakeFrameTask(
                task_index=0,
                timeline_frame=None,
                image_name=image_name,
                output_path=settings.output_directory / f"{image_name}{extension}",
            ),
        )

    return tuple(
        BakeFrameTask(
            task_index=task_index,
            timeline_frame=settings.sequence_start_frame + task_index,
            image_name=(
                f"{stem}_Baked_"
                f"{settings.sequence_start_frame + task_index:0{settings.sequence_frame_digits}d}"
            ),
            output_path=settings.output_directory
            / (
                f"{stem}_Baked_"
                f"{settings.sequence_start_frame + task_index:0{settings.sequence_frame_digits}d}"
                f"{extension}"
            ),
        )
        for task_index in range(settings.sequence_frame_count)
    )


def build_grouped_camera_projection_plan(
    plans: Tuple[CameraProjectionPlan, ...],
    *,
    group_id: str,
    output_stem: str,
) -> GroupedCameraProjectionPlan:
    """Validate compatible source plans and build one collision-free grouped plan."""

    if not isinstance(plans, tuple) or len(plans) < 2:
        raise ValueError("plans must contain at least two CameraProjectionPlan values")
    if not all(isinstance(plan, CameraProjectionPlan) for plan in plans):
        raise TypeError("plans must contain CameraProjectionPlan values")
    if not isinstance(group_id, str) or not group_id.strip():
        raise ValueError("group_id must be a non-empty string")
    if not isinstance(output_stem, str) or not output_stem.strip():
        raise ValueError("output_stem must be a non-empty string")

    first = plans[0]
    first_settings_signature = _settings_signature(first.settings)
    first_frame_signature = _frame_signature(first)
    first_scene_context = first.scene_context
    incompatibilities: list[str] = []
    for index, plan in enumerate(plans[1:], start=1):
        if _settings_signature(plan.settings) != first_settings_signature:
            incompatibilities.append(f"plan[{index}].settings")
        if _frame_signature(plan) != first_frame_signature:
            incompatibilities.append(f"plan[{index}].frame_tasks")
        if plan.scene_context != first_scene_context:
            incompatibilities.append(f"plan[{index}].scene_context")
        if plan.transparent_background != first.transparent_background:
            incompatibilities.append(f"plan[{index}].transparent_background")
    if incompatibilities:
        raise GroupedCameraProjectionPlanError(
            "connected B4 plans do not share one render contract: "
            + ", ".join(incompatibilities)
        )

    source_ids = tuple(plan.source_object_id for plan in plans)
    if len(source_ids) != len(set(source_ids)):
        raise GroupedCameraProjectionPlanError(
            "connected B4 plans contain duplicate source_object_id values"
        )

    settings = replace(
        first.settings,
        output_stem=sanitize_filename_stem(output_stem),
    )
    frame_tasks = _build_group_frame_tasks(settings)
    if tuple(task.timeline_frame for task in frame_tasks) != first_frame_signature:
        raise GroupedCameraProjectionPlanError(
            "grouped frame tasks do not match source camera plans"
        )
    source_paths = {
        task.output_path.expanduser().resolve(strict=False)
        for plan in plans
        for task in plan.frame_tasks
    }
    grouped_paths = tuple(
        task.output_path.expanduser().resolve(strict=False) for task in frame_tasks
    )
    collisions = tuple(path for path in grouped_paths if path in source_paths)
    if collisions:
        raise GroupedCameraProjectionPlanError(
            f"grouped output paths collide with source outputs: {collisions}"
        )

    return GroupedCameraProjectionPlan(
        group_id=group_id,
        source_object_ids=source_ids,
        source_plans=plans,
        settings=settings,
        frame_tasks=frame_tasks,
        transparent_background=first.transparent_background,
    )


def grouped_projection_output_paths(
    plan: GroupedCameraProjectionPlan,
) -> Tuple[Path, ...]:
    if not isinstance(plan, GroupedCameraProjectionPlan):
        raise TypeError("plan must be GroupedCameraProjectionPlan")
    return tuple(task.output_path for task in plan.frame_tasks)


__all__ = [
    "GroupedCameraProjectionPlan",
    "GroupedCameraProjectionPlanError",
    "build_grouped_camera_projection_plan",
    "grouped_projection_output_paths",
]
