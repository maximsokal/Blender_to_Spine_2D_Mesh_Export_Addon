"""Immutable results returned by the Blender bake executor."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

from .model import BakePlan


@dataclass(frozen=True, slots=True)
class BakeArtifact:
    task_index: int
    timeline_frame: int | None
    image_name: str
    output_path: Path
    width: int
    height: int

    def __post_init__(self) -> None:
        if not isinstance(self.task_index, int) or self.task_index < 0:
            raise ValueError("task_index must be a non-negative integer")
        if self.timeline_frame is not None and (
            not isinstance(self.timeline_frame, int) or self.timeline_frame < 0
        ):
            raise ValueError("timeline_frame must be a non-negative integer or None")
        if not isinstance(self.image_name, str) or not self.image_name.strip():
            raise ValueError("image_name must be a non-empty string")
        if not isinstance(self.output_path, Path):
            raise TypeError("output_path must be pathlib.Path")
        if not isinstance(self.width, int) or self.width <= 0:
            raise ValueError("width must be a positive integer")
        if not isinstance(self.height, int) or self.height <= 0:
            raise ValueError("height must be a positive integer")


@dataclass(frozen=True, slots=True)
class BakeExecutionResult:
    plan: BakePlan
    artifacts: Tuple[BakeArtifact, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.plan, BakePlan):
            raise TypeError("plan must be BakePlan")
        if not isinstance(self.artifacts, tuple):
            raise TypeError("artifacts must be tuple")
        if len(self.artifacts) != len(self.plan.frame_tasks):
            raise ValueError("one artifact is required for every bake frame task")
        for task, artifact in zip(self.plan.frame_tasks, self.artifacts):
            if artifact.task_index != task.task_index:
                raise ValueError("artifact task_index does not match BakePlan")
            if artifact.timeline_frame != task.timeline_frame:
                raise ValueError("artifact timeline_frame does not match BakePlan")
            if artifact.image_name != task.image_name:
                raise ValueError("artifact image_name does not match BakePlan")
            if artifact.output_path != task.output_path.resolve(strict=False):
                raise ValueError("artifact output_path does not match BakePlan")

    @property
    def representative_artifact(self) -> BakeArtifact:
        return self.artifacts[self.plan.representative_task_index]
