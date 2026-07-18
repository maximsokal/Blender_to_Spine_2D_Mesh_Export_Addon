"""Immutable settings and results used by Blender texture executors."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Tuple

from .camera_projection import TexturePlan


@dataclass(frozen=True, slots=True)
class BakeExecutionSettings:
    render_engine: str = "CYCLES"
    samples: int = 256
    use_clear: bool = True
    generated_color: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    color_mode: str = "RGBA"
    projection_alpha_threshold: float = 1.0 / 255.0

    def __post_init__(self) -> None:
        if not isinstance(self.render_engine, str) or not self.render_engine.strip():
            raise ValueError("render_engine must be a non-empty string")
        if not isinstance(self.samples, int) or self.samples < 1:
            raise ValueError("samples must be a positive integer")
        if not isinstance(self.use_clear, bool):
            raise TypeError("use_clear must be bool")
        if not isinstance(self.generated_color, tuple) or len(self.generated_color) != 4:
            raise ValueError("generated_color must contain four values")
        if not all(
            isinstance(value, (int, float)) and isfinite(float(value))
            for value in self.generated_color
        ):
            raise ValueError("generated_color must contain finite numeric values")
        if self.color_mode not in {"BW", "RGB", "RGBA"}:
            raise ValueError("color_mode must be BW, RGB, or RGBA")
        if (
            not isinstance(self.projection_alpha_threshold, (int, float))
            or not isfinite(float(self.projection_alpha_threshold))
            or not 0.0 <= float(self.projection_alpha_threshold) <= 1.0
        ):
            raise ValueError(
                "projection_alpha_threshold must be finite and in [0, 1]"
            )


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
    plan: TexturePlan
    artifacts: Tuple[BakeArtifact, ...]

    def __post_init__(self) -> None:
        from .model import BakePlan

        if not isinstance(self.plan, BakePlan):
            raise TypeError("plan must be BakePlan or CameraProjectionPlan")
        if not isinstance(self.artifacts, tuple):
            raise TypeError("artifacts must be tuple")
        if len(self.artifacts) != len(self.plan.frame_tasks):
            raise ValueError("one artifact is required for every texture frame task")
        for task, artifact in zip(self.plan.frame_tasks, self.artifacts):
            if artifact.task_index != task.task_index:
                raise ValueError("artifact task_index does not match plan")
            if artifact.timeline_frame != task.timeline_frame:
                raise ValueError("artifact timeline_frame does not match plan")
            if artifact.image_name != task.image_name:
                raise ValueError("artifact image_name does not match plan")
            if artifact.output_path != task.output_path.resolve(strict=False):
                raise ValueError("artifact output_path does not match plan")

    @property
    def representative_artifact(self) -> BakeArtifact:
        return self.artifacts[self.plan.representative_task_index]
