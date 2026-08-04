"""Immutable settings and results used by Blender texture executors."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Tuple

from ..geometry import DepthCameraProjectionSettings
from .camera_projection import TexturePlan
from .contracts import (
    require_finite_number,
    require_integer,
    require_non_empty_string,
)
from .depth_parallax import DepthParallaxSettings
from .projection_coverage import ProjectionCoveragePolicy
from .projection_layout import ProjectionContourMode
from .projection_output import ProjectionOutputPolicy


class A1TextureExportMode(str, Enum):
    """User-selected topology and texture representation for one Rewrite export."""

    NORMAL_UV_SEGMENTS = "NORMAL_UV_SEGMENTS"
    CAMERA_PROJECTION = "CAMERA_PROJECTION"
    DEPTH_CAMERA_PROJECTION = "DEPTH_CAMERA_PROJECTION"


@dataclass(frozen=True, slots=True)
class CameraProjectionInfluencePolicy:
    """Scene-ray participation retained while the source is camera-isolated."""

    include_scene_shadows: bool = True
    include_scene_reflection_transmission: bool = True
    world_affects_lighting_reflections: bool = True

    def __post_init__(self) -> None:
        for field_name in (
            "include_scene_shadows",
            "include_scene_reflection_transmission",
            "world_affects_lighting_reflections",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"{field_name} must be bool")


@dataclass(frozen=True, slots=True)
class BakeExecutionSettings:
    render_engine: str = "CYCLES"
    samples: int = 256
    use_clear: bool = True
    generated_color: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    color_mode: str = "RGBA"
    projection_alpha_threshold: float = 1.0 / 255.0
    projection_contour_mode: ProjectionContourMode = (
        ProjectionContourMode.SIMPLIFIED_CONCAVE
    )
    projection_contour_simplify_tolerance_pixels: float = 1.0
    projection_coverage_policy: ProjectionCoveragePolicy = ProjectionCoveragePolicy()
    projection_output_policy: ProjectionOutputPolicy = ProjectionOutputPolicy()
    texture_export_mode: A1TextureExportMode = (
        A1TextureExportMode.NORMAL_UV_SEGMENTS
    )
    # Appended to preserve the positional layout of the established settings contract.
    camera_influence_policy: CameraProjectionInfluencePolicy = (
        CameraProjectionInfluencePolicy()
    )
    # Appended for 0.81.0 so established positional construction remains stable.
    depth_projection: DepthCameraProjectionSettings = (
        DepthCameraProjectionSettings()
    )
    # Appended for 0.90.0. Zero preserves the complete 0.81.0 output contract.
    depth_parallax: DepthParallaxSettings = DepthParallaxSettings()

    def __post_init__(self) -> None:
        require_non_empty_string(self.render_engine, "render_engine")
        require_integer(self.samples, "samples", minimum=1)
        if not isinstance(self.use_clear, bool):
            raise TypeError("use_clear must be bool")
        if not isinstance(self.generated_color, tuple) or len(self.generated_color) != 4:
            raise ValueError("generated_color must contain four values")
        for index, value in enumerate(self.generated_color):
            require_finite_number(value, f"generated_color[{index}]")
        if self.color_mode not in {"BW", "RGB", "RGBA"}:
            raise ValueError("color_mode must be BW, RGB, or RGBA")
        require_finite_number(
            self.projection_alpha_threshold,
            "projection_alpha_threshold",
            minimum=0.0,
            maximum=1.0,
        )
        if not isinstance(self.projection_contour_mode, ProjectionContourMode):
            raise TypeError(
                "projection_contour_mode must be ProjectionContourMode"
            )
        require_finite_number(
            self.projection_contour_simplify_tolerance_pixels,
            "projection_contour_simplify_tolerance_pixels",
            minimum=0.0,
        )
        if not isinstance(
            self.projection_coverage_policy,
            ProjectionCoveragePolicy,
        ):
            raise TypeError(
                "projection_coverage_policy must be ProjectionCoveragePolicy"
            )
        if not isinstance(self.projection_output_policy, ProjectionOutputPolicy):
            raise TypeError(
                "projection_output_policy must be ProjectionOutputPolicy"
            )
        if not isinstance(self.texture_export_mode, A1TextureExportMode):
            raise TypeError("texture_export_mode must be A1TextureExportMode")
        if not isinstance(
            self.camera_influence_policy,
            CameraProjectionInfluencePolicy,
        ):
            raise TypeError(
                "camera_influence_policy must be CameraProjectionInfluencePolicy"
            )
        if not isinstance(self.depth_projection, DepthCameraProjectionSettings):
            raise TypeError(
                "depth_projection must be DepthCameraProjectionSettings"
            )
        if not isinstance(self.depth_parallax, DepthParallaxSettings):
            raise TypeError("depth_parallax must be DepthParallaxSettings")


@dataclass(frozen=True, slots=True)
class BakeArtifact:
    task_index: int
    timeline_frame: int | None
    image_name: str
    output_path: Path
    width: int
    height: int

    def __post_init__(self) -> None:
        require_integer(self.task_index, "task_index", minimum=0)
        if self.timeline_frame is not None:
            require_integer(self.timeline_frame, "timeline_frame", minimum=0)
        require_non_empty_string(self.image_name, "image_name")
        if not isinstance(self.output_path, Path):
            raise TypeError("output_path must be pathlib.Path")
        require_integer(self.width, "width", minimum=1)
        require_integer(self.height, "height", minimum=1)


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
        if not all(isinstance(artifact, BakeArtifact) for artifact in self.artifacts):
            raise TypeError("artifacts must contain BakeArtifact values")
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
                raise ValueError("artifact.output_path does not match plan")

    @property
    def representative_artifact(self) -> BakeArtifact:
        return self.artifacts[self.plan.representative_task_index]
