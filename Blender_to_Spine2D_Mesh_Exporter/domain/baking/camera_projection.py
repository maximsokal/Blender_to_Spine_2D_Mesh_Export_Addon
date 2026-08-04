"""Immutable planning for camera-rendered Spine projection textures.

Camera-dependent shaders and volume cannot be represented by Blender's object UV bake
operator. This module routes the complete object through deterministic camera renders and
preserves one typed frame/output contract for the front and optional parallax views.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from math import isfinite
from pathlib import Path
from typing import Tuple, TypeAlias

from .context import ObjectBakeContext, SceneBakeContext
from .graph import MaterialDependencyKind, MaterialSemanticChannel
from .model import (
    BakeCompositePlan,
    BakeEvaluationScope,
    BakeFrameTask,
    BakeMode,
    BakePassPlan,
    BakePlan,
    BakePlanError,
    BakeSettings,
    BakeStrategyId,
    MaterialKind,
    ObjectMaterialAnalysis,
    TextureFormat,
    build_bake_plan,
)
from .output_naming import sanitize_filename_stem


class CameraProjectionMode(str, Enum):
    """Geometry policy used by camera-render projection."""

    FULL_FRAME_QUAD = "FULL_FRAME_QUAD"


_CAMERA_PROJECTION_DEPENDENCIES = frozenset(
    {
        MaterialDependencyKind.CAMERA,
        MaterialDependencyKind.VIEW,
        MaterialDependencyKind.REFLECTION,
        MaterialDependencyKind.TRANSMISSION,
    }
)
_CAMERA_PROJECTION_CHANNELS = frozenset(
    {
        MaterialSemanticChannel.VOLUME,
        MaterialSemanticChannel.DISPLACEMENT,
    }
)


def _build_frame_tasks(settings: BakeSettings) -> Tuple[BakeFrameTask, ...]:
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

    tasks = []
    for task_index in range(settings.sequence_frame_count):
        timeline_frame = settings.sequence_start_frame + task_index
        suffix = f"{timeline_frame:0{settings.sequence_frame_digits}d}"
        image_name = f"{stem}_Baked_{suffix}"
        tasks.append(
            BakeFrameTask(
                task_index=task_index,
                timeline_frame=timeline_frame,
                image_name=image_name,
                output_path=settings.output_directory / f"{image_name}{extension}",
            )
        )
    return tuple(tasks)


def requires_camera_projection(analysis: ObjectMaterialAnalysis) -> bool:
    """Return whether one object must be evaluated from the active camera."""

    if not isinstance(analysis, ObjectMaterialAnalysis):
        raise TypeError("analysis must be ObjectMaterialAnalysis")
    for slot in analysis.slots:
        if slot.kind is MaterialKind.EMPTY:
            continue
        if set(slot.dependencies) & _CAMERA_PROJECTION_DEPENDENCIES:
            return True
        if set(slot.semantic_channels) & _CAMERA_PROJECTION_CHANNELS:
            return True
    return False


@dataclass(frozen=True, slots=True)
class CameraProjectionPlan(BakePlan):
    """BakePlan subtype executed by the camera-render pipeline.

    ``view_id`` is ``FRONT`` for the active-camera render. Depth parallax reserve plans
    append a stable view id, a camera-world override, and a fitted lens scale. The front
    defaults preserve the complete pre-0.90.0 plan contract.
    """

    projection_mode: CameraProjectionMode = CameraProjectionMode.FULL_FRAME_QUAD
    transparent_background: bool = True
    isolate_source_to_camera: bool = True
    view_id: str = "FRONT"
    camera_world_matrix_override: Tuple[float, ...] | None = None
    lens_scale: float = 1.0

    def __post_init__(self) -> None:
        # ``dataclass(slots=True)`` returns a replacement class object. Zero-argument
        # ``super()`` may therefore capture the pre-replacement class in CPython.
        BakePlan.__post_init__(self)
        if not isinstance(self.projection_mode, CameraProjectionMode):
            raise TypeError("projection_mode must be CameraProjectionMode")
        for field_name in ("transparent_background", "isolate_source_to_camera"):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"{field_name} must be bool")
        if not isinstance(self.view_id, str) or not self.view_id.strip():
            raise ValueError("view_id must be a non-empty string")
        object.__setattr__(self, "view_id", self.view_id.strip().upper())
        if self.camera_world_matrix_override is not None:
            matrix = self.camera_world_matrix_override
            if not isinstance(matrix, tuple) or len(matrix) != 16:
                raise TypeError(
                    "camera_world_matrix_override must be a sixteen-value tuple or None"
                )
            if not all(isfinite(float(value)) for value in matrix):
                raise ValueError(
                    "camera_world_matrix_override contains non-finite values"
                )
            object.__setattr__(
                self,
                "camera_world_matrix_override",
                tuple(float(value) for value in matrix),
            )
        if isinstance(self.lens_scale, bool) or not isinstance(
            self.lens_scale,
            (int, float),
        ):
            raise TypeError("lens_scale must be a finite number")
        resolved_lens_scale = float(self.lens_scale)
        if (
            not isfinite(resolved_lens_scale)
            or resolved_lens_scale <= 0.0
            or resolved_lens_scale > 1.0
        ):
            raise ValueError("lens_scale must be finite in (0, 1]")
        object.__setattr__(self, "lens_scale", resolved_lens_scale)
        if self.view_id == "FRONT":
            if self.camera_world_matrix_override is not None:
                raise ValueError("FRONT plan cannot override the active camera matrix")
            if self.lens_scale != 1.0:
                raise ValueError("FRONT plan must keep lens_scale=1")
        elif self.camera_world_matrix_override is None:
            raise ValueError(
                "non-FRONT camera projection plans require a camera matrix override"
            )
        if self.scene_context is None or self.scene_context.camera is None:
            raise ValueError("CameraProjectionPlan requires an active camera snapshot")
        if self.object_context is None:
            raise ValueError("CameraProjectionPlan requires object_context")
        if len(self.passes) != 1:
            raise ValueError("CameraProjectionPlan requires exactly one synthetic pass")
        projection_pass = self.passes[0]
        if projection_pass.strategy_id is not BakeStrategyId.CAMERA_COMBINED:
            raise ValueError("CameraProjectionPlan pass must use CAMERA_COMBINED")
        if projection_pass.evaluation_scope is not BakeEvaluationScope.CAMERA:
            raise ValueError("CameraProjectionPlan pass must use CAMERA evaluation scope")
        if self.settings.texture_format is TextureFormat.JPEG:
            raise ValueError("Camera projection requires an alpha-capable texture format")

    @property
    def camera_object_id(self) -> str:
        assert self.scene_context is not None and self.scene_context.camera is not None
        return self.scene_context.camera.object_id

    @property
    def virtual_view(self) -> bool:
        return self.view_id != "FRONT"


TexturePlan: TypeAlias = BakePlan | CameraProjectionPlan


def build_camera_projection_plan(
    analysis: ObjectMaterialAnalysis,
    settings: BakeSettings,
    *,
    object_context: ObjectBakeContext,
    scene_context: SceneBakeContext,
) -> CameraProjectionPlan:
    if not isinstance(analysis, ObjectMaterialAnalysis):
        raise TypeError("analysis must be ObjectMaterialAnalysis")
    if not isinstance(settings, BakeSettings):
        raise TypeError("settings must be BakeSettings")
    if not isinstance(object_context, ObjectBakeContext):
        raise TypeError("object_context must be ObjectBakeContext")
    if not isinstance(scene_context, SceneBakeContext):
        raise TypeError("scene_context must be SceneBakeContext")
    if object_context.source_object_id != analysis.source_object_id:
        raise BakePlanError("object context does not match material analysis")
    if scene_context.camera is None:
        raise BakePlanError(
            "camera-render projection requires an active scene camera for "
            f"'{analysis.source_object_id}'"
        )
    if settings.texture_format is TextureFormat.JPEG:
        raise BakePlanError(
            "camera-render projection requires PNG, WEBP, or OPEN_EXR because the "
            "background must remain transparent"
        )

    usable = tuple(slot for slot in analysis.slots if slot.kind is not MaterialKind.EMPTY)
    if not usable:
        raise BakePlanError("object has no usable materials")
    channels = tuple(
        sorted(
            {channel for slot in usable for channel in slot.semantic_channels}
            or {MaterialSemanticChannel.SURFACE_COLOR},
            key=lambda value: value.value,
        )
    )
    projection_pass = BakePassPlan(
        pass_index=0,
        strategy_id=BakeStrategyId.CAMERA_COMBINED,
        bake_mode=BakeMode.COMBINED,
        material_slot_indices=tuple(slot.slot_index for slot in usable),
        semantic_channels=channels,
        evaluation_scope=BakeEvaluationScope.CAMERA,
    )
    tasks = _build_frame_tasks(settings)
    return CameraProjectionPlan(
        source_object_id=analysis.source_object_id,
        settings=settings,
        material_analysis=analysis,
        bake_mode=BakeMode.COMBINED,
        frame_tasks=tasks,
        representative_task_index=0,
        passes=(projection_pass,),
        composite=BakeCompositePlan(),
        object_context=object_context,
        scene_context=scene_context,
    )


def build_camera_projection_view_plan(
    front_plan: CameraProjectionPlan,
    *,
    view_id: str,
    camera_world_matrix: Tuple[float, ...],
    lens_scale: float,
) -> CameraProjectionPlan:
    """Clone one front plan into a deterministic virtual-view output namespace."""

    if not isinstance(front_plan, CameraProjectionPlan):
        raise TypeError("front_plan must be CameraProjectionPlan")
    if front_plan.view_id != "FRONT":
        raise ValueError("front_plan must own the FRONT view")
    if not isinstance(view_id, str) or not view_id.strip():
        raise ValueError("view_id must be a non-empty string")
    normalized_view_id = view_id.strip().upper()
    if normalized_view_id == "FRONT":
        raise ValueError("reserve view_id cannot be FRONT")
    suffix = sanitize_filename_stem(normalized_view_id)
    settings = replace(
        front_plan.settings,
        output_stem=(
            f"{front_plan.settings.output_stem}_Parallax_{suffix}"
        ),
    )
    return replace(
        front_plan,
        settings=settings,
        frame_tasks=_build_frame_tasks(settings),
        view_id=normalized_view_id,
        camera_world_matrix_override=tuple(camera_world_matrix),
        lens_scale=lens_scale,
    )


def build_texture_plan(
    analysis: ObjectMaterialAnalysis,
    settings: BakeSettings,
    *,
    object_context: ObjectBakeContext | None = None,
    scene_context: SceneBakeContext | None = None,
) -> TexturePlan:
    """Select object UV baking or full-frame camera projection automatically."""

    if requires_camera_projection(analysis):
        if object_context is None or scene_context is None:
            raise BakePlanError(
                "camera-render projection requires immutable object and scene contexts"
            )
        return build_camera_projection_plan(
            analysis,
            settings,
            object_context=object_context,
            scene_context=scene_context,
        )
    return build_bake_plan(
        analysis,
        settings,
        object_context=object_context,
        scene_context=scene_context,
    )


def texture_plan_output_paths(plan: TexturePlan) -> Tuple[Path, ...]:
    if not isinstance(plan, BakePlan):
        raise TypeError("plan must be BakePlan or CameraProjectionPlan")
    return tuple(task.output_path for task in plan.frame_tasks)


__all__ = [
    "CameraProjectionMode",
    "CameraProjectionPlan",
    "TexturePlan",
    "build_camera_projection_plan",
    "build_camera_projection_view_plan",
    "build_texture_plan",
    "requires_camera_projection",
    "texture_plan_output_paths",
]
