"""Immutable planning for camera-rendered Spine projection textures.

Camera-dependent shaders and volume cannot be represented by Blender's object UV bake
operator. This module routes the complete object through a deterministic active-camera
render and a full-frame Spine quad while preserving the frame/output contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
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
    sanitize_filename_stem,
)


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

    The inherited ``bake_mode`` and explicit synthetic pass preserve one typed
    frame/output plan. They are never passed to ``bpy.ops.object.bake``;
    execution dispatches by this concrete type and invokes
    ``bpy.ops.render.render`` instead.
    """

    projection_mode: CameraProjectionMode = CameraProjectionMode.FULL_FRAME_QUAD
    transparent_background: bool = True
    isolate_source_to_camera: bool = True

    def __post_init__(self) -> None:
        # ``dataclass(slots=True)`` returns a replacement class object. Zero-argument
        # ``super()`` may therefore capture the pre-replacement class in CPython; calling
        # the concrete base validator is deterministic for this frozen slots subclass.
        BakePlan.__post_init__(self)
        if not isinstance(self.projection_mode, CameraProjectionMode):
            raise TypeError("projection_mode must be CameraProjectionMode")
        for field_name in ("transparent_background", "isolate_source_to_camera"):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"{field_name} must be bool")
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
