"""Explicit Normal/UV-segment planning for source- and camera-context materials.

Blender object baking evaluates the original shader graph at surface points while the
``uv_layer`` argument selects only the destination layout. A material may therefore
need object, normal, view, reflection, active-camera, or bump-only displacement context
without requiring the exported mesh to become a full-frame Camera Projection attachment.

This module keeps those concerns separate:

* material evaluation uses a CAMERA-scoped COMBINED object-bake pass;
* exported geometry remains the ordinary Normal / UV Segments topology;
* volume and true render displacement remain rejected;
* bump-only Material Output displacement is accepted only when the production capability
  router has already proved the live Blender material uses Bump Only rather than geometry
  displacement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from .context import ObjectBakeContext, SceneBakeContext
from .graph import MaterialSemanticChannel
from .model import (
    BakeEvaluationScope,
    BakeFrameTask,
    BakeMode,
    BakePlan,
    BakePlanError,
    BakeSettings,
    BakeStrategyId,
    MaterialAnalysis,
    MaterialKind,
    MaterialPreparationMode,
    MaterialSlotPreparation,
    ObjectMaterialAnalysis,
    sanitize_filename_stem,
)
from .strategies import (
    AlphaBakeStrategy,
    BakeStrategyRegistry,
    EmissionBakeStrategy,
    SceneCombinedBakeStrategy,
    SurfaceColorBakeStrategy,
    resolve_bake_strategy_plan,
)


_APPEARANCE_CHANNELS = frozenset(
    {
        MaterialSemanticChannel.SURFACE_COLOR,
        MaterialSemanticChannel.SURFACE_EMISSION,
    }
)


def _appearance_channels(
    slots: Tuple[MaterialAnalysis, ...],
) -> Tuple[MaterialSemanticChannel, ...]:
    channels = {
        channel
        for slot in slots
        for channel in slot.semantic_channels
        if channel in _APPEARANCE_CHANNELS
    }
    if not channels:
        channels.add(MaterialSemanticChannel.SURFACE_COLOR)
    return tuple(sorted(channels, key=lambda value: value.value))


def _material_preparations(
    matched_slots: Tuple[MaterialAnalysis, ...],
    usable_slots: Tuple[MaterialAnalysis, ...],
) -> Tuple[MaterialSlotPreparation, ...]:
    matched_indices = {slot.slot_index for slot in matched_slots}
    return tuple(
        MaterialSlotPreparation(
            slot_index=slot.slot_index,
            mode=(
                MaterialPreparationMode.PRESERVE
                if slot.slot_index in matched_indices
                else MaterialPreparationMode.ZERO_TO_EMISSION
            ),
        )
        for slot in usable_slots
    )


@dataclass(frozen=True, slots=True)
class NormalUvCameraCombinedBakeStrategy:
    """Evaluate camera/source-context appearance through object UV COMBINED bake."""

    strategy_id: BakeStrategyId = BakeStrategyId.CAMERA_COMBINED
    priority: int = 25
    evaluation_scope: BakeEvaluationScope = BakeEvaluationScope.CAMERA

    def supports(self, slot: MaterialAnalysis) -> bool:
        if not isinstance(slot, MaterialAnalysis):
            raise TypeError("slot must be MaterialAnalysis")
        return bool(set(slot.semantic_channels) & _APPEARANCE_CHANNELS)

    def semantic_channels(
        self,
        slots: Tuple[MaterialAnalysis, ...],
    ) -> Tuple[MaterialSemanticChannel, ...]:
        if not isinstance(slots, tuple) or not all(
            isinstance(slot, MaterialAnalysis) for slot in slots
        ):
            raise TypeError("slots must contain MaterialAnalysis values")
        return _appearance_channels(slots)

    def select_mode(
        self,
        slots: Tuple[MaterialAnalysis, ...],
        settings: BakeSettings,
    ) -> BakeMode:
        if not isinstance(slots, tuple) or not slots:
            raise ValueError("slots must be a non-empty tuple")
        if not all(isinstance(slot, MaterialAnalysis) for slot in slots):
            raise TypeError("slots must contain MaterialAnalysis values")
        if not isinstance(settings, BakeSettings):
            raise TypeError("settings must be BakeSettings")
        return BakeMode.COMBINED

    def material_preparations(
        self,
        matched_slots: Tuple[MaterialAnalysis, ...],
        usable_slots: Tuple[MaterialAnalysis, ...],
    ) -> Tuple[MaterialSlotPreparation, ...]:
        if not isinstance(matched_slots, tuple) or not all(
            isinstance(slot, MaterialAnalysis) for slot in matched_slots
        ):
            raise TypeError("matched_slots must contain MaterialAnalysis values")
        if not isinstance(usable_slots, tuple) or not all(
            isinstance(slot, MaterialAnalysis) for slot in usable_slots
        ):
            raise TypeError("usable_slots must contain MaterialAnalysis values")
        return _material_preparations(matched_slots, usable_slots)


def _frame_tasks(settings: BakeSettings) -> Tuple[BakeFrameTask, ...]:
    """Build the same deterministic frame/output contract as the generic planner."""

    if not isinstance(settings, BakeSettings):
        raise TypeError("settings must be BakeSettings")
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


def _validate_normal_uv_channels(
    analysis: ObjectMaterialAnalysis,
    *,
    allow_bump_displacement: bool,
) -> None:
    """Reject channels that cannot be represented by the approved Normal/UV route.

    ``allow_bump_displacement`` is deliberately explicit and fail-closed. The immutable
    material analysis only knows that Material Output -> Displacement is connected; it
    cannot know whether Blender evaluates that connection as Bump Only or true geometry
    displacement. Only the live production capability router may set this flag after it
    has classified every displacement finding and proved that no true-displacement
    blocker remains.
    """

    if not isinstance(analysis, ObjectMaterialAnalysis):
        raise TypeError("analysis must be ObjectMaterialAnalysis")
    if not isinstance(allow_bump_displacement, bool):
        raise TypeError("allow_bump_displacement must be bool")

    forbidden_channels = {MaterialSemanticChannel.VOLUME}
    if not allow_bump_displacement:
        forbidden_channels.add(MaterialSemanticChannel.DISPLACEMENT)

    invalid = tuple(
        (
            slot.slot_index,
            slot.material_name or f"slot-{slot.slot_index}",
            tuple(
                sorted(
                    channel.value
                    for channel in set(slot.semantic_channels) & forbidden_channels
                )
            ),
        )
        for slot in analysis.slots
        if slot.kind is not MaterialKind.EMPTY
        and set(slot.semantic_channels) & forbidden_channels
    )
    if invalid:
        raise BakePlanError(
            "Normal — UV Segments cannot represent volume or true render displacement "
            f"outputs as surface textures: {invalid}"
        )


def build_normal_uv_camera_context_plan(
    analysis: ObjectMaterialAnalysis,
    settings: BakeSettings,
    *,
    object_context: ObjectBakeContext,
    scene_context: SceneBakeContext,
    allow_bump_displacement: bool = False,
) -> BakePlan:
    """Build a Normal/UV-segment plan for camera/source-context shader evaluation."""

    if not isinstance(analysis, ObjectMaterialAnalysis):
        raise TypeError("analysis must be ObjectMaterialAnalysis")
    if not isinstance(settings, BakeSettings):
        raise TypeError("settings must be BakeSettings")
    if not isinstance(object_context, ObjectBakeContext):
        raise TypeError("object_context must be ObjectBakeContext")
    if not isinstance(scene_context, SceneBakeContext):
        raise TypeError("scene_context must be SceneBakeContext")
    if not isinstance(allow_bump_displacement, bool):
        raise TypeError("allow_bump_displacement must be bool")
    if object_context.source_object_id != analysis.source_object_id:
        raise BakePlanError("object context does not match material analysis")
    if scene_context.camera is None:
        raise BakePlanError(
            "camera/source-context UV baking requires an active scene camera for "
            f"'{analysis.source_object_id}'"
        )

    _validate_normal_uv_channels(
        analysis,
        allow_bump_displacement=allow_bump_displacement,
    )
    registry = BakeStrategyRegistry(
        strategies=(
            NormalUvCameraCombinedBakeStrategy(),
            SceneCombinedBakeStrategy(),
            SurfaceColorBakeStrategy(),
            EmissionBakeStrategy(),
            AlphaBakeStrategy(),
        )
    )
    passes, composite = resolve_bake_strategy_plan(
        analysis,
        settings,
        registry=registry,
        object_context=object_context,
        scene_context=scene_context,
    )
    tasks = _frame_tasks(settings)
    return BakePlan(
        source_object_id=analysis.source_object_id,
        settings=settings,
        material_analysis=analysis,
        bake_mode=passes[0].bake_mode,
        frame_tasks=tasks,
        representative_task_index=0,
        passes=passes,
        composite=composite,
        object_context=object_context,
        scene_context=scene_context,
    )


__all__ = [
    "NormalUvCameraCombinedBakeStrategy",
    "build_normal_uv_camera_context_plan",
]
