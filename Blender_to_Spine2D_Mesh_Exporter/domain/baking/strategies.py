"""Deterministic strategy selection for semantic material bake requirements."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol, Tuple

from .context import ObjectBakeContext, SceneBakeContext
from .graph import MaterialDependencyKind, MaterialSemanticChannel
from .model import (
    BakeCompositeMode,
    BakeCompositePlan,
    BakeEvaluationScope,
    BakeMaterialPolicy,
    BakeMode,
    BakePassPlan,
    BakePlanError,
    BakeSettings,
    BakeStrategyId,
    MaterialAnalysis,
    MaterialKind,
    MaterialPreparationMode,
    MaterialSlotPreparation,
    ObjectMaterialAnalysis,
)


_CAMERA_DEPENDENCIES = frozenset(
    {
        MaterialDependencyKind.CAMERA,
        MaterialDependencyKind.VIEW,
        MaterialDependencyKind.REFLECTION,
        MaterialDependencyKind.TRANSMISSION,
    }
)
_SCENE_DEPENDENCIES = frozenset(
    {
        MaterialDependencyKind.WORLD,
        MaterialDependencyKind.LIGHTING,
        MaterialDependencyKind.OCCLUSION,
        MaterialDependencyKind.SCENE_OBJECTS,
    }
)
_APPEARANCE_CHANNELS = frozenset(
    {
        MaterialSemanticChannel.SURFACE_COLOR,
        MaterialSemanticChannel.SURFACE_EMISSION,
    }
)


class BakeStrategy(Protocol):
    """One independently extensible strategy registered with the resolver."""

    strategy_id: BakeStrategyId
    priority: int
    evaluation_scope: BakeEvaluationScope

    def supports(self, slot: MaterialAnalysis) -> bool:
        ...

    def semantic_channels(
        self,
        slots: Tuple[MaterialAnalysis, ...],
    ) -> Tuple[MaterialSemanticChannel, ...]:
        ...

    def select_mode(
        self,
        slots: Tuple[MaterialAnalysis, ...],
        settings: BakeSettings,
    ) -> BakeMode:
        ...

    def material_preparations(
        self,
        matched_slots: Tuple[MaterialAnalysis, ...],
        usable_slots: Tuple[MaterialAnalysis, ...],
    ) -> Tuple[MaterialSlotPreparation, ...]:
        ...


def _slot_requires_procedural_mode(
    slot: MaterialAnalysis,
    policy: BakeMaterialPolicy,
) -> bool:
    if slot.kind in {MaterialKind.PROCEDURAL, MaterialKind.SOLID_COLOR}:
        return True
    if slot.kind is MaterialKind.MIXED:
        if policy is BakeMaterialPolicy.LEGACY_ANY_IMAGE:
            return not slot.has_image_dependency
        return True
    return False


def _slot_evaluation_scope(slot: MaterialAnalysis) -> BakeEvaluationScope:
    dependencies = set(slot.dependencies)
    if dependencies & _CAMERA_DEPENDENCIES:
        return BakeEvaluationScope.CAMERA
    if dependencies & _SCENE_DEPENDENCIES:
        return BakeEvaluationScope.SCENE
    return BakeEvaluationScope.LOCAL


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


def _mask_unmatched_preparations(
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
class CameraCombinedBakeStrategy:
    """Evaluate view/ray/reflection/transmission appearance from the active camera."""

    strategy_id: BakeStrategyId = BakeStrategyId.CAMERA_COMBINED
    priority: int = 25
    evaluation_scope: BakeEvaluationScope = BakeEvaluationScope.CAMERA

    def supports(self, slot: MaterialAnalysis) -> bool:
        return bool(set(slot.semantic_channels) & _APPEARANCE_CHANNELS)

    def semantic_channels(
        self,
        slots: Tuple[MaterialAnalysis, ...],
    ) -> Tuple[MaterialSemanticChannel, ...]:
        return _appearance_channels(slots)

    def select_mode(
        self,
        slots: Tuple[MaterialAnalysis, ...],
        settings: BakeSettings,
    ) -> BakeMode:
        del slots, settings
        return BakeMode.ACTIVE_CAMERA

    def material_preparations(
        self,
        matched_slots: Tuple[MaterialAnalysis, ...],
        usable_slots: Tuple[MaterialAnalysis, ...],
    ) -> Tuple[MaterialSlotPreparation, ...]:
        return _mask_unmatched_preparations(matched_slots, usable_slots)


@dataclass(frozen=True, slots=True)
class SceneCombinedBakeStrategy:
    """Evaluate lighting, World, occlusion and other scene-object appearance."""

    strategy_id: BakeStrategyId = BakeStrategyId.SCENE_COMBINED
    priority: int = 50
    evaluation_scope: BakeEvaluationScope = BakeEvaluationScope.SCENE

    def supports(self, slot: MaterialAnalysis) -> bool:
        return bool(set(slot.semantic_channels) & _APPEARANCE_CHANNELS)

    def semantic_channels(
        self,
        slots: Tuple[MaterialAnalysis, ...],
    ) -> Tuple[MaterialSemanticChannel, ...]:
        return _appearance_channels(slots)

    def select_mode(
        self,
        slots: Tuple[MaterialAnalysis, ...],
        settings: BakeSettings,
    ) -> BakeMode:
        del slots, settings
        return BakeMode.COMBINED

    def material_preparations(
        self,
        matched_slots: Tuple[MaterialAnalysis, ...],
        usable_slots: Tuple[MaterialAnalysis, ...],
    ) -> Tuple[MaterialSlotPreparation, ...]:
        return _mask_unmatched_preparations(matched_slots, usable_slots)


@dataclass(frozen=True, slots=True)
class SurfaceColorBakeStrategy:
    strategy_id: BakeStrategyId = BakeStrategyId.SURFACE_COLOR
    priority: int = 100
    evaluation_scope: BakeEvaluationScope = BakeEvaluationScope.LOCAL

    def supports(self, slot: MaterialAnalysis) -> bool:
        return MaterialSemanticChannel.SURFACE_COLOR in slot.semantic_channels

    def semantic_channels(
        self,
        slots: Tuple[MaterialAnalysis, ...],
    ) -> Tuple[MaterialSemanticChannel, ...]:
        del slots
        return (MaterialSemanticChannel.SURFACE_COLOR,)

    def select_mode(
        self,
        slots: Tuple[MaterialAnalysis, ...],
        settings: BakeSettings,
    ) -> BakeMode:
        if any(MaterialSemanticChannel.ALPHA in slot.semantic_channels for slot in slots):
            return BakeMode.EMIT
        if any(
            _slot_requires_procedural_mode(slot, settings.material_policy)
            for slot in slots
        ):
            return settings.procedural_mode
        return settings.diffuse_mode

    def material_preparations(
        self,
        matched_slots: Tuple[MaterialAnalysis, ...],
        usable_slots: Tuple[MaterialAnalysis, ...],
    ) -> Tuple[MaterialSlotPreparation, ...]:
        return _mask_unmatched_preparations(matched_slots, usable_slots)


@dataclass(frozen=True, slots=True)
class EmissionBakeStrategy:
    strategy_id: BakeStrategyId = BakeStrategyId.EMISSION
    priority: int = 200
    evaluation_scope: BakeEvaluationScope = BakeEvaluationScope.LOCAL

    def supports(self, slot: MaterialAnalysis) -> bool:
        return MaterialSemanticChannel.SURFACE_EMISSION in slot.semantic_channels

    def semantic_channels(
        self,
        slots: Tuple[MaterialAnalysis, ...],
    ) -> Tuple[MaterialSemanticChannel, ...]:
        del slots
        return (MaterialSemanticChannel.SURFACE_EMISSION,)

    def select_mode(
        self,
        slots: Tuple[MaterialAnalysis, ...],
        settings: BakeSettings,
    ) -> BakeMode:
        del slots, settings
        return BakeMode.EMIT

    def material_preparations(
        self,
        matched_slots: Tuple[MaterialAnalysis, ...],
        usable_slots: Tuple[MaterialAnalysis, ...],
    ) -> Tuple[MaterialSlotPreparation, ...]:
        return _mask_unmatched_preparations(matched_slots, usable_slots)


@dataclass(frozen=True, slots=True)
class AlphaBakeStrategy:
    """Expose computed material opacity as a grayscale EMIT pass."""

    strategy_id: BakeStrategyId = BakeStrategyId.ALPHA
    priority: int = 300
    evaluation_scope: BakeEvaluationScope = BakeEvaluationScope.AUXILIARY

    def supports(self, slot: MaterialAnalysis) -> bool:
        return MaterialSemanticChannel.ALPHA in slot.semantic_channels

    def semantic_channels(
        self,
        slots: Tuple[MaterialAnalysis, ...],
    ) -> Tuple[MaterialSemanticChannel, ...]:
        del slots
        return (MaterialSemanticChannel.ALPHA,)

    def select_mode(
        self,
        slots: Tuple[MaterialAnalysis, ...],
        settings: BakeSettings,
    ) -> BakeMode:
        del slots, settings
        return BakeMode.EMIT

    def material_preparations(
        self,
        matched_slots: Tuple[MaterialAnalysis, ...],
        usable_slots: Tuple[MaterialAnalysis, ...],
    ) -> Tuple[MaterialSlotPreparation, ...]:
        alpha_slots = {slot.slot_index for slot in matched_slots}
        return tuple(
            MaterialSlotPreparation(
                slot_index=slot.slot_index,
                mode=(
                    MaterialPreparationMode.EXTRACT_ALPHA_TO_EMISSION
                    if slot.slot_index in alpha_slots
                    else MaterialPreparationMode.OPAQUE_ALPHA_TO_EMISSION
                ),
            )
            for slot in usable_slots
        )


@dataclass(frozen=True, slots=True)
class BakeStrategyRegistry:
    strategies: Tuple[BakeStrategy, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.strategies, tuple) or not self.strategies:
            raise ValueError("strategies must be a non-empty tuple")
        ids = tuple(strategy.strategy_id for strategy in self.strategies)
        if len(ids) != len(set(ids)):
            raise ValueError("strategy IDs must be unique")
        priorities = tuple(strategy.priority for strategy in self.strategies)
        if len(priorities) != len(set(priorities)):
            raise ValueError("strategy priorities must be unique")

    def resolve(
        self,
        analysis: ObjectMaterialAnalysis,
        settings: BakeSettings,
        *,
        object_context: ObjectBakeContext | None = None,
        scene_context: SceneBakeContext | None = None,
    ) -> tuple[Tuple[BakePassPlan, ...], BakeCompositePlan]:
        if not isinstance(analysis, ObjectMaterialAnalysis):
            raise TypeError("analysis must be ObjectMaterialAnalysis")
        if not isinstance(settings, BakeSettings):
            raise TypeError("settings must be BakeSettings")
        if object_context is not None:
            if not isinstance(object_context, ObjectBakeContext):
                raise TypeError("object_context must be ObjectBakeContext or None")
            if object_context.source_object_id != analysis.source_object_id:
                raise BakePlanError("object context does not match material analysis")
        if scene_context is not None and not isinstance(scene_context, SceneBakeContext):
            raise TypeError("scene_context must be SceneBakeContext or None")
        if not analysis.slots:
            raise BakePlanError("object has no material slots")

        unsupported = tuple(
            slot for slot in analysis.slots if slot.kind is MaterialKind.UNSUPPORTED
        )
        if unsupported:
            names = tuple(
                slot.material_name or f"slot-{slot.slot_index}" for slot in unsupported
            )
            raise BakePlanError(f"unsupported materials cannot be baked safely: {names}")

        usable = tuple(
            slot for slot in analysis.slots if slot.kind is not MaterialKind.EMPTY
        )
        if not usable:
            raise BakePlanError("object has no usable materials")

        volume_slots = tuple(
            slot
            for slot in usable
            if MaterialSemanticChannel.VOLUME in slot.semantic_channels
        )
        if volume_slots:
            names = tuple(
                slot.material_name or f"slot-{slot.slot_index}" for slot in volume_slots
            )
            raise BakePlanError(
                "volume output requires a camera-projection strategy that is not "
                f"registered yet: {names}"
            )

        scope_by_slot = {
            slot.slot_index: _slot_evaluation_scope(slot) for slot in usable
        }
        scene_slots = tuple(
            slot
            for slot in usable
            if scope_by_slot[slot.slot_index] is BakeEvaluationScope.SCENE
        )
        camera_slots = tuple(
            slot
            for slot in usable
            if scope_by_slot[slot.slot_index] is BakeEvaluationScope.CAMERA
        )
        if (scene_slots or camera_slots) and scene_context is None:
            names = tuple(
                slot.material_name or f"slot-{slot.slot_index}"
                for slot in scene_slots + camera_slots
            )
            raise BakePlanError(
                "scene-aware materials require an immutable SceneBakeContext: " + str(names)
            )
        if camera_slots and (scene_context is None or scene_context.camera is None):
            names = tuple(
                slot.material_name or f"slot-{slot.slot_index}" for slot in camera_slots
            )
            raise BakePlanError(
                "camera-dependent materials require an active scene camera: " + str(names)
            )

        resolved: list[BakePassPlan] = []
        primary_covered_slots: set[int] = set()
        ordered_strategies = tuple(
            sorted(self.strategies, key=lambda strategy: strategy.priority)
        )
        for strategy in ordered_strategies:
            if strategy.evaluation_scope is BakeEvaluationScope.AUXILIARY:
                candidates = usable
            else:
                candidates = tuple(
                    slot
                    for slot in usable
                    if scope_by_slot[slot.slot_index] is strategy.evaluation_scope
                )
            matched = tuple(slot for slot in candidates if strategy.supports(slot))
            if not matched:
                continue
            slot_indices = tuple(slot.slot_index for slot in matched)
            if strategy.evaluation_scope is not BakeEvaluationScope.AUXILIARY:
                primary_covered_slots.update(slot_indices)
            elif strategy.strategy_id is BakeStrategyId.ALPHA:
                # Pure Transparent/Holdout has no RGB appearance pass. In that one case
                # Alpha is both the only evaluable output and the final coverage pass.
                primary_covered_slots.update(
                    slot.slot_index
                    for slot in matched
                    if not (set(slot.semantic_channels) & _APPEARANCE_CHANNELS)
                )
            resolved.append(
                BakePassPlan(
                    pass_index=len(resolved),
                    strategy_id=strategy.strategy_id,
                    bake_mode=strategy.select_mode(matched, settings),
                    material_slot_indices=slot_indices,
                    semantic_channels=strategy.semantic_channels(matched),
                    evaluation_scope=strategy.evaluation_scope,
                    material_preparations=strategy.material_preparations(
                        matched,
                        usable,
                    ),
                )
            )

        missing = tuple(
            slot.material_name or f"slot-{slot.slot_index}"
            for slot in usable
            if slot.slot_index not in primary_covered_slots
        )
        if missing:
            raise BakePlanError(
                "no registered primary bake strategy can evaluate material slots: "
                f"{missing}"
            )
        if not resolved:
            raise BakePlanError("strategy registry produced no bake passes")

        has_emission_pass = any(
            item.strategy_id is BakeStrategyId.EMISSION for item in resolved
        )
        if has_emission_pass:
            resolved = [
                replace(item, bake_mode=BakeMode.DIFFUSE)
                if item.strategy_id is BakeStrategyId.SURFACE_COLOR
                and item.bake_mode is BakeMode.COMBINED
                else item
                for item in resolved
            ]

        alpha_indices = tuple(
            item.pass_index
            for item in resolved
            if item.strategy_id is BakeStrategyId.ALPHA
        )
        if len(alpha_indices) > 1:
            raise BakePlanError("only one alpha strategy pass may be registered")
        if alpha_indices:
            alpha_index = alpha_indices[0]
            color_indices = tuple(
                item.pass_index
                for item in resolved
                if item.pass_index != alpha_index
            )
            scene_color = any(
                resolved[index].evaluation_scope
                in {BakeEvaluationScope.SCENE, BakeEvaluationScope.CAMERA}
                for index in color_indices
            )
            composite = BakeCompositePlan(
                mode=BakeCompositeMode.ADD_RGB_REPLACE_ALPHA,
                clamp_rgb=True,
                color_pass_indices=color_indices,
                alpha_pass_index=alpha_index,
                unpremultiply_color_by_alpha=scene_color,
            )
        elif len(resolved) == 1:
            composite = BakeCompositePlan(mode=BakeCompositeMode.SINGLE, clamp_rgb=True)
        else:
            composite = BakeCompositePlan(
                mode=BakeCompositeMode.ADD_RGB_MAX_ALPHA,
                clamp_rgb=True,
                color_pass_indices=tuple(item.pass_index for item in resolved),
            )
        return tuple(resolved), composite


def build_default_bake_strategy_registry() -> BakeStrategyRegistry:
    return BakeStrategyRegistry(
        strategies=(
            CameraCombinedBakeStrategy(),
            SceneCombinedBakeStrategy(),
            SurfaceColorBakeStrategy(),
            EmissionBakeStrategy(),
            AlphaBakeStrategy(),
        )
    )


def resolve_bake_strategy_plan(
    analysis: ObjectMaterialAnalysis,
    settings: BakeSettings,
    *,
    registry: BakeStrategyRegistry | None = None,
    object_context: ObjectBakeContext | None = None,
    scene_context: SceneBakeContext | None = None,
) -> tuple[Tuple[BakePassPlan, ...], BakeCompositePlan]:
    resolved_registry = registry or build_default_bake_strategy_registry()
    if not isinstance(resolved_registry, BakeStrategyRegistry):
        raise TypeError("registry must be BakeStrategyRegistry or None")
    return resolved_registry.resolve(
        analysis,
        settings,
        object_context=object_context,
        scene_context=scene_context,
    )
