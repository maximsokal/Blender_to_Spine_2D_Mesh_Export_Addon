"""Deterministic strategy selection for semantic material bake requirements."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol, Tuple

from .graph import MaterialSemanticChannel
from .model import (
    BakeCompositeMode,
    BakeCompositePlan,
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


class BakeStrategy(Protocol):
    """One independently extensible strategy registered with the resolver."""

    strategy_id: BakeStrategyId
    priority: int

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


@dataclass(frozen=True, slots=True)
class SurfaceColorBakeStrategy:
    strategy_id: BakeStrategyId = BakeStrategyId.SURFACE_COLOR
    priority: int = 100

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
        del matched_slots, usable_slots
        return ()


@dataclass(frozen=True, slots=True)
class EmissionBakeStrategy:
    strategy_id: BakeStrategyId = BakeStrategyId.EMISSION
    priority: int = 200

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
        del matched_slots, usable_slots
        return ()


@dataclass(frozen=True, slots=True)
class AlphaBakeStrategy:
    """Expose computed material opacity as a grayscale EMIT pass."""

    strategy_id: BakeStrategyId = BakeStrategyId.ALPHA
    priority: int = 300

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
    ) -> tuple[Tuple[BakePassPlan, ...], BakeCompositePlan]:
        if not isinstance(analysis, ObjectMaterialAnalysis):
            raise TypeError("analysis must be ObjectMaterialAnalysis")
        if not isinstance(settings, BakeSettings):
            raise TypeError("settings must be BakeSettings")
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

        resolved: list[BakePassPlan] = []
        covered_slots: set[int] = set()
        ordered_strategies = tuple(
            sorted(self.strategies, key=lambda strategy: strategy.priority)
        )
        for strategy in ordered_strategies:
            matched = tuple(slot for slot in usable if strategy.supports(slot))
            if not matched:
                continue
            slot_indices = tuple(slot.slot_index for slot in matched)
            covered_slots.update(slot_indices)
            resolved.append(
                BakePassPlan(
                    pass_index=len(resolved),
                    strategy_id=strategy.strategy_id,
                    bake_mode=strategy.select_mode(matched, settings),
                    material_slot_indices=slot_indices,
                    semantic_channels=strategy.semantic_channels(matched),
                    material_preparations=strategy.material_preparations(
                        matched,
                        usable,
                    ),
                )
            )

        missing = tuple(
            slot.material_name or f"slot-{slot.slot_index}"
            for slot in usable
            if slot.slot_index not in covered_slots
        )
        if missing:
            raise BakePlanError(
                "no registered bake strategy can evaluate material slots: "
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
            composite = BakeCompositePlan(
                mode=BakeCompositeMode.ADD_RGB_REPLACE_ALPHA,
                clamp_rgb=True,
                color_pass_indices=color_indices,
                alpha_pass_index=alpha_index,
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
) -> tuple[Tuple[BakePassPlan, ...], BakeCompositePlan]:
    resolved_registry = registry or build_default_bake_strategy_registry()
    if not isinstance(resolved_registry, BakeStrategyRegistry):
        raise TypeError("registry must be BakeStrategyRegistry or None")
    return resolved_registry.resolve(analysis, settings)
