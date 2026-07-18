"""Shared texture staging and post-render finalization for prepared A1 objects."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from types import MappingProxyType
from typing import Any, Mapping, Tuple

from ..infrastructure import AtomicFileTransaction, AtomicOutputReservation
from .a1_multi_object_export import PreparedA1MultiObject, record_object_statistics
from .a1_object_preparation import PreparedA1Object, StatisticsValue
from .a1_projection_finalization import finalize_prepared_camera_projection
from .texture_executor import stage_texture_plan_outputs


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class A1StagedFinalizedObjects:
    """Finalized prepared objects and all per-object texture reservations."""

    objects: Tuple[PreparedA1Object, ...]
    reservations: Tuple[AtomicOutputReservation, ...]
    statistics: Mapping[str, StatisticsValue]

    def __post_init__(self) -> None:
        if not isinstance(self.objects, tuple) or not self.objects:
            raise ValueError("objects must be a non-empty tuple")
        if not all(isinstance(item, PreparedA1Object) for item in self.objects):
            raise TypeError("objects must contain PreparedA1Object values")
        if not isinstance(self.reservations, tuple) or not self.reservations:
            raise ValueError("reservations must be a non-empty tuple")
        if not all(isinstance(item, AtomicOutputReservation) for item in self.reservations):
            raise TypeError("reservations must contain AtomicOutputReservation values")
        if not isinstance(self.statistics, Mapping):
            raise TypeError("statistics must be a mapping")


def stage_and_finalize_a1_objects(
    prepared: PreparedA1MultiObject,
    transaction: AtomicFileTransaction,
    statistics: Mapping[str, StatisticsValue],
    *,
    context: Any | None = None,
    scene: Any | None = None,
) -> A1StagedFinalizedObjects:
    """Stage every object texture plan and apply its render-derived final layout."""

    if not isinstance(prepared, PreparedA1MultiObject):
        raise TypeError("prepared must be PreparedA1MultiObject")
    if not isinstance(transaction, AtomicFileTransaction):
        raise TypeError("transaction must be AtomicFileTransaction")
    if not isinstance(statistics, Mapping):
        raise TypeError("statistics must be a mapping")

    resolved_statistics: dict[str, StatisticsValue] = dict(statistics)
    reservations: list[AtomicOutputReservation] = []
    finalized_objects: list[PreparedA1Object] = []
    for source, item in zip(prepared.sources, prepared.objects, strict=True):
        staged = stage_texture_plan_outputs(
            item.source_object,
            item.bake_target_snapshot,
            item.bake_plan,
            transaction,
            item.settings.bake_execution,
            context=context,
            scene=scene,
        )
        reservations.extend(staged.reservations)
        finalized = finalize_prepared_camera_projection(
            item,
            staged.projection_layout,
        )
        finalized_objects.append(finalized)
        record_object_statistics(
            resolved_statistics,
            source.component_id,
            finalized.statistics,
        )

    result = A1StagedFinalizedObjects(
        objects=tuple(finalized_objects),
        reservations=tuple(reservations),
        statistics=MappingProxyType(resolved_statistics),
    )
    logger.debug(
        "Staged and finalized %d A1 objects into %d texture reservations",
        len(result.objects),
        len(result.reservations),
    )
    return result


__all__ = ["A1StagedFinalizedObjects", "stage_and_finalize_a1_objects"]
