"""Shared texture staging and post-render finalization for prepared A1 objects."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from types import MappingProxyType
from typing import Any, Mapping, Tuple

from ..application import (
    A1ExportProgressCallback,
    A1MultiObjectStage,
    emit_a1_export_progress,
    scale_a1_export_progress_callback,
)
from ..infrastructure import AtomicFileTransaction, AtomicOutputReservation
from .a1_multi_object_contracts import (
    PreparedA1MultiObject,
    record_object_statistics,
)
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
    progress_callback: A1ExportProgressCallback | None = None,
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
    object_count = len(prepared.objects)
    emit_a1_export_progress(
        progress_callback,
        percent=0,
        stage=A1MultiObjectStage.STAGE_OUTPUTS,
        message="Starting texture staging",
    )
    for index, (source, item) in enumerate(
        zip(prepared.sources, prepared.objects, strict=True)
    ):
        start_percent = int(round(index * 100.0 / object_count))
        end_percent = int(round((index + 1) * 100.0 / object_count))
        emit_a1_export_progress(
            progress_callback,
            percent=start_percent,
            stage=A1MultiObjectStage.STAGE_OUTPUTS,
            message=f"Staging textures for {item.object_id}",
            object_id=source.component_id,
            object_index=index + 1,
            object_count=object_count,
        )
        object_progress = scale_a1_export_progress_callback(
            progress_callback,
            start_percent=start_percent,
            end_percent=end_percent,
            object_id=source.component_id,
            object_index=index + 1,
            object_count=object_count,
        )
        staged = stage_texture_plan_outputs(
            item.source_object,
            item.bake_target_snapshot,
            item.bake_plan,
            transaction,
            item.settings.bake_execution,
            context=context,
            scene=scene,
            progress_callback=object_progress,
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
        emit_a1_export_progress(
            progress_callback,
            percent=end_percent,
            stage=A1MultiObjectStage.STAGE_OUTPUTS,
            message=f"Textures staged for {item.object_id}",
            object_id=source.component_id,
            object_index=index + 1,
            object_count=object_count,
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
