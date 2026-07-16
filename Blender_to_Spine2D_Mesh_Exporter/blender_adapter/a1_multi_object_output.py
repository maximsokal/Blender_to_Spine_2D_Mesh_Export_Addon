"""Post-render atomic output service for prepared A1 multi-object exports."""

from __future__ import annotations

import logging
from typing import Any, Mapping, Tuple

from ..application import (
    A1MultiObjectExportSettings,
    A1MultiObjectStage,
    ExportIssue,
    ExportResult,
    IssueSeverity,
)
from ..domain.spine import SpineSerializer
from ..infrastructure import (
    AtomicFileCommitError,
    atomic_file_transaction,
    write_staged_utf8_text,
)
from .a1_multi_object_export import (
    A1MultiObjectPreparationError,
    A1MultiObjectSource,
    _compose_document,
    _record_object_statistics,
    prepare_a1_multi_object,
)
from .a1_object_preparation import StatisticsValue
from .a1_projection_finalization import finalize_prepared_camera_projection
from .texture_executor import stage_texture_plan_outputs

logger = logging.getLogger(__name__)


def _failure_result(
    *,
    stage: A1MultiObjectStage,
    exc: Exception,
    statistics: Mapping[str, StatisticsValue],
    warnings: Tuple[ExportIssue, ...],
    component_id: str | None = None,
    object_id: str | None = None,
    object_stage: str | None = None,
) -> ExportResult:
    context: dict[str, object] = {"exception_type": type(exc).__name__}
    if component_id is not None:
        context["component_id"] = component_id
    if object_stage is not None:
        context["object_stage"] = object_stage
    logger.exception(
        "A1 multi-object output failed at %s (component=%s, object=%s)",
        stage.value,
        component_id,
        object_id,
    )
    error = ExportIssue(
        severity=IssueSeverity.ERROR,
        stage=stage.value,
        code=stage.error_code,
        message=str(exc) or type(exc).__name__,
        object_id=object_id,
        technical_details=f"{type(exc).__name__}: {exc}",
        context=context,
    )
    return ExportResult(
        success=False,
        issues=warnings + (error,),
        statistics=dict(statistics),
    )


def export_a1_multi_object(
    sources: Tuple[A1MultiObjectSource, ...],
    settings: A1MultiObjectExportSettings,
    *,
    context: Any | None = None,
    scene: Any | None = None,
) -> ExportResult:
    """Stage all textures, finalize B4 layouts, compose JSON, and commit together."""

    try:
        prepared = prepare_a1_multi_object(
            sources,
            settings,
            context=context,
            scene=scene,
        )
    except A1MultiObjectPreparationError as exc:
        return _failure_result(
            stage=exc.stage,
            exc=exc.cause,
            statistics=exc.statistics,
            warnings=exc.warnings,
            component_id=exc.component_id,
            object_id=exc.object_id,
            object_stage=exc.object_stage,
        )
    except Exception as exc:
        return _failure_result(
            stage=A1MultiObjectStage.VALIDATE_REQUEST,
            exc=exc,
            statistics={},
            warnings=(),
        )

    stage = A1MultiObjectStage.STAGE_OUTPUTS
    statistics = dict(prepared.statistics)
    try:
        with atomic_file_transaction() as output_transaction:
            json_reservation = output_transaction.reserve(prepared.json_path)
            texture_reservations = []
            finalized_objects = []
            for source, item in zip(prepared.sources, prepared.objects):
                texture_stage = stage_texture_plan_outputs(
                    item.source_object,
                    item.bake_target_snapshot,
                    item.bake_plan,
                    output_transaction,
                    item.settings.bake_execution,
                    context=context,
                    scene=scene,
                )
                texture_reservations.extend(texture_stage.reservations)
                finalized = finalize_prepared_camera_projection(
                    item,
                    texture_stage.projection_layout,
                )
                finalized_objects.append(finalized)
                _record_object_statistics(
                    statistics,
                    source.component_id,
                    finalized.statistics,
                )

            stage = A1MultiObjectStage.COMPOSE_DOCUMENT
            composition = _compose_document(
                prepared.sources,
                tuple(finalized_objects),
                settings,
            )
            document = composition.document
            statistics.update(
                {
                    "final_bone_count": len(document.bones),
                    "final_slot_count": len(document.slots),
                    "final_skin_count": len(document.skins),
                    "projection_cropped_component_count": sum(
                        1
                        for item in finalized_objects
                        if "projection_crop_width" in item.statistics
                    ),
                }
            )

            stage = A1MultiObjectStage.SERIALIZE_DOCUMENT
            json_text = SpineSerializer().to_json(
                document,
                indent=settings.json_indent,
            )
            write_staged_utf8_text(
                json_reservation.staged_path,
                json_text,
                ensure_trailing_newline=True,
            )

            stage = A1MultiObjectStage.COMMIT_OUTPUTS
            committed_paths = output_transaction.commit()

        expected_paths = (
            json_reservation.final_path,
            *(reservation.final_path for reservation in texture_reservations),
        )
        if tuple(committed_paths) != expected_paths:
            raise AtomicFileCommitError(
                "Committed output order does not match final JSON and texture reservations"
            )
        statistics["output_file_count"] = len(committed_paths)
        logger.info(
            "A1 multi-object export completed (%s): %s",
            settings.mode.value,
            tuple(str(path) for path in committed_paths),
        )
        return ExportResult(
            success=True,
            output_files=tuple(committed_paths),
            issues=prepared.warnings,
            statistics=statistics,
        )
    except Exception as exc:
        return _failure_result(
            stage=stage,
            exc=exc,
            statistics=statistics,
            warnings=prepared.warnings,
        )


__all__ = ["export_a1_multi_object"]
