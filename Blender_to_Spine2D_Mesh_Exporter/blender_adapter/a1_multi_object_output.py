"""Post-render atomic output service for prepared A1 multi-object exports."""

from __future__ import annotations

from dataclasses import replace
import logging
from typing import Any, Mapping, Tuple

from ..application import (
    A1MultiObjectExportSettings,
    A1MultiObjectMode,
    A1MultiObjectStage,
    ExportIssue,
    ExportResult,
    IssueSeverity,
    apply_grouped_camera_overlay,
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
from .grouped_camera_projection_executor import (
    stage_grouped_camera_projection_outputs,
)
from .grouped_camera_projection_policy import (
    resolve_grouped_camera_projection_request,
)
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
    """Stage textures, optional grouped B4, compose JSON, and commit atomically."""

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

            resolved_finalized = tuple(finalized_objects)
            grouped_request = (
                resolve_grouped_camera_projection_request(
                    resolved_finalized,
                    settings,
                )
                if settings.mode is A1MultiObjectMode.CONNECTED
                else None
            )
            grouped_stage = None
            if grouped_request is not None:
                grouped_stage = stage_grouped_camera_projection_outputs(
                    grouped_request.source_objects,
                    grouped_request.plan,
                    output_transaction,
                    grouped_request.execution_settings,
                    context=context,
                    scene=scene,
                )

            stage = A1MultiObjectStage.COMPOSE_DOCUMENT
            composition = _compose_document(
                prepared.sources,
                resolved_finalized,
                settings,
            )
            overlay = None
            if grouped_request is not None:
                if grouped_stage is None:
                    raise RuntimeError(
                        "grouped B4 request completed without a stage result"
                    )
                overlay = apply_grouped_camera_overlay(
                    composition.document,
                    grouped_request.plan,
                    grouped_stage.layout,
                    visual_slot_names=grouped_request.visual_slot_names,
                    image_relative_directory=(
                        grouped_request.image_relative_directory
                    ),
                    slot_name=grouped_request.slot_name,
                    attachment_name=grouped_request.attachment_name,
                )
                composition = replace(
                    composition,
                    document=overlay.document,
                )

            document = composition.document
            statistics.update(
                {
                    "final_bone_count": len(document.bones),
                    "final_slot_count": len(document.slots),
                    "final_skin_count": len(document.skins),
                    "projection_cropped_component_count": sum(
                        1
                        for item in resolved_finalized
                        if "projection_crop_width" in item.statistics
                    ),
                    "grouped_b4_enabled": int(grouped_request is not None),
                }
            )
            if grouped_request is not None and grouped_stage is not None:
                statistics.update(
                    {
                        "grouped_b4_source_count": len(
                            grouped_request.plan.source_object_ids
                        ),
                        "grouped_b4_frame_count": len(
                            grouped_request.plan.frame_tasks
                        ),
                        "grouped_b4_crop_width": grouped_stage.layout.cropped_width,
                        "grouped_b4_crop_height": grouped_stage.layout.cropped_height,
                        "grouped_b4_contour_vertex_count": len(
                            grouped_stage.layout.hull
                        ),
                        "grouped_b4_hidden_slot_count": len(
                            overlay.hidden_slot_names if overlay is not None else ()
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

        grouped_reservations = (
            () if grouped_stage is None else grouped_stage.reservations
        )
        expected_paths = (
            json_reservation.final_path,
            *(reservation.final_path for reservation in texture_reservations),
            *(reservation.final_path for reservation in grouped_reservations),
        )
        if tuple(committed_paths) != expected_paths:
            raise AtomicFileCommitError(
                "Committed output order does not match final JSON and texture reservations"
            )
        statistics["output_file_count"] = len(committed_paths)
        logger.info(
            "A1 multi-object export completed (%s, grouped_b4=%s): %s",
            settings.mode.value,
            grouped_request is not None,
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
