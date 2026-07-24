"""Post-render atomic output service for mixed connected/standalone A1 exports."""

from __future__ import annotations

import logging
from typing import Any, Mapping, Tuple

from ..application import (
    A1ExportProgressCallback,
    A1MultiObjectExportSettings,
    A1MultiObjectStage,
    ExportIssue,
    ExportResult,
    emit_a1_export_progress,
    scale_a1_export_progress_callback,
    validate_a1_realized_output_namespace,
)
from ..domain.spine import SpineSerializer
from ..infrastructure import (
    AtomicFileCommitError,
    atomic_file_transaction,
    write_staged_utf8_text,
)
from .a1_mixed_composition import (
    compose_a1_mixed_document,
    partition_mixed_prepared_objects,
)
from .a1_mixed_object_export import prepare_a1_mixed_object
from .a1_mixed_settings import build_connected_subgroup_settings
from .a1_multi_object_contracts import A1MultiObjectPreparationError, A1MultiObjectSource
from .a1_multi_object_result import build_multi_object_failure_result
from .a1_output_staging import stage_and_finalize_a1_objects
from .a1_output_statistics import (
    record_final_document_statistics,
    record_grouped_camera_statistics,
)
from .grouped_camera_projection_output import stage_grouped_camera_projection_outputs
from .grouped_camera_projection_policy import resolve_grouped_camera_projection_request


logger = logging.getLogger(__name__)
_OPERATION = "A1 mixed-object output"
_TRANSACTION_NAME = "a1-mixed-object"


def _progress(
    callback: A1ExportProgressCallback | None,
    percent: int,
    stage: A1MultiObjectStage,
    message: str,
) -> None:
    emit_a1_export_progress(callback, percent=percent, stage=stage, message=message)


def _scaled_progress(
    callback: A1ExportProgressCallback | None,
    start: float,
    end: float,
) -> A1ExportProgressCallback | None:
    return scale_a1_export_progress_callback(
        callback,
        start_percent=start,
        end_percent=end,
    )


def _failure(
    stage: A1MultiObjectStage,
    exc: Exception,
    statistics: Mapping[str, int | float | str],
    warnings: Tuple[ExportIssue, ...],
) -> ExportResult:
    return build_multi_object_failure_result(
        logger=logger,
        operation=_OPERATION,
        stage=stage,
        exc=exc,
        statistics=statistics,
        warnings=warnings,
    )


def _preparation_failure(exc: A1MultiObjectPreparationError) -> ExportResult:
    return build_multi_object_failure_result(
        logger=logger,
        operation=_OPERATION,
        stage=exc.stage,
        exc=exc.cause,
        statistics=exc.statistics,
        warnings=exc.warnings,
        component_id=exc.component_id,
        object_id=exc.object_id,
        object_stage=exc.object_stage,
    )


def export_a1_mixed_object(
    connected_sources: Tuple[A1MultiObjectSource, ...],
    standalone_sources: Tuple[A1MultiObjectSource, ...],
    settings: A1MultiObjectExportSettings,
    *,
    context: Any | None = None,
    scene: Any | None = None,
    progress_callback: A1ExportProgressCallback | None = None,
) -> ExportResult:
    """Finalize both groups, compose once, serialize once, and commit atomically."""

    _progress(progress_callback, 0, A1MultiObjectStage.VALIDATE_REQUEST, "Starting mixed-object export")
    try:
        prepared = prepare_a1_mixed_object(
            connected_sources,
            standalone_sources,
            settings,
            context=context,
            scene=scene,
            progress_callback=_scaled_progress(progress_callback, 5.0, 55.0),
        )
    except A1MultiObjectPreparationError as exc:
        return _preparation_failure(exc)
    except Exception as exc:
        return _failure(A1MultiObjectStage.VALIDATE_REQUEST, exc, {}, ())

    stage = A1MultiObjectStage.VALIDATE_OUTPUTS
    statistics = dict(prepared.statistics)
    try:
        _progress(progress_callback, 58, stage, "Validating final mixed output namespace")
        prepared_partition = partition_mixed_prepared_objects(
            prepared.objects,
            connected_sources,
            standalone_sources,
        )
        anchor = settings.anchor_component_id or connected_sources[0].component_id
        connected_settings = build_connected_subgroup_settings(settings, anchor)
        grouped_request = resolve_grouped_camera_projection_request(
            prepared_partition.connected,
            connected_settings,
        )
        grouped_paths = (
            ()
            if grouped_request is None
            else tuple(task.output_path for task in grouped_request.plan.frame_tasks)
        )
        validate_a1_realized_output_namespace(
            output_root=settings.output_directory,
            json_path=prepared.json_path,
            texture_paths=prepared.texture_output_paths,
            additional_texture_paths=grouped_paths,
        )

        stage = A1MultiObjectStage.STAGE_OUTPUTS
        with atomic_file_transaction(operation_name=_TRANSACTION_NAME) as transaction:
            json_reservation = transaction.reserve(prepared.json_path)
            staged_objects = stage_and_finalize_a1_objects(
                prepared,
                transaction,
                statistics,
                context=context,
                scene=scene,
                progress_callback=_scaled_progress(progress_callback, 60.0, 80.0),
            )
            statistics = dict(staged_objects.statistics)
            finalized_partition = partition_mixed_prepared_objects(
                staged_objects.objects,
                connected_sources,
                standalone_sources,
            )

            grouped_stage = None
            if grouped_request is not None:
                _progress(progress_callback, 82, stage, "Staging grouped camera projection")
                grouped_stage = stage_grouped_camera_projection_outputs(
                    grouped_request.source_objects,
                    grouped_request.plan,
                    transaction,
                    grouped_request.execution_settings,
                    context=context,
                    scene=scene,
                )

            stage = A1MultiObjectStage.COMPOSE_DOCUMENT
            _progress(progress_callback, 86, stage, "Composing mixed Spine document")
            composition = compose_a1_mixed_document(
                connected_sources,
                standalone_sources,
                finalized_partition,
                settings,
                grouped_request=grouped_request,
                grouped_stage=grouped_stage,
            )
            document = composition.document
            record_final_document_statistics(
                statistics,
                document,
                staged_objects.objects,
                grouped_enabled=grouped_request is not None,
            )
            if grouped_request is not None and grouped_stage is not None and composition.overlay is not None:
                record_grouped_camera_statistics(
                    statistics,
                    grouped_request,
                    grouped_stage,
                    composition.overlay,
                )

            stage = A1MultiObjectStage.SERIALIZE_DOCUMENT
            _progress(progress_callback, 93, stage, "Serializing Spine JSON")
            json_text = SpineSerializer().to_json(document, indent=settings.json_indent)
            write_staged_utf8_text(
                json_reservation.staged_path,
                json_text,
                ensure_trailing_newline=True,
            )

            stage = A1MultiObjectStage.COMMIT_OUTPUTS
            _progress(progress_callback, 98, stage, "Committing JSON and texture files")
            committed_paths = transaction.commit()

        grouped_reservations = () if grouped_stage is None else grouped_stage.reservations
        expected_paths = (
            json_reservation.final_path,
            *(item.final_path for item in staged_objects.reservations),
            *(item.final_path for item in grouped_reservations),
        )
        if tuple(committed_paths) != expected_paths:
            raise AtomicFileCommitError(
                "Committed output order does not match mixed JSON and texture reservations"
            )
        statistics["output_file_count"] = len(committed_paths)
        logger.info(
            "A1 mixed export completed (grouped_b4=%s): %s",
            grouped_request is not None,
            tuple(str(path) for path in committed_paths),
        )
        _progress(progress_callback, 100, A1MultiObjectStage.COMMIT_OUTPUTS, "Export complete")
        return ExportResult(
            success=True,
            output_files=tuple(committed_paths),
            issues=prepared.warnings,
            statistics=statistics,
        )
    except Exception as exc:
        return _failure(stage, exc, statistics, prepared.warnings)


__all__ = ["export_a1_mixed_object"]
