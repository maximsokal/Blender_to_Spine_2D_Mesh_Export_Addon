"""Post-render atomic output service for prepared A1 multi-object exports."""

from __future__ import annotations

import logging
from typing import Any, Tuple

from ..application import (
    A1ExportProgressCallback,
    A1MultiObjectExportSettings,
    A1MultiObjectMode,
    A1MultiObjectStage,
    ExportResult,
    emit_a1_export_progress,
    scale_a1_export_progress_callback,
    validate_a1_realized_output_namespace,
)
from ..domain.spine import ConnectedGroupBuildResult, SpineSerializer
from ..infrastructure import (
    AtomicFileCommitError,
    atomic_file_transaction,
    write_staged_utf8_text,
)
from .a1_composition_result import replace_a1_composition_document
from .a1_grouped_output import apply_staged_grouped_camera_overlay
from .a1_multi_object_composition import compose_a1_multi_object_document
from .a1_multi_object_export import (
    A1MultiObjectPreparationError,
    A1MultiObjectSource,
    prepare_a1_multi_object,
)
from .a1_multi_object_result import build_multi_object_failure_result
from .a1_output_staging import stage_and_finalize_a1_objects
from .a1_output_statistics import (
    record_final_document_statistics,
    record_grouped_camera_statistics,
)
from .grouped_camera_projection_output import (
    stage_grouped_camera_projection_outputs,
)
from .grouped_camera_projection_policy import (
    resolve_grouped_camera_projection_request,
)


logger = logging.getLogger(__name__)
_OPERATION = "A1 multi-object output"
_TRANSACTION_NAME = "a1-multi-object"


def export_a1_multi_object(
    sources: Tuple[A1MultiObjectSource, ...],
    settings: A1MultiObjectExportSettings,
    *,
    context: Any | None = None,
    scene: Any | None = None,
    progress_callback: A1ExportProgressCallback | None = None,
) -> ExportResult:
    """Finalize textures, compose once, serialize once, and commit atomically."""

    emit_a1_export_progress(
        progress_callback,
        percent=0,
        stage=A1MultiObjectStage.VALIDATE_REQUEST,
        message="Starting multi-object export",
    )
    preparation_progress = scale_a1_export_progress_callback(
        progress_callback,
        start_percent=5.0,
        end_percent=55.0,
    )
    try:
        prepared = prepare_a1_multi_object(
            sources,
            settings,
            context=context,
            scene=scene,
            progress_callback=preparation_progress,
        )
    except A1MultiObjectPreparationError as exc:
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
    except Exception as exc:
        return build_multi_object_failure_result(
            logger=logger,
            operation=_OPERATION,
            stage=A1MultiObjectStage.VALIDATE_REQUEST,
            exc=exc,
            statistics={},
            warnings=(),
        )

    stage = A1MultiObjectStage.VALIDATE_OUTPUTS
    statistics = dict(prepared.statistics)
    try:
        emit_a1_export_progress(
            progress_callback,
            percent=58,
            stage=stage,
            message="Validating final output namespace",
        )
        grouped_request = (
            resolve_grouped_camera_projection_request(
                prepared.objects,
                settings,
            )
            if settings.mode is A1MultiObjectMode.CONNECTED
            else None
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
        with atomic_file_transaction(
            operation_name=_TRANSACTION_NAME,
        ) as output_transaction:
            json_reservation = output_transaction.reserve(prepared.json_path)
            staging_progress = scale_a1_export_progress_callback(
                progress_callback,
                start_percent=60.0,
                end_percent=80.0,
            )
            staged_objects = stage_and_finalize_a1_objects(
                prepared,
                output_transaction,
                statistics,
                context=context,
                scene=scene,
                progress_callback=staging_progress,
            )
            statistics = dict(staged_objects.statistics)
            finalized_objects = staged_objects.objects

            grouped_stage = None
            if grouped_request is not None:
                emit_a1_export_progress(
                    progress_callback,
                    percent=82,
                    stage=stage,
                    message="Staging grouped camera projection",
                )
                grouped_stage = stage_grouped_camera_projection_outputs(
                    grouped_request.source_objects,
                    grouped_request.plan,
                    output_transaction,
                    grouped_request.execution_settings,
                    context=context,
                    scene=scene,
                )

            stage = A1MultiObjectStage.COMPOSE_DOCUMENT
            emit_a1_export_progress(
                progress_callback,
                percent=86,
                stage=stage,
                message="Composing final Spine document",
            )
            composition = compose_a1_multi_object_document(
                prepared.sources,
                finalized_objects,
                settings,
            )
            overlay = None
            if grouped_request is not None:
                if grouped_stage is None:
                    raise RuntimeError(
                        "grouped camera request completed without a stage result"
                    )
                overlay = apply_staged_grouped_camera_overlay(
                    composition.document,
                    grouped_request,
                    grouped_stage,
                )
                composition = replace_a1_composition_document(
                    composition,
                    overlay.document,
                )

            document = composition.document
            record_final_document_statistics(
                statistics,
                document,
                finalized_objects,
                grouped_enabled=grouped_request is not None,
            )
            if isinstance(composition, ConnectedGroupBuildResult):
                statistics["connected_layer_count"] = len(composition.layers)
            if (
                grouped_request is not None
                and grouped_stage is not None
                and overlay is not None
            ):
                record_grouped_camera_statistics(
                    statistics,
                    grouped_request,
                    grouped_stage,
                    overlay,
                )

            stage = A1MultiObjectStage.SERIALIZE_DOCUMENT
            emit_a1_export_progress(
                progress_callback,
                percent=93,
                stage=stage,
                message="Serializing Spine JSON",
            )
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
            emit_a1_export_progress(
                progress_callback,
                percent=98,
                stage=stage,
                message="Committing JSON and texture files",
            )
            committed_paths = output_transaction.commit()

        grouped_reservations = (
            () if grouped_stage is None else grouped_stage.reservations
        )
        expected_paths = (
            json_reservation.final_path,
            *(item.final_path for item in staged_objects.reservations),
            *(item.final_path for item in grouped_reservations),
        )
        if tuple(committed_paths) != expected_paths:
            raise AtomicFileCommitError(
                "Committed output order does not match final JSON and texture reservations"
            )
        statistics["output_file_count"] = len(committed_paths)
        logger.info(
            "A1 multi-object export completed (%s, grouped_camera=%s): %s",
            settings.mode.value,
            grouped_request is not None,
            tuple(str(path) for path in committed_paths),
        )
        emit_a1_export_progress(
            progress_callback,
            percent=100,
            stage=A1MultiObjectStage.COMMIT_OUTPUTS,
            message="Export complete",
        )
        return ExportResult(
            success=True,
            output_files=tuple(committed_paths),
            issues=prepared.warnings,
            statistics=statistics,
        )
    except Exception as exc:
        return build_multi_object_failure_result(
            logger=logger,
            operation=_OPERATION,
            stage=stage,
            exc=exc,
            statistics=statistics,
            warnings=prepared.warnings,
        )


__all__ = ["export_a1_multi_object"]
