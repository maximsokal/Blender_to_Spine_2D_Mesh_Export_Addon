"""Post-render atomic output service for prepared A1 multi-object exports."""

from __future__ import annotations

import logging
from typing import Any, Mapping, Tuple

from ..application import (
    A1ExportProgressCallback,
    A1MultiObjectExportSettings,
    A1MultiObjectMode,
    A1MultiObjectStage,
    ExportIssue,
    ExportResult,
    emit_a1_export_progress,
    scale_a1_export_progress_callback,
    validate_a1_realized_output_namespace,
)
from ..domain.spine import ConnectedGroupBuildResult
from ..domain.spine.connected_group_serialization_validator import (
    ConnectedGroupSerializationValidator,
)
from ..domain.spine.validator import SpineValidator
from ..domain.spine.version_codecs import serialize_spine_document
from ..domain.spine.version_target import SpineJsonTarget
from ..infrastructure import (
    AtomicFileCommitError,
    atomic_file_transaction,
    write_staged_utf8_text,
)
from .a1_composition_result import replace_a1_composition_document
from .a1_grouped_output import apply_staged_grouped_camera_overlay
from .a1_multi_object_composition import compose_a1_multi_object_document
from .a1_multi_object_contracts import A1MultiObjectPreparationError, A1MultiObjectSource
from .a1_multi_object_export import prepare_a1_multi_object
from .a1_multi_object_result import build_multi_object_failure_result
from .a1_output_staging import stage_and_finalize_a1_objects
from .a1_output_statistics import (
    record_final_document_statistics,
    record_grouped_camera_statistics,
)
from .a1_spine_version_output import resolve_a1_sources_spine_target
from .grouped_camera_projection_output import stage_grouped_camera_projection_outputs
from .grouped_camera_projection_policy import resolve_grouped_camera_projection_request


logger = logging.getLogger(__name__)
_OPERATION = "A1 multi-object output"
_TRANSACTION_NAME = "a1-multi-object"


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


def _serialization_validator_for_composition(
    composition: object,
) -> SpineValidator | None:
    """Keep strict default validation outside validated connected composition."""

    if isinstance(composition, ConnectedGroupBuildResult):
        return ConnectedGroupSerializationValidator()
    return None


def _serialize_composition(
    composition: Any,
    spine_target: SpineJsonTarget,
    json_indent: int,
) -> str:
    """Serialize one completed composition through the selected version codec."""

    return serialize_spine_document(
        composition.document,
        spine_target,
        indent=json_indent,
        validator=_serialization_validator_for_composition(composition),
    )


def export_a1_multi_object(
    sources: Tuple[A1MultiObjectSource, ...],
    settings: A1MultiObjectExportSettings,
    *,
    context: Any | None = None,
    scene: Any | None = None,
    progress_callback: A1ExportProgressCallback | None = None,
) -> ExportResult:
    """Finalize textures, compose once, serialize once, and commit atomically."""

    _progress(
        progress_callback,
        0,
        A1MultiObjectStage.VALIDATE_REQUEST,
        "Starting multi-object export",
    )
    try:
        spine_target = resolve_a1_sources_spine_target(sources)
        prepared = prepare_a1_multi_object(
            sources,
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
        _progress(progress_callback, 58, stage, "Validating final output namespace")
        grouped_request = (
            resolve_grouped_camera_projection_request(prepared.objects, settings)
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
            finalized_objects = staged_objects.objects

            grouped_stage = None
            if grouped_request is not None:
                _progress(
                    progress_callback,
                    82,
                    stage,
                    "Staging grouped camera projection",
                )
                grouped_stage = stage_grouped_camera_projection_outputs(
                    grouped_request.source_objects,
                    grouped_request.plan,
                    transaction,
                    grouped_request.execution_settings,
                    context=context,
                    scene=scene,
                    progress_callback=_scaled_progress(
                        progress_callback,
                        82.0,
                        85.0,
                    ),
                )

            stage = A1MultiObjectStage.COMPOSE_DOCUMENT
            _progress(progress_callback, 86, stage, "Composing final Spine document")
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
            _progress(progress_callback, 93, stage, "Serializing Spine JSON")
            json_text = _serialize_composition(
                composition,
                spine_target,
                settings.json_indent,
            )
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
                "Committed output order does not match final JSON and texture reservations"
            )
        statistics["output_file_count"] = len(committed_paths)
        logger.info(
            "A1 multi-object export completed (%s, grouped_camera=%s): %s",
            settings.mode.value,
            grouped_request is not None,
            tuple(str(path) for path in committed_paths),
        )
        _progress(
            progress_callback,
            100,
            A1MultiObjectStage.COMMIT_OUTPUTS,
            "Export complete",
        )
        return ExportResult(
            success=True,
            output_files=tuple(committed_paths),
            issues=prepared.warnings,
            statistics=statistics,
        )
    except Exception as exc:
        return _failure(stage, exc, statistics, prepared.warnings)


__all__ = ["export_a1_multi_object"]
