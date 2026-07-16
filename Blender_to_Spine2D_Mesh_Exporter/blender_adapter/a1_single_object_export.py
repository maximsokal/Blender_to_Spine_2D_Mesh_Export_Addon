"""Atomic output wrapper for one prepared A1 Blender object.

All geometry, UV, material, rig, and initial document construction lives in
``prepare_a1_object``. Camera projection textures additionally produce a render-derived
crop/hull layout; this module finalizes that attachment before serializing JSON, while all
files still share one caller-owned atomic transaction.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Tuple

from ..application import (
    A1SingleObjectExportSettings,
    A1SingleObjectStage,
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
from .a1_object_preparation import (
    A1ObjectPreparationError,
    StatisticsValue,
    prepare_a1_object,
)
from .a1_projection_finalization import finalize_prepared_camera_projection
from .texture_executor import stage_texture_plan_outputs

logger = logging.getLogger(__name__)


def _failure_result(
    *,
    stage: A1SingleObjectStage,
    exc: Exception,
    object_id: str | None,
    statistics: Mapping[str, StatisticsValue],
    warnings: Tuple[ExportIssue, ...] = (),
) -> ExportResult:
    if not isinstance(stage, A1SingleObjectStage):
        raise TypeError("stage must be A1SingleObjectStage")
    if not isinstance(exc, Exception):
        raise TypeError("exc must be Exception")
    if not isinstance(warnings, tuple) or not all(
        isinstance(issue, ExportIssue) for issue in warnings
    ):
        raise TypeError("warnings must be a tuple of ExportIssue values")

    technical_details = f"{type(exc).__name__}: {exc}"
    logger.exception(
        "A1 single-object export failed at stage %s for '%s'",
        stage.value,
        object_id,
    )
    error = ExportIssue(
        severity=IssueSeverity.ERROR,
        stage=stage.value,
        code=stage.error_code,
        message=str(exc) or type(exc).__name__,
        object_id=object_id,
        technical_details=technical_details,
        context={"exception_type": type(exc).__name__},
    )
    return ExportResult(
        success=False,
        issues=warnings + (error,),
        statistics=dict(statistics),
    )


def export_a1_single_object(
    source_obj: Any,
    settings: A1SingleObjectExportSettings,
    *,
    context: Any | None = None,
    scene: Any | None = None,
) -> ExportResult:
    """Prepare, stage textures, finalize JSON, and atomically commit one export.

    Source Object, Mesh, materials, selection, active object, mode, frame, render settings,
    and temporary image datablocks are restored on success and failure. JSON is reserved
    before textures to preserve the public output order, but its bytes are written only after
    a camera render has produced the final sequence-union crop and screen-space hull.
    """

    try:
        prepared = prepare_a1_object(
            source_obj,
            settings,
            context=context,
            scene=scene,
        )
    except A1ObjectPreparationError as exc:
        return _failure_result(
            stage=exc.stage,
            exc=exc.cause,
            object_id=exc.object_id,
            statistics=exc.statistics,
            warnings=exc.warnings,
        )
    except Exception as exc:
        return _failure_result(
            stage=A1SingleObjectStage.VALIDATE_REQUEST,
            exc=exc,
            object_id=None,
            statistics={},
        )

    stage = A1SingleObjectStage.STAGE_OUTPUTS
    statistics = dict(prepared.statistics)
    try:
        with atomic_file_transaction() as output_transaction:
            json_reservation = output_transaction.reserve(
                prepared.output_paths.json_path
            )
            texture_stage = stage_texture_plan_outputs(
                prepared.source_object,
                prepared.bake_target_snapshot,
                prepared.bake_plan,
                output_transaction,
                settings.bake_execution,
                context=context,
                scene=scene,
            )

            stage = A1SingleObjectStage.ASSEMBLE_DOCUMENT
            finalized = finalize_prepared_camera_projection(
                prepared,
                texture_stage.projection_layout,
            )
            statistics = dict(finalized.statistics)
            json_text = SpineSerializer().to_json(
                finalized.document,
                indent=settings.json_indent,
            )
            write_staged_utf8_text(
                json_reservation.staged_path,
                json_text,
                ensure_trailing_newline=True,
            )

            stage = A1SingleObjectStage.COMMIT_OUTPUTS
            committed_paths = output_transaction.commit()

        expected_paths = (
            json_reservation.final_path,
            *(reservation.final_path for reservation in texture_stage.reservations),
        )
        if tuple(committed_paths) != expected_paths:
            raise AtomicFileCommitError(
                "Committed output order does not match reserved JSON and texture files"
            )
        statistics["output_file_count"] = len(committed_paths)
        logger.info(
            "A1 single-object export completed for '%s': %s",
            finalized.object_id,
            tuple(str(path) for path in committed_paths),
        )
        return ExportResult(
            success=True,
            output_files=tuple(committed_paths),
            issues=finalized.warnings,
            statistics=statistics,
        )
    except Exception as exc:
        return _failure_result(
            stage=stage,
            exc=exc,
            object_id=prepared.object_id,
            statistics=statistics,
            warnings=prepared.warnings,
        )
