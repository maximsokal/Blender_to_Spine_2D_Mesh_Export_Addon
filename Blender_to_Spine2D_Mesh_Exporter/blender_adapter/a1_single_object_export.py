"""Atomic output wrapper for one prepared A1 Blender object.

All geometry, UV, material, rig, and initial document construction lives in
``prepare_a1_object``. Camera projection textures additionally produce a render-derived
crop/hull layout; this module finalizes that attachment before serializing JSON, while all
files still share one caller-owned atomic transaction.
"""

from __future__ import annotations

import logging
from typing import Any

from ..application import (
    A1ExportProgressCallback,
    A1SingleObjectExportSettings,
    A1SingleObjectStage,
    ExportResult,
    emit_a1_export_progress,
    scale_a1_export_progress_callback,
)
from ..domain.spine import SpineSerializer
from ..infrastructure import (
    AtomicFileCommitError,
    atomic_file_transaction,
    write_staged_utf8_text,
)
from .a1_export_result import build_a1_failure_result
from .a1_object_preparation import A1ObjectPreparationError, prepare_a1_object
from .a1_projection_finalization import finalize_prepared_camera_projection
from .bake_uv_spine_validation import validate_staged_normal_bake_coverage
from .source_uv_integrity import (
    capture_source_uv_fingerprint,
    require_object_mode,
    require_source_uv_unchanged,
)
from .texture_executor import stage_texture_plan_outputs


logger = logging.getLogger(__name__)
_OPERATION = "A1 single-object output"


def export_a1_single_object(
    source_obj: Any,
    settings: A1SingleObjectExportSettings,
    *,
    context: Any | None = None,
    scene: Any | None = None,
    progress_callback: A1ExportProgressCallback | None = None,
) -> ExportResult:
    """Prepare, bake, validate UV pixels, and atomically commit one object."""

    emit_a1_export_progress(
        progress_callback,
        percent=0,
        stage=A1SingleObjectStage.VALIDATE_REQUEST,
        message="Starting single-object export",
    )
    try:
        require_object_mode(context)
        export_uv_fingerprint = capture_source_uv_fingerprint(source_obj)
    except Exception as exc:
        return build_a1_failure_result(
            logger=logger,
            operation=_OPERATION,
            stage=A1SingleObjectStage.VALIDATE_REQUEST,
            exc=exc,
            object_id=None,
            statistics={},
        )

    preparation_progress = scale_a1_export_progress_callback(
        progress_callback,
        start_percent=5.0,
        end_percent=60.0,
    )
    try:
        prepared = prepare_a1_object(
            source_obj,
            settings,
            context=context,
            scene=scene,
            progress_callback=preparation_progress,
        )
    except A1ObjectPreparationError as exc:
        return build_a1_failure_result(
            logger=logger,
            operation=_OPERATION,
            stage=exc.stage,
            exc=exc.cause,
            object_id=exc.object_id,
            statistics=exc.statistics,
            warnings=exc.warnings,
        )
    except Exception as exc:
        return build_a1_failure_result(
            logger=logger,
            operation=_OPERATION,
            stage=A1SingleObjectStage.VALIDATE_REQUEST,
            exc=exc,
            object_id=None,
            statistics={},
        )

    stage = A1SingleObjectStage.STAGE_OUTPUTS
    statistics = dict(prepared.statistics)
    try:
        emit_a1_export_progress(
            progress_callback,
            percent=65,
            stage=stage,
            message="Staging texture outputs",
            object_id=prepared.object_id,
        )
        texture_progress = scale_a1_export_progress_callback(
            progress_callback,
            start_percent=65.0,
            end_percent=80.0,
            object_id=prepared.object_id,
        )
        with atomic_file_transaction(operation_name="a1-single-object") as output_transaction:
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
                progress_callback=texture_progress,
            )

            stage = A1SingleObjectStage.ASSEMBLE_DOCUMENT
            emit_a1_export_progress(
                progress_callback,
                percent=81,
                stage=stage,
                message="Validating baked UV-to-Spine pixel coverage",
                object_id=prepared.object_id,
            )
            coverage_samples = validate_staged_normal_bake_coverage(
                prepared,
                texture_stage.reservations,
            )
            statistics["bake_uv_coverage_sample_count"] = len(coverage_samples)

            # The semantic bake owner has restored every Blender context and material
            # transaction at this point. Verify the original source UV datablock before
            # any staged JSON or texture can become visible at its final path.
            require_source_uv_unchanged(export_uv_fingerprint, source_obj)

            emit_a1_export_progress(
                progress_callback,
                percent=84,
                stage=stage,
                message="Finalizing render-derived attachment layout",
                object_id=prepared.object_id,
            )
            finalized = finalize_prepared_camera_projection(
                prepared,
                texture_stage.projection_layout,
            )
            statistics.update(finalized.statistics)
            statistics["bake_uv_coverage_sample_count"] = len(coverage_samples)
            emit_a1_export_progress(
                progress_callback,
                percent=90,
                stage=stage,
                message="Serializing Spine JSON",
                object_id=prepared.object_id,
            )
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
            emit_a1_export_progress(
                progress_callback,
                percent=97,
                stage=stage,
                message="Committing JSON and texture files",
                object_id=prepared.object_id,
            )
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
        emit_a1_export_progress(
            progress_callback,
            percent=100,
            stage=A1SingleObjectStage.COMMIT_OUTPUTS,
            message="Export complete",
            object_id=finalized.object_id,
        )
        return ExportResult(
            success=True,
            output_files=tuple(committed_paths),
            issues=finalized.warnings,
            statistics=statistics,
        )
    except Exception as exc:
        try:
            require_source_uv_unchanged(export_uv_fingerprint, source_obj)
        except Exception as mutation_exc:
            stage = A1SingleObjectStage.READ_GEOMETRY
            exc = mutation_exc
        return build_a1_failure_result(
            logger=logger,
            operation=_OPERATION,
            stage=stage,
            exc=exc,
            object_id=prepared.object_id,
            statistics=statistics,
            warnings=prepared.warnings,
        )


__all__ = ["export_a1_single_object"]
