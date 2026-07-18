"""Post-render atomic output service for mixed connected/standalone A1 exports."""

from __future__ import annotations

from dataclasses import replace
import logging
from typing import Any, Tuple

from ..application import (
    A1MultiObjectExportSettings,
    A1MultiObjectMode,
    A1MultiObjectStage,
    ExportResult,
)
from ..domain.spine import (
    ConstraintOrderPolicy,
    SpineCompositionSettings,
    SpineDocumentComponent,
    SpineSerializer,
    compose_spine_documents,
)
from ..infrastructure import (
    AtomicFileCommitError,
    atomic_file_transaction,
    write_staged_utf8_text,
)
from .a1_grouped_output import apply_staged_grouped_camera_overlay
from .a1_mixed_object_export import (
    build_connected_subgroup_settings,
    prepare_a1_mixed_object,
)
from .a1_multi_object_composition import compose_a1_multi_object_document
from .a1_multi_object_export import (
    A1MultiObjectPreparationError,
    A1MultiObjectSource,
)
from .a1_multi_object_result import build_multi_object_failure_result
from .a1_object_preparation import PreparedA1Object
from .a1_output_staging import stage_and_finalize_a1_objects
from .a1_output_statistics import (
    record_final_document_statistics,
    record_grouped_camera_statistics,
)
from .grouped_camera_projection_executor import (
    stage_grouped_camera_projection_outputs,
)
from .grouped_camera_projection_policy import (
    resolve_grouped_camera_projection_request,
)


logger = logging.getLogger(__name__)
_OPERATION = "A1 mixed-object output"
_TRANSACTION_NAME = "a1-mixed-object"


def _standalone_settings(
    settings: A1MultiObjectExportSettings,
) -> A1MultiObjectExportSettings:
    return replace(
        settings,
        mode=A1MultiObjectMode.STANDALONE,
        output_stem=f"{settings.resolved_output_stem}__standalone",
        anchor_component_id=None,
    )


def _compose_mixed_document_from_groups(
    connected_document,
    standalone_document,
    settings: A1MultiObjectExportSettings,
):
    return compose_spine_documents(
        (
            SpineDocumentComponent(
                component_id="connected_group",
                document=connected_document,
            ),
            SpineDocumentComponent(
                component_id="standalone_group",
                document=standalone_document,
            ),
        ),
        SpineCompositionSettings(
            shared_bone_names=("root",),
            constraint_order_policy=ConstraintOrderPolicy.REBASE_CONTIGUOUS,
            namespace_animations=False,
            animation_separator=settings.animation_separator,
        ),
    )


def _partition_finalized_objects(
    finalized_objects: Tuple[PreparedA1Object, ...],
    connected_sources: Tuple[A1MultiObjectSource, ...],
    standalone_sources: Tuple[A1MultiObjectSource, ...],
) -> tuple[Tuple[PreparedA1Object, ...], Tuple[PreparedA1Object, ...]]:
    connected_count = len(connected_sources)
    connected = tuple(finalized_objects[:connected_count])
    standalone = tuple(finalized_objects[connected_count:])
    if len(connected) != len(connected_sources):
        raise ValueError("finalized connected partition does not match sources")
    if len(standalone) != len(standalone_sources):
        raise ValueError("finalized standalone partition does not match sources")
    return connected, standalone


def export_a1_mixed_object(
    connected_sources: Tuple[A1MultiObjectSource, ...],
    standalone_sources: Tuple[A1MultiObjectSource, ...],
    settings: A1MultiObjectExportSettings,
    *,
    context: Any | None = None,
    scene: Any | None = None,
) -> ExportResult:
    """Finalize both groups, compose each once, then atomically commit one document."""

    try:
        prepared = prepare_a1_mixed_object(
            connected_sources,
            standalone_sources,
            settings,
            context=context,
            scene=scene,
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

    stage = A1MultiObjectStage.STAGE_OUTPUTS
    statistics = dict(prepared.statistics)
    try:
        with atomic_file_transaction(
            operation_name=_TRANSACTION_NAME,
        ) as transaction:
            json_reservation = transaction.reserve(prepared.json_path)
            staged_objects = stage_and_finalize_a1_objects(
                prepared,
                transaction,
                statistics,
                context=context,
                scene=scene,
            )
            statistics = dict(staged_objects.statistics)
            finalized_objects = staged_objects.objects

            connected_objects, standalone_objects = _partition_finalized_objects(
                finalized_objects,
                connected_sources,
                standalone_sources,
            )

            anchor = settings.anchor_component_id or connected_sources[0].component_id
            connected_settings = build_connected_subgroup_settings(settings, anchor)
            grouped_request = resolve_grouped_camera_projection_request(
                connected_objects,
                connected_settings,
            )
            grouped_stage = None
            if grouped_request is not None:
                grouped_stage = stage_grouped_camera_projection_outputs(
                    grouped_request.source_objects,
                    grouped_request.plan,
                    transaction,
                    grouped_request.execution_settings,
                    context=context,
                    scene=scene,
                )

            stage = A1MultiObjectStage.COMPOSE_DOCUMENT
            connected_composition = compose_a1_multi_object_document(
                connected_sources,
                connected_objects,
                connected_settings,
            )
            overlay = None
            if grouped_request is not None:
                if grouped_stage is None:
                    raise RuntimeError(
                        "grouped mixed request completed without a stage result"
                    )
                overlay = apply_staged_grouped_camera_overlay(
                    connected_composition.document,
                    grouped_request,
                    grouped_stage,
                )
                connected_composition = replace(
                    connected_composition,
                    document=overlay.document,
                )

            standalone_composition = compose_a1_multi_object_document(
                standalone_sources,
                standalone_objects,
                _standalone_settings(settings),
            )
            composition = _compose_mixed_document_from_groups(
                connected_composition.document,
                standalone_composition.document,
                settings,
            )
            document = composition.document
            record_final_document_statistics(
                statistics,
                document,
                finalized_objects,
                grouped_enabled=grouped_request is not None,
            )
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
            committed_paths = transaction.commit()

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
                "Committed output order does not match mixed JSON and texture reservations"
            )
        statistics["output_file_count"] = len(committed_paths)
        logger.info(
            "A1 mixed export completed (grouped_b4=%s): %s",
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
        return build_multi_object_failure_result(
            logger=logger,
            operation=_OPERATION,
            stage=stage,
            exc=exc,
            statistics=statistics,
            warnings=prepared.warnings,
        )


__all__ = ["export_a1_mixed_object"]
