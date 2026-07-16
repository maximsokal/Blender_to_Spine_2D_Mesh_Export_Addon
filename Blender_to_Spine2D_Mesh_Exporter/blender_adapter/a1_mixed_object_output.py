"""Post-render atomic output service for mixed connected/standalone A1 exports."""

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
from .a1_mixed_object_export import (
    _connected_settings,
    prepare_a1_mixed_object,
)
from .a1_multi_object_export import (
    A1MultiObjectPreparationError,
    A1MultiObjectSource,
    _compose_document,
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
    issue_context: dict[str, object] = {"exception_type": type(exc).__name__}
    if component_id is not None:
        issue_context["component_id"] = component_id
    if object_stage is not None:
        issue_context["object_stage"] = object_stage
    logger.exception(
        "A1 mixed output failed at %s (component=%s, object=%s)",
        stage.value,
        component_id,
        object_id,
    )
    return ExportResult(
        success=False,
        issues=warnings
        + (
            ExportIssue(
                severity=IssueSeverity.ERROR,
                stage=stage.value,
                code=stage.error_code,
                message=str(exc) or type(exc).__name__,
                object_id=object_id,
                technical_details=f"{type(exc).__name__}: {exc}",
                context=issue_context,
            ),
        ),
        statistics=dict(statistics),
    )


def _compose_standalone_group(
    sources: Tuple[A1MultiObjectSource, ...],
    objects,
    settings: A1MultiObjectExportSettings,
):
    components = tuple(
        SpineDocumentComponent(
            component_id=source.component_id,
            document=item.document,
            animation_namespace=source.animation_namespace or source.component_id,
        )
        for source, item in zip(sources, objects)
    )
    return compose_spine_documents(
        components,
        SpineCompositionSettings(
            shared_bone_names=("root",),
            constraint_order_policy=ConstraintOrderPolicy.REBASE_CONTIGUOUS,
            namespace_animations=settings.namespace_animations,
            animation_separator=settings.animation_separator,
        ),
    )


def _compose_mixed_document(
    connected_sources: Tuple[A1MultiObjectSource, ...],
    standalone_sources: Tuple[A1MultiObjectSource, ...],
    connected_objects,
    standalone_objects,
    settings: A1MultiObjectExportSettings,
):
    anchor = settings.anchor_component_id or connected_sources[0].component_id
    connected = _compose_document(
        connected_sources,
        tuple(connected_objects),
        _connected_settings(settings, anchor),
    )
    standalone = _compose_standalone_group(
        standalone_sources,
        tuple(standalone_objects),
        settings,
    )
    return compose_spine_documents(
        (
            SpineDocumentComponent(
                component_id="connected_group",
                document=connected.document,
            ),
            SpineDocumentComponent(
                component_id="standalone_group",
                document=standalone.document,
            ),
        ),
        SpineCompositionSettings(
            shared_bone_names=("root",),
            constraint_order_policy=ConstraintOrderPolicy.REBASE_CONTIGUOUS,
            namespace_animations=False,
            animation_separator=settings.animation_separator,
        ),
    )


def export_a1_mixed_object(
    connected_sources: Tuple[A1MultiObjectSource, ...],
    standalone_sources: Tuple[A1MultiObjectSource, ...],
    settings: A1MultiObjectExportSettings,
    *,
    context: Any | None = None,
    scene: Any | None = None,
) -> ExportResult:
    """Stage all mixed textures, rebuild B4 documents, and commit one transaction."""

    try:
        prepared = prepare_a1_mixed_object(
            connected_sources,
            standalone_sources,
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
            stage=A1MultiObjectStage.COMPOSE_DOCUMENT,
            exc=exc,
            statistics={},
            warnings=(),
        )

    stage = A1MultiObjectStage.STAGE_OUTPUTS
    statistics = dict(prepared.statistics)
    try:
        with atomic_file_transaction() as transaction:
            json_reservation = transaction.reserve(prepared.json_path)
            reservations = []
            finalized = []
            for source, item in zip(prepared.sources, prepared.objects):
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
                resolved = finalize_prepared_camera_projection(
                    item,
                    staged.projection_layout,
                )
                finalized.append(resolved)
                for key, value in resolved.statistics.items():
                    statistics[f"component.{source.component_id}.{key}"] = value

            connected_count = len(connected_sources)
            connected_objects = tuple(finalized[:connected_count])
            standalone_objects = tuple(finalized[connected_count:])
            if len(standalone_objects) != len(standalone_sources):
                raise ValueError("finalized mixed object partition does not match sources")

            stage = A1MultiObjectStage.COMPOSE_DOCUMENT
            composition = _compose_mixed_document(
                connected_sources,
                standalone_sources,
                connected_objects,
                standalone_objects,
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
                        for item in finalized
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
            committed_paths = transaction.commit()

        expected_paths = (
            json_reservation.final_path,
            *(reservation.final_path for reservation in reservations),
        )
        if tuple(committed_paths) != expected_paths:
            raise AtomicFileCommitError(
                "Committed output order does not match mixed JSON and texture reservations"
            )
        statistics["output_file_count"] = len(committed_paths)
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


__all__ = ["export_a1_mixed_object"]
