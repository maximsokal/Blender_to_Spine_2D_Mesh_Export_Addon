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
    apply_grouped_camera_overlay,
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
    build_connected_subgroup_settings,
    prepare_a1_mixed_object,
)
from .a1_multi_object_composition import compose_a1_multi_object_document
from .a1_multi_object_export import (
    A1MultiObjectPreparationError,
    A1MultiObjectSource,
    record_object_statistics,
)
from .a1_multi_object_result import build_multi_object_failure_result
from .a1_projection_finalization import finalize_prepared_camera_projection
from .grouped_camera_projection_executor import (
    stage_grouped_camera_projection_outputs,
)
from .grouped_camera_projection_policy import (
    resolve_grouped_camera_projection_request,
)
from .texture_executor import stage_texture_plan_outputs

logger = logging.getLogger(__name__)


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
            operation="A1 mixed-object output",
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
            operation="A1 mixed-object output",
            stage=A1MultiObjectStage.VALIDATE_REQUEST,
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
                record_object_statistics(
                    statistics,
                    source.component_id,
                    resolved.statistics,
                )

            connected_count = len(connected_sources)
            connected_objects = tuple(finalized[:connected_count])
            standalone_objects = tuple(finalized[connected_count:])
            if len(connected_objects) != len(connected_sources):
                raise ValueError("finalized connected partition does not match sources")
            if len(standalone_objects) != len(standalone_sources):
                raise ValueError("finalized standalone partition does not match sources")

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
                reservations.extend(grouped_stage.reservations)

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
                overlay = apply_grouped_camera_overlay(
                    connected_composition.document,
                    grouped_request.plan,
                    grouped_stage.layout,
                    visual_slot_names=grouped_request.visual_slot_names,
                    image_relative_directory=(
                        grouped_request.image_relative_directory
                    ),
                    slot_name=grouped_request.slot_name,
                    attachment_name=grouped_request.attachment_name,
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
            statistics.update(
                {
                    "final_bone_count": len(document.bones),
                    "final_slot_count": len(document.slots),
                    "final_skin_count": len(document.skins),
                    "final_constraint_count": (
                        len(document.ik) + len(document.transform)
                    ),
                    "projection_cropped_component_count": sum(
                        1
                        for item in finalized
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
            operation="A1 mixed-object output",
            stage=stage,
            exc=exc,
            statistics=statistics,
            warnings=prepared.warnings,
        )


__all__ = ["export_a1_mixed_object"]
