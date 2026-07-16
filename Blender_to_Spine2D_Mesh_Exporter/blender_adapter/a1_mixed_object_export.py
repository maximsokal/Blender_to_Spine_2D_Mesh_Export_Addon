"""Mixed A1 multi-object export preserving the legacy Connect-flag semantics.

At least two connected objects form one typed ``all_objects`` document. Remaining
selected objects stay standalone. All documents are composed in memory and the final
JSON plus every texture frame are committed by one atomic transaction.
"""

from __future__ import annotations

from dataclasses import replace
import logging
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Tuple

from ..application import (
    A1MultiObjectExportSettings,
    A1MultiObjectMode,
    A1MultiObjectStage,
    ExportIssue,
    ExportResult,
    IssueSeverity,
)
from ..domain.spine import (
    ConstraintOrderPolicy,
    SpineCompositionSettings,
    SpineDocumentComponent,
    SpineDocumentCompositionResult,
    SpineSerializer,
    compose_spine_documents,
)
from ..infrastructure import (
    AtomicFileCommitError,
    atomic_file_transaction,
    write_staged_utf8_text,
)
from .a1_multi_object_export import (
    A1MultiObjectPreparationError,
    A1MultiObjectSource,
    PreparedA1MultiObject,
    prepare_a1_multi_object,
)
from .a1_object_preparation import (
    A1ObjectPreparationError,
    PreparedA1Object,
    StatisticsValue,
    prepare_a1_object,
)
from .bake_executor import stage_bake_plan_outputs

logger = logging.getLogger(__name__)


def _validate_mixed_sources(
    connected_sources: Tuple[A1MultiObjectSource, ...],
    standalone_sources: Tuple[A1MultiObjectSource, ...],
    settings: A1MultiObjectExportSettings,
) -> None:
    if not isinstance(settings, A1MultiObjectExportSettings):
        raise TypeError("settings must be A1MultiObjectExportSettings")
    if settings.mode is not A1MultiObjectMode.MIXED:
        raise ValueError("mixed export requires A1MultiObjectMode.MIXED")
    if not isinstance(connected_sources, tuple) or len(connected_sources) < 2:
        raise ValueError("connected_sources must contain at least two objects")
    if not isinstance(standalone_sources, tuple) or not standalone_sources:
        raise ValueError("standalone_sources must contain at least one object")

    all_sources = connected_sources + standalone_sources
    if not all(isinstance(item, A1MultiObjectSource) for item in all_sources):
        raise TypeError("all sources must contain A1MultiObjectSource values")
    component_ids = tuple(item.component_id for item in all_sources)
    if len(component_ids) != len(set(component_ids)):
        raise ValueError("component_id values must be unique across mixed groups")
    connected_ids = {item.component_id for item in connected_sources}
    if settings.anchor_component_id is not None and (
        settings.anchor_component_id not in connected_ids
    ):
        raise ValueError(
            "anchor_component_id must identify an object in connected_sources"
        )

    output_root = settings.output_directory.expanduser().resolve(strict=False)
    for source in all_sources:
        source_root = source.settings.export.output_directory.expanduser().resolve(
            strict=False
        )
        if source_root != output_root:
            raise ValueError(
                f"Component '{source.component_id}' uses output root '{source_root}', "
                f"but mixed export uses '{output_root}'"
            )


def _connected_settings(
    settings: A1MultiObjectExportSettings,
    anchor_component_id: str,
) -> A1MultiObjectExportSettings:
    return replace(
        settings,
        mode=A1MultiObjectMode.CONNECTED,
        output_stem=f"{settings.resolved_output_stem}__connected",
        anchor_component_id=anchor_component_id,
    )


def _prepare_standalone_objects(
    sources: Tuple[A1MultiObjectSource, ...],
    settings: A1MultiObjectExportSettings,
    *,
    context: Any | None,
    scene: Any | None,
) -> tuple[
    Tuple[PreparedA1Object, ...],
    SpineDocumentCompositionResult,
    Tuple[ExportIssue, ...],
    Mapping[str, StatisticsValue],
]:
    prepared_objects: list[PreparedA1Object] = []
    warnings: list[ExportIssue] = []
    statistics: dict[str, StatisticsValue] = {
        "object_count": len(sources),
        "mode": A1MultiObjectMode.STANDALONE.value,
    }

    for source in sources:
        try:
            prepared = prepare_a1_object(
                source.source_object,
                source.settings,
                context=context,
                scene=scene,
            )
        except A1ObjectPreparationError as exc:
            warnings.extend(exc.warnings)
            raise A1MultiObjectPreparationError(
                stage=A1MultiObjectStage.PREPARE_OBJECTS,
                cause=exc.cause,
                statistics=statistics,
                warnings=tuple(warnings),
                component_id=source.component_id,
                object_id=exc.object_id,
                object_stage=exc.stage.value,
            ) from exc
        prepared_objects.append(prepared)
        warnings.extend(prepared.warnings)
        for key, value in prepared.statistics.items():
            statistics[f"component.{source.component_id}.{key}"] = value

    components = tuple(
        SpineDocumentComponent(
            component_id=source.component_id,
            document=prepared.document,
            animation_namespace=source.animation_namespace or source.component_id,
        )
        for source, prepared in zip(sources, prepared_objects)
    )
    composition = compose_spine_documents(
        components,
        SpineCompositionSettings(
            shared_bone_names=("root",),
            constraint_order_policy=ConstraintOrderPolicy.REBASE_CONTIGUOUS,
            namespace_animations=settings.namespace_animations,
            animation_separator=settings.animation_separator,
        ),
    )
    statistics.update(
        {
            "final_bone_count": len(composition.document.bones),
            "final_slot_count": len(composition.document.slots),
            "final_skin_count": len(composition.document.skins),
            "final_constraint_count": len(composition.document.ik)
            + len(composition.document.transform),
        }
    )
    return (
        tuple(prepared_objects),
        composition,
        tuple(warnings),
        MappingProxyType(statistics),
    )


def _texture_paths(objects: Tuple[PreparedA1Object, ...]) -> Tuple[Path, ...]:
    return tuple(
        task.output_path.expanduser().resolve(strict=False)
        for prepared in objects
        for task in prepared.bake_plan.frame_tasks
    )


def _validate_final_paths(
    json_path: Path,
    connected_paths: Tuple[Path, ...],
    standalone_paths: Tuple[Path, ...],
) -> Tuple[Path, ...]:
    resolved_json = json_path.expanduser().resolve(strict=False)
    owner_by_path: dict[Path, str] = {resolved_json: "final mixed JSON"}
    result: list[Path] = []
    for group_name, paths in (
        ("connected", connected_paths),
        ("standalone", standalone_paths),
    ):
        for path in paths:
            resolved = path.expanduser().resolve(strict=False)
            previous = owner_by_path.get(resolved)
            if previous is not None:
                raise ValueError(
                    f"Mixed output path collision '{resolved}' between {previous} "
                    f"and {group_name} group"
                )
            owner_by_path[resolved] = f"{group_name} texture"
            result.append(resolved)
    return tuple(result)


def prepare_a1_mixed_object(
    connected_sources: Tuple[A1MultiObjectSource, ...],
    standalone_sources: Tuple[A1MultiObjectSource, ...],
    settings: A1MultiObjectExportSettings,
    *,
    context: Any | None = None,
    scene: Any | None = None,
) -> PreparedA1MultiObject:
    """Prepare connected and standalone subgroups without writing output files."""

    _validate_mixed_sources(connected_sources, standalone_sources, settings)
    anchor = settings.anchor_component_id or connected_sources[0].component_id
    connected = prepare_a1_multi_object(
        connected_sources,
        _connected_settings(settings, anchor),
        context=context,
        scene=scene,
    )
    (
        standalone_objects,
        standalone_composition,
        standalone_warnings,
        standalone_statistics,
    ) = _prepare_standalone_objects(
        standalone_sources,
        settings,
        context=context,
        scene=scene,
    )

    # Both subgroups already applied their per-object animation namespaces. The outer
    # composition preserves these names and only performs the final bone-index remap.
    composition = compose_spine_documents(
        (
            SpineDocumentComponent(
                component_id="connected_group",
                document=connected.document,
            ),
            SpineDocumentComponent(
                component_id="standalone_group",
                document=standalone_composition.document,
            ),
        ),
        SpineCompositionSettings(
            shared_bone_names=("root",),
            constraint_order_policy=ConstraintOrderPolicy.REBASE_CONTIGUOUS,
            namespace_animations=False,
            animation_separator=settings.animation_separator,
        ),
    )
    connected_paths = connected.texture_output_paths
    standalone_paths = _texture_paths(standalone_objects)
    texture_paths = _validate_final_paths(
        settings.json_path,
        connected_paths,
        standalone_paths,
    )

    statistics: dict[str, StatisticsValue] = {
        "object_count": len(connected_sources) + len(standalone_sources),
        "connected_object_count": len(connected_sources),
        "standalone_object_count": len(standalone_sources),
        "mode": A1MultiObjectMode.MIXED.value,
        "texture_output_count": len(texture_paths),
        "final_bone_count": len(composition.document.bones),
        "final_slot_count": len(composition.document.slots),
        "final_skin_count": len(composition.document.skins),
        "final_constraint_count": len(composition.document.ik)
        + len(composition.document.transform),
    }
    for prefix, values in (
        ("connected", connected.statistics),
        ("standalone", standalone_statistics),
    ):
        for key, value in values.items():
            statistics[f"{prefix}.{key}"] = value

    return PreparedA1MultiObject(
        settings=settings,
        sources=connected.sources + standalone_sources,
        objects=connected.objects + standalone_objects,
        document=composition.document,
        composition=composition,
        texture_output_paths=texture_paths,
        warnings=connected.warnings + standalone_warnings,
        statistics=MappingProxyType(statistics),
    )


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
        "A1 mixed multi-object export failed at %s (component=%s, object=%s)",
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


def export_a1_mixed_object(
    connected_sources: Tuple[A1MultiObjectSource, ...],
    standalone_sources: Tuple[A1MultiObjectSource, ...],
    settings: A1MultiObjectExportSettings,
    *,
    context: Any | None = None,
    scene: Any | None = None,
) -> ExportResult:
    """Export mixed connected and standalone objects in one atomic transaction."""

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

    stage = A1MultiObjectStage.SERIALIZE_DOCUMENT
    statistics = dict(prepared.statistics)
    try:
        json_text = SpineSerializer().to_json(
            prepared.document,
            indent=settings.json_indent,
        )
        stage = A1MultiObjectStage.STAGE_OUTPUTS
        with atomic_file_transaction() as transaction:
            json_reservation = transaction.reserve(prepared.json_path)
            write_staged_utf8_text(
                json_reservation.staged_path,
                json_text,
                ensure_trailing_newline=True,
            )
            bake_reservations = []
            for item in prepared.objects:
                bake_reservations.extend(
                    stage_bake_plan_outputs(
                        item.source_object,
                        item.bake_target_snapshot,
                        item.bake_plan,
                        transaction,
                        item.settings.bake_execution,
                        context=context,
                        scene=scene,
                    )
                )
            stage = A1MultiObjectStage.COMMIT_OUTPUTS
            committed_paths = transaction.commit()

        expected_paths = (
            json_reservation.final_path,
            *(reservation.final_path for reservation in bake_reservations),
        )
        if tuple(committed_paths) != expected_paths:
            raise AtomicFileCommitError(
                "Committed output order does not match mixed JSON and texture "
                "reservations"
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
