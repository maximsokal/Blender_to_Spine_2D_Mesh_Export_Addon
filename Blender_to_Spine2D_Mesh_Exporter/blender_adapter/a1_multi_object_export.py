"""Prepare and atomically export several Blender objects into one A1 document.

The service never calls the single-object output function and never merges serialized
JSON. Every object is prepared in memory, typed documents are composed with strict
weighted-index remapping, and one caller-owned transaction stages the final JSON and
all texture frames.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import logging
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Tuple

from ..application import (
    A1MultiObjectExportSettings,
    A1MultiObjectMode,
    A1MultiObjectStage,
    A1SingleObjectExportSettings,
    ExportIssue,
    ExportResult,
    IssueSeverity,
)
from ..domain.spine import (
    ConnectedGroupBuildResult,
    ConnectedGroupSettings,
    ConnectedObjectDocument,
    ConstraintOrderPolicy,
    SpineCompositionSettings,
    SpineDocument,
    SpineDocumentComponent,
    SpineDocumentCompositionResult,
    SpineSerializer,
    build_connected_group_document,
    compose_spine_documents,
)
from ..infrastructure import (
    AtomicFileCommitError,
    atomic_file_transaction,
    write_staged_utf8_text,
)
from .a1_object_preparation import (
    A1ObjectPreparationError,
    PreparedA1Object,
    StatisticsValue,
    prepare_a1_object,
)
from .bake_executor import stage_bake_plan_outputs

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class A1MultiObjectSource:
    source_object: Any
    component_id: str
    settings: A1SingleObjectExportSettings
    animation_namespace: str | None = None

    def __post_init__(self) -> None:
        if self.source_object is None:
            raise ValueError("source_object cannot be None")
        if not isinstance(self.component_id, str) or not self.component_id.strip():
            raise ValueError("component_id must be a non-empty string")
        if not isinstance(self.settings, A1SingleObjectExportSettings):
            raise TypeError("settings must be A1SingleObjectExportSettings")
        if self.animation_namespace is not None and (
            not isinstance(self.animation_namespace, str)
            or not self.animation_namespace.strip()
        ):
            raise ValueError(
                "animation_namespace must be a non-empty string or None"
            )


@dataclass(frozen=True, slots=True)
class PreparedA1MultiObject:
    settings: A1MultiObjectExportSettings
    sources: Tuple[A1MultiObjectSource, ...]
    objects: Tuple[PreparedA1Object, ...]
    document: SpineDocument
    composition: SpineDocumentCompositionResult | ConnectedGroupBuildResult
    texture_output_paths: Tuple[Path, ...]
    warnings: Tuple[ExportIssue, ...]
    statistics: Mapping[str, StatisticsValue]

    def __post_init__(self) -> None:
        if not isinstance(self.settings, A1MultiObjectExportSettings):
            raise TypeError("settings must be A1MultiObjectExportSettings")
        if not isinstance(self.sources, tuple) or not self.sources:
            raise ValueError("sources must be a non-empty tuple")
        if not all(isinstance(item, A1MultiObjectSource) for item in self.sources):
            raise TypeError("sources must contain A1MultiObjectSource values")
        if not isinstance(self.objects, tuple) or len(self.objects) != len(self.sources):
            raise ValueError("objects must correspond one-to-one with sources")
        if not all(isinstance(item, PreparedA1Object) for item in self.objects):
            raise TypeError("objects must contain PreparedA1Object values")
        if not isinstance(self.document, SpineDocument):
            raise TypeError("document must be SpineDocument")
        if not isinstance(
            self.composition,
            (SpineDocumentCompositionResult, ConnectedGroupBuildResult),
        ):
            raise TypeError("composition has an unsupported result type")
        if not isinstance(self.texture_output_paths, tuple) or not all(
            isinstance(path, Path) for path in self.texture_output_paths
        ):
            raise TypeError("texture_output_paths must be a tuple of Path values")
        if not isinstance(self.warnings, tuple) or not all(
            isinstance(issue, ExportIssue) for issue in self.warnings
        ):
            raise TypeError("warnings must be a tuple of ExportIssue values")
        if not isinstance(self.statistics, Mapping):
            raise TypeError("statistics must be a mapping")

    @property
    def json_path(self) -> Path:
        return self.settings.json_path


class A1MultiObjectPreparationError(RuntimeError):
    """Capture one failed multi-object preparation stage and object substage."""

    def __init__(
        self,
        *,
        stage: A1MultiObjectStage,
        cause: Exception,
        statistics: Mapping[str, StatisticsValue],
        warnings: Tuple[ExportIssue, ...],
        component_id: str | None = None,
        object_id: str | None = None,
        object_stage: str | None = None,
    ) -> None:
        if not isinstance(stage, A1MultiObjectStage):
            raise TypeError("stage must be A1MultiObjectStage")
        if not isinstance(cause, Exception):
            raise TypeError("cause must be Exception")
        self.stage = stage
        self.cause = cause
        self.statistics = MappingProxyType(dict(statistics))
        self.warnings = warnings
        self.component_id = component_id
        self.object_id = object_id
        self.object_stage = object_stage
        message = str(cause) or type(cause).__name__
        details = ""
        if component_id is not None:
            details += f" component='{component_id}'"
        if object_stage is not None:
            details += f" object_stage='{object_stage}'"
        super().__init__(
            f"A1 multi-object preparation failed at {stage.value}{details}: {message}"
        )


def _validate_sources(
    sources: Tuple[A1MultiObjectSource, ...],
    settings: A1MultiObjectExportSettings,
) -> None:
    if not isinstance(sources, tuple) or len(sources) < 2:
        raise ValueError("sources must contain at least two objects")
    if not all(isinstance(item, A1MultiObjectSource) for item in sources):
        raise TypeError("sources must contain A1MultiObjectSource values")
    component_ids = tuple(item.component_id for item in sources)
    if len(component_ids) != len(set(component_ids)):
        raise ValueError("component_id values must be unique")
    if settings.anchor_component_id is not None and (
        settings.anchor_component_id not in set(component_ids)
    ):
        raise ValueError("anchor_component_id is not present in sources")

    output_root = settings.output_directory.expanduser().resolve(strict=False)
    for item in sources:
        object_root = item.settings.export.output_directory.expanduser().resolve(
            strict=False
        )
        if object_root != output_root:
            raise ValueError(
                f"Component '{item.component_id}' uses output root '{object_root}', "
                f"but the multi-object transaction uses '{output_root}'"
            )


def _settings_for_preparation(
    source: A1MultiObjectSource,
    mode: A1MultiObjectMode,
) -> A1SingleObjectExportSettings:
    if mode is A1MultiObjectMode.CONNECTED:
        # Connected placement is applied exactly once by the typed global rig using
        # source_snapshot.world_matrix. A standalone main-bone world offset here
        # would double the translation.
        return replace(
            source.settings,
            use_world_location_for_main_bone=False,
        )
    return source.settings


def _record_object_statistics(
    target: dict[str, StatisticsValue],
    component_id: str,
    values: Mapping[str, StatisticsValue],
) -> None:
    for key, value in values.items():
        target[f"component.{component_id}.{key}"] = value


def _validate_prepared_outputs(
    prepared: Tuple[PreparedA1Object, ...],
    settings: A1MultiObjectExportSettings,
) -> Tuple[Path, ...]:
    prefixes = tuple(item.prefix for item in prepared)
    if len(prefixes) != len(set(prefixes)):
        raise ValueError("prepared rig prefixes must be unique")

    output_root = settings.output_directory.expanduser().resolve(strict=False)
    final_json = settings.json_path.expanduser().resolve(strict=False)
    paths: list[Path] = []
    owner_by_path: dict[Path, str] = {final_json: "final JSON"}
    for item in prepared:
        for task in item.bake_plan.frame_tasks:
            path = task.output_path.expanduser().resolve(strict=False)
            try:
                path.relative_to(output_root)
            except ValueError as exc:
                raise ValueError(
                    f"Texture output for '{item.object_id}' escapes the multi-object "
                    f"root: {path}"
                ) from exc
            previous_owner = owner_by_path.get(path)
            if previous_owner is not None:
                raise ValueError(
                    f"Output path collision '{path}' between {previous_owner} and "
                    f"component '{item.object_id}'"
                )
            owner_by_path[path] = item.object_id
            paths.append(path)
    if not paths:
        raise ValueError("multi-object export contains no texture output tasks")
    return tuple(paths)


def _compose_document(
    sources: Tuple[A1MultiObjectSource, ...],
    prepared: Tuple[PreparedA1Object, ...],
    settings: A1MultiObjectExportSettings,
) -> SpineDocumentCompositionResult | ConnectedGroupBuildResult:
    if settings.mode is A1MultiObjectMode.STANDALONE:
        components = tuple(
            SpineDocumentComponent(
                component_id=source.component_id,
                document=item.document,
                animation_namespace=(
                    source.animation_namespace or source.component_id
                ),
            )
            for source, item in zip(sources, prepared)
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

    dimensions = {
        (
            item.settings.export.texture_width,
            item.settings.export.texture_height,
        )
        for item in prepared
    }
    if len(dimensions) != 1:
        raise ValueError(
            "CONNECTED mode requires identical texture dimensions for every object"
        )
    texture_width, texture_height = next(iter(dimensions))
    connected_objects = tuple(
        ConnectedObjectDocument(
            component_id=source.component_id,
            prefix=item.prefix,
            document=item.document,
            world_position=item.world_position,
            animation_namespace=(
                source.animation_namespace or source.component_id
            ),
        )
        for source, item in zip(sources, prepared)
    )
    return build_connected_group_document(
        connected_objects,
        ConnectedGroupSettings(
            texture_width=texture_width,
            texture_height=texture_height,
            group_prefix=settings.connected_group_prefix,
            anchor_component_id=settings.anchor_component_id,
            z_tolerance=settings.z_tolerance,
            scale_mode=settings.connected_scale_mode,
            animation_separator=settings.animation_separator,
        ),
    )


def prepare_a1_multi_object(
    sources: Tuple[A1MultiObjectSource, ...],
    settings: A1MultiObjectExportSettings,
    *,
    context: Any | None = None,
    scene: Any | None = None,
) -> PreparedA1MultiObject:
    """Prepare all objects and compose one final document without writing files."""

    stage = A1MultiObjectStage.VALIDATE_REQUEST
    statistics: dict[str, StatisticsValue] = {}
    warnings: list[ExportIssue] = []
    current_component: str | None = None

    try:
        if not isinstance(settings, A1MultiObjectExportSettings):
            raise TypeError("settings must be A1MultiObjectExportSettings")
        _validate_sources(sources, settings)
        statistics.update(
            {
                "object_count": len(sources),
                "mode": settings.mode.value,
                "output_stem": settings.resolved_output_stem,
            }
        )

        stage = A1MultiObjectStage.PREPARE_OBJECTS
        prepared_objects: list[PreparedA1Object] = []
        for source in sources:
            current_component = source.component_id
            try:
                prepared = prepare_a1_object(
                    source.source_object,
                    _settings_for_preparation(source, settings.mode),
                    context=context,
                    scene=scene,
                )
            except A1ObjectPreparationError as exc:
                warnings.extend(exc.warnings)
                raise A1MultiObjectPreparationError(
                    stage=stage,
                    cause=exc.cause,
                    statistics=statistics,
                    warnings=tuple(warnings),
                    component_id=source.component_id,
                    object_id=exc.object_id,
                    object_stage=exc.stage.value,
                ) from exc
            prepared_objects.append(prepared)
            warnings.extend(prepared.warnings)
            _record_object_statistics(
                statistics,
                source.component_id,
                prepared.statistics,
            )
        resolved_objects = tuple(prepared_objects)

        stage = A1MultiObjectStage.VALIDATE_OUTPUTS
        texture_paths = _validate_prepared_outputs(resolved_objects, settings)
        statistics["texture_output_count"] = len(texture_paths)

        stage = A1MultiObjectStage.COMPOSE_DOCUMENT
        composition = _compose_document(sources, resolved_objects, settings)
        document = composition.document
        statistics.update(
            {
                "final_bone_count": len(document.bones),
                "final_slot_count": len(document.slots),
                "final_skin_count": len(document.skins),
                "final_constraint_count": len(document.ik)
                + len(document.transform),
            }
        )
        if isinstance(composition, ConnectedGroupBuildResult):
            statistics["connected_layer_count"] = len(composition.layers)

        return PreparedA1MultiObject(
            settings=settings,
            sources=sources,
            objects=resolved_objects,
            document=document,
            composition=composition,
            texture_output_paths=texture_paths,
            warnings=tuple(warnings),
            statistics=MappingProxyType(dict(statistics)),
        )
    except A1MultiObjectPreparationError:
        raise
    except Exception as exc:
        raise A1MultiObjectPreparationError(
            stage=stage,
            cause=exc,
            statistics=statistics,
            warnings=tuple(warnings),
            component_id=current_component,
        ) from exc


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
        "A1 multi-object export failed at %s (component=%s, object=%s)",
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
    """Export one standalone or connected multi-object A1 transaction."""

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

    stage = A1MultiObjectStage.SERIALIZE_DOCUMENT
    statistics = dict(prepared.statistics)
    try:
        json_text = SpineSerializer().to_json(
            prepared.document,
            indent=settings.json_indent,
        )

        stage = A1MultiObjectStage.STAGE_OUTPUTS
        with atomic_file_transaction() as output_transaction:
            json_reservation = output_transaction.reserve(prepared.json_path)
            write_staged_utf8_text(
                json_reservation.staged_path,
                json_text,
                ensure_trailing_newline=True,
            )
            all_bake_reservations = []
            for item in prepared.objects:
                all_bake_reservations.extend(
                    stage_bake_plan_outputs(
                        item.source_object,
                        item.bake_target_snapshot,
                        item.bake_plan,
                        output_transaction,
                        item.settings.bake_execution,
                        context=context,
                        scene=scene,
                    )
                )

            stage = A1MultiObjectStage.COMMIT_OUTPUTS
            committed_paths = output_transaction.commit()

        expected_paths = (
            json_reservation.final_path,
            *(reservation.final_path for reservation in all_bake_reservations),
        )
        if tuple(committed_paths) != expected_paths:
            raise AtomicFileCommitError(
                "Committed output order does not match final JSON and texture "
                "reservations"
            )
        statistics["output_file_count"] = len(committed_paths)
        logger.info(
            "A1 multi-object export completed (%s): %s",
            settings.mode.value,
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
