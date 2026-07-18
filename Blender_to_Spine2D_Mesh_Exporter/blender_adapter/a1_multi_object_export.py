"""Prepare several Blender objects for one A1 output transaction.

This module is preparation-only. It validates sources and output paths, prepares every
object in memory, and returns immutable preparation data. Rendering, document composition,
serialization, and file commits belong to ``a1_multi_object_output``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Tuple

from ..application import (
    A1MultiObjectExportSettings,
    A1MultiObjectMode,
    A1MultiObjectStage,
    A1SingleObjectExportSettings,
    ExportIssue,
)
from .a1_object_preparation import (
    A1ObjectPreparationError,
    PreparedA1Object,
    StatisticsValue,
    prepare_a1_object,
)


@dataclass(frozen=True, slots=True)
class A1MultiObjectSource:
    """One live Blender source plus its immutable per-object export settings."""

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
    """Preparation result with no draft/final document ownership."""

    settings: A1MultiObjectExportSettings
    sources: Tuple[A1MultiObjectSource, ...]
    objects: Tuple[PreparedA1Object, ...]
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
        if not isinstance(warnings, tuple) or not all(
            isinstance(issue, ExportIssue) for issue in warnings
        ):
            raise TypeError("warnings must be a tuple of ExportIssue values")
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
    if not isinstance(settings, A1MultiObjectExportSettings):
        raise TypeError("settings must be A1MultiObjectExportSettings")
    if settings.mode is A1MultiObjectMode.MIXED:
        raise ValueError(
            "prepare_a1_multi_object does not accept MIXED mode; "
            "use prepare_a1_mixed_object"
        )
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
        # Connected placement is applied exactly once by the global typed rig.
        return replace(
            source.settings,
            use_world_location_for_main_bone=False,
        )
    return source.settings


def record_object_statistics(
    target: dict[str, StatisticsValue],
    component_id: str,
    values: Mapping[str, StatisticsValue],
) -> None:
    """Merge object statistics under one deterministic component namespace."""

    if not isinstance(target, dict):
        raise TypeError("target must be dict")
    if not isinstance(component_id, str) or not component_id.strip():
        raise ValueError("component_id must be a non-empty string")
    if not isinstance(values, Mapping):
        raise TypeError("values must be a mapping")
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


def prepare_a1_multi_object(
    sources: Tuple[A1MultiObjectSource, ...],
    settings: A1MultiObjectExportSettings,
    *,
    context: Any | None = None,
    scene: Any | None = None,
) -> PreparedA1MultiObject:
    """Prepare all objects and validate output ownership without composing a document."""

    stage = A1MultiObjectStage.VALIDATE_REQUEST
    statistics: dict[str, StatisticsValue] = {}
    warnings: list[ExportIssue] = []
    current_component: str | None = None

    try:
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
            record_object_statistics(
                statistics,
                source.component_id,
                prepared.statistics,
            )
        resolved_objects = tuple(prepared_objects)

        stage = A1MultiObjectStage.VALIDATE_OUTPUTS
        texture_paths = _validate_prepared_outputs(resolved_objects, settings)
        statistics["texture_output_count"] = len(texture_paths)

        return PreparedA1MultiObject(
            settings=settings,
            sources=sources,
            objects=resolved_objects,
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


__all__ = [
    "A1MultiObjectPreparationError",
    "A1MultiObjectSource",
    "PreparedA1MultiObject",
    "prepare_a1_multi_object",
    "record_object_statistics",
]
