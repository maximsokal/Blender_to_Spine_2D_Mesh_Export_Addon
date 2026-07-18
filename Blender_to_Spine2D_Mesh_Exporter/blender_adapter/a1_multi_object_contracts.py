"""Typed contracts shared by A1 multi-object preparation, composition, and output."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Tuple

from ..application import (
    A1MultiObjectExportSettings,
    A1MultiObjectStage,
    A1SingleObjectExportSettings,
    ExportIssue,
)
from .a1_object_preparation import PreparedA1Object, StatisticsValue


@dataclass(frozen=True, slots=True)
class A1MultiObjectSource:
    """One live Blender source plus immutable per-object export settings."""

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
            raise ValueError("animation_namespace must be a non-empty string or None")


@dataclass(frozen=True, slots=True)
class PreparedA1MultiObject:
    """Preparation result with no draft or final document ownership."""

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
        object_ids = tuple(item.object_id for item in self.objects)
        if len(object_ids) != len(set(object_ids)):
            raise ValueError("prepared object IDs must be unique")

    @property
    def json_path(self) -> Path:
        return self.settings.json_path


class A1MultiObjectPreparationError(RuntimeError):
    """Capture one failed multi-object stage and its optional object substage."""

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
        if not isinstance(statistics, Mapping):
            raise TypeError("statistics must be a mapping")
        if not isinstance(warnings, tuple) or not all(
            isinstance(issue, ExportIssue) for issue in warnings
        ):
            raise TypeError("warnings must be a tuple of ExportIssue values")
        for field_name, value in (
            ("component_id", component_id),
            ("object_id", object_id),
            ("object_stage", object_stage),
        ):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{field_name} must be a non-empty string or None")

        self.stage = stage
        self.cause = cause
        self.statistics = MappingProxyType(dict(statistics))
        self.warnings = warnings
        self.component_id = component_id
        self.object_id = object_id
        self.object_stage = object_stage

        details = ""
        if component_id is not None:
            details += f" component='{component_id}'"
        if object_stage is not None:
            details += f" object_stage='{object_stage}'"
        message = str(cause) or type(cause).__name__
        super().__init__(
            f"A1 multi-object preparation failed at {stage.value}{details}: {message}"
        )


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
    prefix = f"component.{component_id}."
    keys = tuple(values.keys())
    if not all(isinstance(key, str) and key for key in keys):
        raise ValueError("statistics keys must be non-empty strings")
    for key, value in values.items():
        # Post-render finalization intentionally replaces preparation values under the
        # same component namespace (for example projection crop statistics).
        target[prefix + key] = value


__all__ = [
    "A1MultiObjectPreparationError",
    "A1MultiObjectSource",
    "PreparedA1MultiObject",
    "record_object_statistics",
]
