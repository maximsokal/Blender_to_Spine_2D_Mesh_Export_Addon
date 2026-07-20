"""Prepare several Blender objects for one A1 output transaction.

This module owns preparation orchestration only. Shared source/result/error contracts live in
``a1_multi_object_contracts`` so UI, composition, and output modules do not depend on a
preparation implementation merely to import dataclasses.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Tuple

from ..application import (
    A1MultiObjectExportSettings,
    A1MultiObjectMode,
    A1MultiObjectStage,
    A1OutputPreflightSource,
    A1SingleObjectExportSettings,
    ExportIssue,
    preflight_a1_output_namespace,
)
from ..domain.baking import windows_path_identity
from .a1_multi_object_contracts import (
    A1MultiObjectPreparationError,
    A1MultiObjectSource,
    PreparedA1MultiObject,
    record_object_statistics,
)
from .a1_object_preparation import (
    A1ObjectPreparationError,
    PreparedA1Object,
    StatisticsValue,
    prepare_a1_object,
)


def _source_object_name(source: A1MultiObjectSource) -> str:
    """Resolve the object name required for pure output prediction."""

    if not isinstance(source, A1MultiObjectSource):
        raise TypeError("source must be A1MultiObjectSource")
    value = str(
        getattr(source.source_object, "name_full", None)
        or getattr(source.source_object, "name", None)
        or ""
    ).strip()
    if not value:
        raise ValueError(
            f"Component '{source.component_id}' has an empty source object name"
        )
    return value


def _preflight_sources(
    sources: Tuple[A1MultiObjectSource, ...],
    settings: A1MultiObjectExportSettings,
) -> Tuple[Path, ...]:
    """Validate the complete Windows output namespace before geometry preparation."""

    result = preflight_a1_output_namespace(
        output_root=settings.output_directory,
        json_path=settings.json_path,
        sources=tuple(
            A1OutputPreflightSource(
                owner=source.component_id,
                object_name=_source_object_name(source),
                settings=_settings_for_preparation(source, settings.mode),
            )
            for source in sources
        ),
    )
    return result.texture_paths


def _validate_sources(
    sources: Tuple[A1MultiObjectSource, ...],
    settings: A1MultiObjectExportSettings,
) -> Tuple[Path, ...]:
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
    return _preflight_sources(sources, settings)


def _settings_for_preparation(
    source: A1MultiObjectSource,
    mode: A1MultiObjectMode,
) -> A1SingleObjectExportSettings:
    if not isinstance(source, A1MultiObjectSource):
        raise TypeError("source must be A1MultiObjectSource")
    if not isinstance(mode, A1MultiObjectMode):
        raise TypeError("mode must be A1MultiObjectMode")
    if mode is A1MultiObjectMode.CONNECTED:
        return replace(source.settings, use_world_location_for_main_bone=False)
    return source.settings


def _validate_prepared_outputs(
    prepared: Tuple[PreparedA1Object, ...],
    settings: A1MultiObjectExportSettings,
) -> Tuple[Path, ...]:
    if not isinstance(prepared, tuple) or not prepared:
        raise ValueError("prepared must be a non-empty tuple")
    if not all(isinstance(item, PreparedA1Object) for item in prepared):
        raise TypeError("prepared must contain PreparedA1Object values")
    prefixes = tuple(item.prefix for item in prepared)
    if len(prefixes) != len(set(prefixes)):
        raise ValueError("prepared rig prefixes must be unique")

    output_root = settings.output_directory.expanduser().resolve(strict=False)
    final_json = settings.json_path.expanduser().resolve(strict=False)
    paths: list[Path] = []
    owner_by_identity: dict[Tuple[str, ...], tuple[str, Path]] = {
        windows_path_identity(final_json): ("final JSON", final_json)
    }
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
            identity = windows_path_identity(path)
            previous = owner_by_identity.get(identity)
            if previous is not None:
                previous_owner, previous_path = previous
                raise ValueError(
                    f"Windows output path collision between {previous_owner} "
                    f"({previous_path}) and component '{item.object_id}' ({path})"
                )
            owner_by_identity[identity] = (item.object_id, path)
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
        predicted_texture_paths = _validate_sources(sources, settings)
        statistics.update(
            {
                "object_count": len(sources),
                "mode": settings.mode.value,
                "output_stem": settings.resolved_output_stem,
                "predicted_texture_output_count": len(predicted_texture_paths),
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
