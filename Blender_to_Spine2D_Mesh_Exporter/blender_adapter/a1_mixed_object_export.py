"""Prepare mixed connected/standalone A1 sources without producing output.

At least two connected objects form one future connected subgroup. Remaining selected
objects are prepared as standalone components. This module performs no document
composition, rendering, serialization, or file writes.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Tuple

from ..application import (
    A1MultiObjectExportSettings,
    A1MultiObjectMode,
    A1MultiObjectStage,
    ExportIssue,
)
from .a1_multi_object_export import (
    A1MultiObjectPreparationError,
    A1MultiObjectSource,
    PreparedA1MultiObject,
    prepare_a1_multi_object,
    record_object_statistics,
)
from .a1_object_preparation import (
    A1ObjectPreparationError,
    PreparedA1Object,
    StatisticsValue,
    prepare_a1_object,
)


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


def build_connected_subgroup_settings(
    settings: A1MultiObjectExportSettings,
    anchor_component_id: str,
) -> A1MultiObjectExportSettings:
    """Derive the internal connected settings used by mixed preparation/output."""

    if not isinstance(settings, A1MultiObjectExportSettings):
        raise TypeError("settings must be A1MultiObjectExportSettings")
    if settings.mode is not A1MultiObjectMode.MIXED:
        raise ValueError("connected subgroup settings require MIXED parent settings")
    if not isinstance(anchor_component_id, str) or not anchor_component_id.strip():
        raise ValueError("anchor_component_id must be a non-empty string")
    return replace(
        settings,
        mode=A1MultiObjectMode.CONNECTED,
        output_stem=f"{settings.resolved_output_stem}__connected",
        anchor_component_id=anchor_component_id,
    )


def _prepare_standalone_objects(
    sources: Tuple[A1MultiObjectSource, ...],
    *,
    context: Any | None,
    scene: Any | None,
) -> tuple[
    Tuple[PreparedA1Object, ...],
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
        record_object_statistics(
            statistics,
            source.component_id,
            prepared.statistics,
        )

    return (
        tuple(prepared_objects),
        tuple(warnings),
        MappingProxyType(dict(statistics)),
    )


def _texture_paths(objects: Tuple[PreparedA1Object, ...]) -> Tuple[Path, ...]:
    return tuple(
        task.output_path.expanduser().resolve(strict=False)
        for prepared in objects
        for task in prepared.bake_plan.frame_tasks
    )


def _validate_final_paths(
    settings: A1MultiObjectExportSettings,
    connected_paths: Tuple[Path, ...],
    standalone_paths: Tuple[Path, ...],
) -> Tuple[Path, ...]:
    output_root = settings.output_directory.expanduser().resolve(strict=False)
    resolved_json = settings.json_path.expanduser().resolve(strict=False)
    owner_by_path: dict[Path, str] = {resolved_json: "final mixed JSON"}
    result: list[Path] = []
    for group_name, paths in (
        ("connected", connected_paths),
        ("standalone", standalone_paths),
    ):
        for path in paths:
            resolved = path.expanduser().resolve(strict=False)
            try:
                resolved.relative_to(output_root)
            except ValueError as exc:
                raise ValueError(
                    f"Mixed {group_name} texture escapes output root: {resolved}"
                ) from exc
            previous = owner_by_path.get(resolved)
            if previous is not None:
                raise ValueError(
                    f"Mixed output path collision '{resolved}' between {previous} "
                    f"and {group_name} group"
                )
            owner_by_path[resolved] = f"{group_name} texture"
            result.append(resolved)
    if not result:
        raise ValueError("mixed export contains no texture output tasks")
    return tuple(result)


def prepare_a1_mixed_object(
    connected_sources: Tuple[A1MultiObjectSource, ...],
    standalone_sources: Tuple[A1MultiObjectSource, ...],
    settings: A1MultiObjectExportSettings,
    *,
    context: Any | None = None,
    scene: Any | None = None,
) -> PreparedA1MultiObject:
    """Prepare both mixed subgroups and validate one shared output namespace."""

    _validate_mixed_sources(connected_sources, standalone_sources, settings)
    anchor = settings.anchor_component_id or connected_sources[0].component_id
    connected = prepare_a1_multi_object(
        connected_sources,
        build_connected_subgroup_settings(settings, anchor),
        context=context,
        scene=scene,
    )
    (
        standalone_objects,
        standalone_warnings,
        standalone_statistics,
    ) = _prepare_standalone_objects(
        standalone_sources,
        context=context,
        scene=scene,
    )

    connected_paths = connected.texture_output_paths
    standalone_paths = _texture_paths(standalone_objects)
    texture_paths = _validate_final_paths(
        settings,
        connected_paths,
        standalone_paths,
    )

    statistics: dict[str, StatisticsValue] = {
        "object_count": len(connected_sources) + len(standalone_sources),
        "connected_object_count": len(connected_sources),
        "standalone_object_count": len(standalone_sources),
        "mode": A1MultiObjectMode.MIXED.value,
        "texture_output_count": len(texture_paths),
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
        texture_output_paths=texture_paths,
        warnings=connected.warnings + standalone_warnings,
        statistics=MappingProxyType(dict(statistics)),
    )


__all__ = [
    "build_connected_subgroup_settings",
    "prepare_a1_mixed_object",
]
