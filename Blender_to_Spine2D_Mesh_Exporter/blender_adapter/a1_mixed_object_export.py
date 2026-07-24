"""Prepare mixed connected/standalone A1 sources without producing output."""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Tuple

from ..application import (
    A1ExportProgressCallback,
    A1MultiObjectExportSettings,
    A1MultiObjectMode,
    A1MultiObjectStage,
    A1OutputPreflightSource,
    ExportIssue,
    emit_a1_export_progress,
    preflight_a1_output_namespace,
    scale_a1_export_progress_callback,
)
from ..domain.baking import windows_path_identity
from .a1_mixed_settings import build_connected_subgroup_settings
from .a1_multi_object_contracts import (
    A1MultiObjectPreparationError,
    A1MultiObjectSource,
    PreparedA1MultiObject,
    record_object_statistics,
)
from .a1_multi_object_export import _source_object_name, prepare_a1_multi_object
from .a1_object_preparation import (
    A1ObjectPreparationError,
    PreparedA1Object,
    StatisticsValue,
    prepare_a1_object,
)


def _preflight_mixed_sources(
    sources: Tuple[A1MultiObjectSource, ...],
    settings: A1MultiObjectExportSettings,
) -> Tuple[Path, ...]:
    result = preflight_a1_output_namespace(
        output_root=settings.output_directory,
        json_path=settings.json_path,
        sources=tuple(
            A1OutputPreflightSource(
                owner=source.component_id,
                object_name=_source_object_name(source),
                settings=source.settings,
            )
            for source in sources
        ),
    )
    return result.texture_paths


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
        raise ValueError("anchor_component_id must identify an object in connected_sources")

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


def _prepare_standalone_objects(
    sources: Tuple[A1MultiObjectSource, ...],
    *,
    context: Any | None,
    scene: Any | None,
    progress_callback: A1ExportProgressCallback | None = None,
    progress_start: float = 0.0,
    progress_end: float = 100.0,
    object_index_offset: int = 0,
    total_object_count: int | None = None,
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
    object_count = len(sources)
    overall_count = total_object_count or object_count
    span = float(progress_end) - float(progress_start)
    for local_index, source in enumerate(sources):
        start = float(progress_start) + span * local_index / object_count
        end = float(progress_start) + span * (local_index + 1) / object_count
        global_index = object_index_offset + local_index + 1
        object_progress = scale_a1_export_progress_callback(
            progress_callback,
            start_percent=start,
            end_percent=end,
            object_id=source.component_id,
            object_index=global_index,
            object_count=overall_count,
            message_prefix=f"[{global_index}/{overall_count}] ",
        )
        try:
            prepared = prepare_a1_object(
                source.source_object,
                source.settings,
                context=context,
                scene=scene,
                progress_callback=object_progress,
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
        record_object_statistics(statistics, source.component_id, prepared.statistics)
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
    owner_by_identity: dict[Tuple[str, ...], tuple[str, Path]] = {
        windows_path_identity(resolved_json): ("final mixed JSON", resolved_json)
    }
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
            identity = windows_path_identity(resolved)
            previous = owner_by_identity.get(identity)
            if previous is not None:
                previous_owner, previous_path = previous
                raise ValueError(
                    "Windows mixed output path collision between "
                    f"{previous_owner} ({previous_path}) and {group_name} texture "
                    f"({resolved})"
                )
            owner_by_identity[identity] = (f"{group_name} texture", resolved)
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
    progress_callback: A1ExportProgressCallback | None = None,
) -> PreparedA1MultiObject:
    """Prepare both mixed subgroups and validate one shared output namespace."""

    emit_a1_export_progress(
        progress_callback,
        percent=0,
        stage=A1MultiObjectStage.VALIDATE_REQUEST,
        message="Validating mixed export request",
    )
    _validate_mixed_sources(connected_sources, standalone_sources, settings)
    all_sources = connected_sources + standalone_sources
    emit_a1_export_progress(
        progress_callback,
        percent=5,
        stage=A1MultiObjectStage.VALIDATE_REQUEST,
        message="Checking mixed output paths",
    )
    predicted_texture_paths = _preflight_mixed_sources(all_sources, settings)
    anchor = settings.anchor_component_id or connected_sources[0].component_id

    total_count = len(all_sources)
    preparation_start = 10.0
    preparation_end = 90.0
    preparation_span = preparation_end - preparation_start
    connected_end = (
        preparation_start
        + preparation_span * len(connected_sources) / total_count
    )
    connected_progress = scale_a1_export_progress_callback(
        progress_callback,
        start_percent=preparation_start,
        end_percent=connected_end,
        message_prefix="Connected group: ",
    )
    connected = prepare_a1_multi_object(
        connected_sources,
        build_connected_subgroup_settings(settings, anchor),
        context=context,
        scene=scene,
        progress_callback=connected_progress,
    )
    standalone_objects, standalone_warnings, _ = _prepare_standalone_objects(
        standalone_sources,
        context=context,
        scene=scene,
        progress_callback=progress_callback,
        progress_start=connected_end,
        progress_end=preparation_end,
        object_index_offset=len(connected_sources),
        total_object_count=total_count,
    )
    emit_a1_export_progress(
        progress_callback,
        percent=95,
        stage=A1MultiObjectStage.VALIDATE_OUTPUTS,
        message="Validating mixed realized output paths",
    )
    texture_paths = _validate_final_paths(
        settings,
        connected.texture_output_paths,
        _texture_paths(standalone_objects),
    )
    statistics: dict[str, StatisticsValue] = {
        "object_count": len(connected_sources) + len(standalone_sources),
        "connected_object_count": len(connected_sources),
        "standalone_object_count": len(standalone_sources),
        "mode": A1MultiObjectMode.MIXED.value,
        "predicted_texture_output_count": len(predicted_texture_paths),
        "texture_output_count": len(texture_paths),
    }
    result = PreparedA1MultiObject(
        settings=settings,
        sources=connected.sources + standalone_sources,
        objects=connected.objects + standalone_objects,
        texture_output_paths=texture_paths,
        warnings=connected.warnings + standalone_warnings,
        statistics=MappingProxyType(dict(statistics)),
    )
    emit_a1_export_progress(
        progress_callback,
        percent=100,
        stage=A1MultiObjectStage.VALIDATE_OUTPUTS,
        message="Mixed object preparation complete",
    )
    return result


__all__ = ["build_connected_subgroup_settings", "prepare_a1_mixed_object"]
