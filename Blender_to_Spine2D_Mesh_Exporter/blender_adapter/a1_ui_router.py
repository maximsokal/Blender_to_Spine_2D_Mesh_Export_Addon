"""Route captured Blender UI requests to typed A1 output services."""

from __future__ import annotations

from dataclasses import replace
import logging
from typing import Any, Tuple

from ..application import (
    A1MultiObjectExportSettings,
    A1MultiObjectMode,
    A1MultiObjectStage,
    ExportIssue,
    ExportResult,
    IssueSeverity,
)
from ..domain.baking import sanitize_filename_stem
from .a1_mixed_object_output import export_a1_mixed_object
from .a1_multi_object_contracts import A1MultiObjectSource
from .a1_multi_object_output import export_a1_multi_object
from .a1_single_object_export import export_a1_single_object
from .a1_ui_scene_capture import _capture_scene_profile
from .a1_ui_selection import (
    _ObjectExportProfile,
    _active_mesh,
    _capture_object_profile,
    _connect_enabled,
    _ordered_selected_meshes,
)
from .a1_ui_settings import _build_sources_from_profiles, _settings_from_profiles


logger = logging.getLogger(__name__)


def _append_issue(
    result: ExportResult,
    issue: ExportIssue,
    *,
    statistics: dict[str, int | float | str] | None = None,
) -> ExportResult:
    if not isinstance(result, ExportResult):
        raise TypeError("result must be ExportResult")
    if not isinstance(issue, ExportIssue):
        raise TypeError("issue must be ExportIssue")
    resolved_statistics = dict(result.statistics)
    if statistics is not None:
        if not isinstance(statistics, dict):
            raise TypeError("statistics must be dict or None")
        resolved_statistics.update(statistics)
    return replace(
        result,
        issues=(issue, *result.issues),
        statistics=resolved_statistics,
    )


def _single_connect_fallback_issue(
    profile: _ObjectExportProfile,
    *,
    selected_count: int,
) -> ExportIssue:
    if not isinstance(profile, _ObjectExportProfile):
        raise TypeError("profile must be _ObjectExportProfile")
    if not isinstance(selected_count, int) or selected_count < 2:
        raise ValueError("selected_count must be at least two")
    return ExportIssue(
        severity=IssueSeverity.WARNING,
        stage=A1MultiObjectStage.VALIDATE_REQUEST.value,
        code="A1_SINGLE_CONNECT_FALLBACK",
        message=(
            "Exactly one selected object has Connect enabled. Connected export "
            "requires at least two objects, so all selected objects were "
            "exported standalone."
        ),
        object_id=profile.object_name,
        context={
            "selected_object_count": selected_count,
            "connected_object_count": 1,
            "fallback_mode": A1MultiObjectMode.STANDALONE.value,
        },
    )


def _capture_selected_profiles(
    ordered_objects: Tuple[Any, ...],
) -> Tuple[_ObjectExportProfile, ...]:
    if not isinstance(ordered_objects, tuple) or len(ordered_objects) < 2:
        raise ValueError("ordered_objects must contain at least two objects")
    return tuple(
        _capture_object_profile(
            obj,
            sequence_start_frame=int(
                getattr(
                    getattr(obj, "spine2d_bake_settings", None),
                    "bake_frame_start",
                    0,
                )
            ),
            sequence_frame_count=int(
                getattr(
                    getattr(obj, "spine2d_bake_settings", None),
                    "frames_for_render",
                    0,
                )
            ),
            connect_enabled=_connect_enabled(obj),
        )
        for obj in ordered_objects
    )


def _partition_sources(
    sources: Tuple[A1MultiObjectSource, ...],
    profiles: Tuple[_ObjectExportProfile, ...],
) -> tuple[Tuple[A1MultiObjectSource, ...], Tuple[A1MultiObjectSource, ...]]:
    if len(sources) != len(profiles):
        raise ValueError("sources and profiles must correspond one-to-one")
    connected = tuple(
        source
        for source, profile in zip(sources, profiles, strict=True)
        if profile.connect_enabled
    )
    standalone = tuple(
        source
        for source, profile in zip(sources, profiles, strict=True)
        if not profile.connect_enabled
    )
    return connected, standalone


def export_active_object_a1(context: Any) -> ExportResult:
    """Export the active Mesh through the complete single-object A1 output service."""

    if context is None:
        raise ValueError("context cannot be None")
    scene = getattr(context, "scene", None)
    if scene is None:
        raise ValueError("context.scene is missing")
    obj = _active_mesh(context)
    scene_profile = _capture_scene_profile(scene)
    object_profile = _capture_object_profile(
        obj,
        sequence_start_frame=int(
            getattr(scene, "spine2d_bake_frame_start", 0)
        ),
        sequence_frame_count=int(
            getattr(scene, "spine2d_frames_for_render", 0)
        ),
        connect_enabled=False,
    )
    settings = _settings_from_profiles(
        object_profile,
        scene_profile,
        json_output_stem=(
            f"{sanitize_filename_stem(object_profile.object_name)}_merged"
        ),
    )
    return export_a1_single_object(
        obj,
        settings,
        context=context,
        scene=scene,
    )


def export_selected_objects_a1(context: Any) -> ExportResult:
    """Export selected meshes through standalone, connected, or mixed A1 output."""

    if context is None:
        raise ValueError("context cannot be None")
    scene = getattr(context, "scene", None)
    if scene is None:
        raise ValueError("context.scene is missing")

    ordered_objects = _ordered_selected_meshes(context)
    scene_profile = _capture_scene_profile(scene)
    object_profiles = _capture_selected_profiles(ordered_objects)
    sources = _build_sources_from_profiles(object_profiles, scene_profile)
    connected, standalone = _partition_sources(sources, object_profiles)

    fallback_profile = None
    if len(connected) == 1:
        fallback_profile = next(
            profile for profile in object_profiles if profile.connect_enabled
        )
        logger.warning(
            "One selected object has Connect enabled; connected export requires "
            "at least two objects, so all selected objects will be exported "
            "standalone"
        )
        connected = ()
        standalone = sources

    base_name = sanitize_filename_stem(object_profiles[0].object_name)
    output_stem = f"{base_name}_plus_{len(object_profiles) - 1}_objects"
    if connected and standalone:
        settings = A1MultiObjectExportSettings(
            output_directory=scene_profile.output_directory,
            output_stem=output_stem,
            mode=A1MultiObjectMode.MIXED,
            anchor_component_id=connected[0].component_id,
        )
        result = export_a1_mixed_object(
            connected,
            standalone,
            settings,
            context=context,
            scene=scene,
        )
    else:
        mode = (
            A1MultiObjectMode.CONNECTED
            if connected
            else A1MultiObjectMode.STANDALONE
        )
        settings = A1MultiObjectExportSettings(
            output_directory=scene_profile.output_directory,
            output_stem=output_stem,
            mode=mode,
            anchor_component_id=(
                connected[0].component_id if connected else None
            ),
        )
        result = export_a1_multi_object(
            connected or standalone,
            settings,
            context=context,
            scene=scene,
        )

    if fallback_profile is None:
        return result
    return _append_issue(
        result,
        _single_connect_fallback_issue(
            fallback_profile,
            selected_count=len(object_profiles),
        ),
        statistics={"single_connect_fallback_count": 1},
    )


__all__ = ["export_active_object_a1", "export_selected_objects_a1"]
