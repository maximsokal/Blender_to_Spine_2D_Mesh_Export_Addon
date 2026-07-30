"""Build immutable single/multi export plans from one captured Blender UI state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Tuple

from ..application import (
    A1MultiObjectExportSettings,
    A1MultiObjectMode,
    A1MultiObjectStage,
    A1SingleObjectExportSettings,
    ExportIssue,
    IssueSeverity,
)
from ..domain.baking import sanitize_filename_stem
from ..domain.spine.rig_profiles import A1RigSetupPoseMode
from ..domain.spine.version_target import spine_json_version_filename_token
from .a1_multi_object_contracts import A1MultiObjectSource
from .a1_ui_scene_capture import _SceneExportProfile, _capture_scene_profile
from .a1_ui_selection import (
    _ObjectExportProfile,
    _active_mesh,
    _capture_object_profile,
    _connect_enabled,
    _ordered_selected_meshes,
)
from .a1_ui_settings import _build_sources_from_profiles, _settings_from_profiles


_ProfileCapture = Callable[
    [Tuple[Any, ...]],
    Tuple[_ObjectExportProfile, ...],
]


@dataclass(frozen=True, slots=True)
class A1UiSingleExportPlan:
    source_object: Any
    settings: A1SingleObjectExportSettings

    def __post_init__(self) -> None:
        if self.source_object is None:
            raise ValueError("source_object cannot be None")
        if not isinstance(self.settings, A1SingleObjectExportSettings):
            raise TypeError("settings must be A1SingleObjectExportSettings")


@dataclass(frozen=True, slots=True)
class A1UiMultiExportPlan:
    connected_sources: Tuple[A1MultiObjectSource, ...]
    standalone_sources: Tuple[A1MultiObjectSource, ...]
    settings: A1MultiObjectExportSettings
    issues: Tuple[ExportIssue, ...] = ()

    def __post_init__(self) -> None:
        for field_name in ("connected_sources", "standalone_sources"):
            values = getattr(self, field_name)
            if not isinstance(values, tuple) or not all(
                isinstance(item, A1MultiObjectSource) for item in values
            ):
                raise TypeError(
                    f"{field_name} must be a tuple of A1MultiObjectSource values"
                )
        if not isinstance(self.settings, A1MultiObjectExportSettings):
            raise TypeError("settings must be A1MultiObjectExportSettings")
        if not isinstance(self.issues, tuple) or not all(
            isinstance(issue, ExportIssue) for issue in self.issues
        ):
            raise TypeError("issues must be a tuple of ExportIssue values")

        all_sources = self.all_sources
        if len(all_sources) < 2:
            raise ValueError("multi-object export plan requires at least two sources")
        component_ids = tuple(source.component_id for source in all_sources)
        if len(component_ids) != len(set(component_ids)):
            raise ValueError("multi-object export plan component IDs must be unique")

        mode = self.settings.mode
        if mode is A1MultiObjectMode.CONNECTED:
            if len(self.connected_sources) < 2 or self.standalone_sources:
                raise ValueError(
                    "CONNECTED UI plan requires at least two connected sources only"
                )
        elif mode is A1MultiObjectMode.STANDALONE:
            if self.connected_sources or len(self.standalone_sources) < 2:
                raise ValueError(
                    "STANDALONE UI plan requires at least two standalone sources only"
                )
        elif mode is A1MultiObjectMode.MIXED:
            if len(self.connected_sources) < 2 or not self.standalone_sources:
                raise ValueError(
                    "MIXED UI plan requires connected and standalone subgroups"
                )
        else:
            raise TypeError(f"Unsupported multi-object mode: {mode!r}")

    @property
    def all_sources(self) -> Tuple[A1MultiObjectSource, ...]:
        if self.settings.mode is A1MultiObjectMode.MIXED:
            return self.connected_sources + self.standalone_sources
        return self.connected_sources or self.standalone_sources


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
            "requires at least two objects, so all selected objects will be "
            "exported standalone."
        ),
        object_id=profile.object_name,
        context={
            "selected_object_count": selected_count,
            "connected_object_count": 1,
            "fallback_mode": A1MultiObjectMode.STANDALONE.value,
        },
    )


def _capture_profiles_with_connect_resolver(
    ordered_objects: Tuple[Any, ...],
    connect_resolver: Callable[[Any], bool],
) -> Tuple[_ObjectExportProfile, ...]:
    """Capture selected objects with an explicit connected-state resolver."""

    if not isinstance(ordered_objects, tuple) or len(ordered_objects) < 2:
        raise ValueError("ordered_objects must contain at least two objects")
    if not callable(connect_resolver):
        raise TypeError("connect_resolver must be callable")

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
            connect_enabled=bool(connect_resolver(obj)),
        )
        for obj in ordered_objects
    )


def _capture_selected_profiles(
    ordered_objects: Tuple[Any, ...],
) -> Tuple[_ObjectExportProfile, ...]:
    """Capture production profiles while ignoring hidden persisted Connect flags."""

    return _capture_profiles_with_connect_resolver(
        ordered_objects,
        lambda _obj: False,
    )


def _capture_development_selected_profiles(
    ordered_objects: Tuple[Any, ...],
) -> Tuple[_ObjectExportProfile, ...]:
    """Capture the explicit development-only connected/mixed profile set."""

    return _capture_profiles_with_connect_resolver(
        ordered_objects,
        _connect_enabled,
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


def _capture_multi_plan_inputs(
    context: Any,
    profile_capture: _ProfileCapture,
) -> tuple[
    _SceneExportProfile,
    Tuple[_ObjectExportProfile, ...],
    Tuple[A1MultiObjectSource, ...],
]:
    if context is None:
        raise ValueError("context cannot be None")
    if not callable(profile_capture):
        raise TypeError("profile_capture must be callable")
    scene = getattr(context, "scene", None)
    if scene is None:
        raise ValueError("context.scene is missing")

    ordered_objects = _ordered_selected_meshes(context)
    scene_profile = _capture_scene_profile(scene)
    object_profiles = profile_capture(ordered_objects)
    sources = _build_sources_from_profiles(object_profiles, scene_profile)
    return scene_profile, object_profiles, sources


def _resolve_development_partition(
    sources: Tuple[A1MultiObjectSource, ...],
    profiles: Tuple[_ObjectExportProfile, ...],
) -> tuple[
    Tuple[A1MultiObjectSource, ...],
    Tuple[A1MultiObjectSource, ...],
    Tuple[ExportIssue, ...],
]:
    connected, standalone = _partition_sources(sources, profiles)
    if len(connected) != 1:
        return connected, standalone, ()

    fallback_profile = next(
        profile for profile in profiles if profile.connect_enabled
    )
    issue = _single_connect_fallback_issue(
        fallback_profile,
        selected_count=len(profiles),
    )
    return (), sources, (issue,)


def _resolve_multi_mode(
    connected: Tuple[A1MultiObjectSource, ...],
    standalone: Tuple[A1MultiObjectSource, ...],
) -> A1MultiObjectMode:
    if connected and standalone:
        return A1MultiObjectMode.MIXED
    if connected:
        return A1MultiObjectMode.CONNECTED
    return A1MultiObjectMode.STANDALONE


def _build_multi_plan(
    scene_profile: _SceneExportProfile,
    profiles: Tuple[_ObjectExportProfile, ...],
    sources: Tuple[A1MultiObjectSource, ...],
    *,
    connected: Tuple[A1MultiObjectSource, ...],
    standalone: Tuple[A1MultiObjectSource, ...],
    issues: Tuple[ExportIssue, ...] = (),
) -> A1UiMultiExportPlan:
    base_name = sanitize_filename_stem(profiles[0].object_name)
    version_token = spine_json_version_filename_token(
        sources[0].settings.export.spine_target
    )
    output_stem = (
        f"{base_name}_plus_{len(profiles) - 1}_objects_{version_token}"
    )
    settings = A1MultiObjectExportSettings(
        output_directory=scene_profile.output_directory,
        output_stem=output_stem,
        mode=_resolve_multi_mode(connected, standalone),
        anchor_component_id=(connected[0].component_id if connected else None),
    )
    return A1UiMultiExportPlan(
        connected_sources=connected,
        standalone_sources=standalone,
        settings=settings,
        issues=issues,
    )


def build_active_ui_export_plan(context: Any) -> A1UiSingleExportPlan:
    """Build one active-object plan with a neutral single-object rig setup pose."""

    if context is None:
        raise ValueError("context cannot be None")
    scene = getattr(context, "scene", None)
    if scene is None:
        raise ValueError("context.scene is missing")

    source_object = _active_mesh(context)
    scene_profile = _capture_scene_profile(scene)
    object_profile = _capture_object_profile(
        source_object,
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
        rig_setup_pose_mode=A1RigSetupPoseMode.NORMALIZED_SINGLE,
    )
    return A1UiSingleExportPlan(source_object=source_object, settings=settings)


def build_selected_ui_export_plan(context: Any) -> A1UiMultiExportPlan:
    """Build the production selected-object plan as standalone unconditionally."""

    scene_profile, profiles, sources = _capture_multi_plan_inputs(
        context,
        _capture_selected_profiles,
    )
    return _build_multi_plan(
        scene_profile,
        profiles,
        sources,
        connected=(),
        standalone=sources,
    )


def build_development_connected_ui_export_plan(
    context: Any,
) -> A1UiMultiExportPlan:
    """Build the explicit development-only connected/mixed selected-object plan."""

    scene_profile, profiles, sources = _capture_multi_plan_inputs(
        context,
        _capture_development_selected_profiles,
    )
    connected, standalone, issues = _resolve_development_partition(
        sources,
        profiles,
    )
    return _build_multi_plan(
        scene_profile,
        profiles,
        sources,
        connected=connected,
        standalone=standalone,
        issues=issues,
    )


__all__ = [
    "A1UiMultiExportPlan",
    "A1UiSingleExportPlan",
    "build_active_ui_export_plan",
    "build_development_connected_ui_export_plan",
    "build_selected_ui_export_plan",
]
