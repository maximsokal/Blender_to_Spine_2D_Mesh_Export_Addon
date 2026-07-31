"""Build immutable single/multi export plans from one captured Blender UI state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

from ..application import (
    A1MultiObjectExportSettings,
    A1MultiObjectMode,
    A1SingleObjectExportSettings,
    ExportIssue,
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
    _ordered_selected_meshes,
)
from .a1_ui_settings import _build_sources_from_profiles, _settings_from_profiles


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


def _capture_selected_profiles(
    ordered_objects: Tuple[Any, ...],
) -> Tuple[_ObjectExportProfile, ...]:
    """Capture selected objects for the public standalone multi-export route.

    ``spine2d_connect_settings`` is retained in RNA only so existing ``.blend`` files
    continue to load safely. Connected and mixed composition are developer-only while
    Spine target-version support is being finalized, therefore persisted per-object
    Connect values must never affect the public Export Selected Objects operator.
    """

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
            connect_enabled=False,
        )
        for obj in ordered_objects
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
    """Build a public multi-object plan that is always standalone.

    Connected/mixed composition remains available only through explicit internal APIs
    and acceptance workers. The normal UI operator must not infer a hidden composition
    mode from persistent object data.
    """

    if context is None:
        raise ValueError("context cannot be None")
    scene = getattr(context, "scene", None)
    if scene is None:
        raise ValueError("context.scene is missing")

    ordered_objects = _ordered_selected_meshes(context)
    scene_profile: _SceneExportProfile = _capture_scene_profile(scene)
    object_profiles = _capture_selected_profiles(ordered_objects)
    sources = _build_sources_from_profiles(object_profiles, scene_profile)

    base_name = sanitize_filename_stem(object_profiles[0].object_name)
    version_token = spine_json_version_filename_token(
        sources[0].settings.export.spine_target
    )
    output_stem = (
        f"{base_name}_plus_{len(object_profiles) - 1}_objects_{version_token}"
    )
    settings = A1MultiObjectExportSettings(
        output_directory=scene_profile.output_directory,
        output_stem=output_stem,
        mode=A1MultiObjectMode.STANDALONE,
        anchor_component_id=None,
    )
    return A1UiMultiExportPlan(
        connected_sources=(),
        standalone_sources=sources,
        settings=settings,
        issues=(),
    )


__all__ = [
    "A1UiMultiExportPlan",
    "A1UiSingleExportPlan",
    "build_active_ui_export_plan",
    "build_selected_ui_export_plan",
]
