"""Build immutable A1 application settings from captured Blender UI profiles."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Tuple

from ..application import (
    A1SingleObjectExportSettings,
    A1SourceGeometryMode,
    ExportSettings,
)
from ..domain.baking import sanitize_filename_stem
from ..domain.spine.rig_profiles import A1RigSetupPoseMode
from ..domain.uv import UvUnwrapSettings
from .a1_multi_object_contracts import A1MultiObjectSource
from .a1_ui_scene_capture import (
    _SceneExportProfile,
    _capture_scene_profile,
)
from .a1_ui_selection import (
    _ObjectExportProfile,
    _capture_object_profile,
    _connect_enabled,
    _object_name,
)


_DEFAULT_BAKE_MARGIN = 4
_DEFAULT_UV_LAYER_NAME = "SpineBakeUV"


def _settings_from_profiles(
    obj: _ObjectExportProfile,
    scene: _SceneExportProfile,
    *,
    json_output_stem: str | None = None,
    rig_setup_pose_mode: A1RigSetupPoseMode = (
        A1RigSetupPoseMode.PRESERVE_COMPOSITION
    ),
) -> A1SingleObjectExportSettings:
    if not isinstance(obj, _ObjectExportProfile):
        raise TypeError("obj must be _ObjectExportProfile")
    if not isinstance(scene, _SceneExportProfile):
        raise TypeError("scene must be _SceneExportProfile")
    if not isinstance(rig_setup_pose_mode, A1RigSetupPoseMode):
        raise TypeError("rig_setup_pose_mode must be A1RigSetupPoseMode")
    return A1SingleObjectExportSettings(
        export=ExportSettings(
            texture_width=scene.texture_size,
            texture_height=scene.texture_size,
            output_directory=scene.output_directory,
            images_relative_path=scene.images_relative_path,
            rig_profile=scene.rig_profile.value,
            seam_mode=scene.seam_mode,
            angle_limit_degrees=scene.angle_limit_degrees,
            bake_margin=_DEFAULT_BAKE_MARGIN,
            sequence_start_frame=obj.sequence_start_frame,
            sequence_frame_count=obj.sequence_frame_count,
        ),
        prefix=obj.object_name,
        output_stem=sanitize_filename_stem(obj.object_name),
        json_output_stem=json_output_stem,
        source_geometry_mode=A1SourceGeometryMode.ORIGINAL,
        geometry=scene.geometry,
        uv=UvUnwrapSettings(layer_name=_DEFAULT_UV_LAYER_NAME),
        bake_execution=scene.bake_execution,
        include_control_icons=scene.include_control_icons,
        include_preview_animation=scene.include_preview_animation,
        material_source_policy=scene.material_source_policy,
        generated_material_pattern=scene.generated_material_pattern,
        generated_gray_color=scene.generated_gray_color,
        rig_setup_pose_mode=rig_setup_pose_mode,
    )


def _common_object_settings(
    obj: Any,
    scene: Any,
    *,
    output_directory: Path,
    texture_size: int,
    images_relative_path: str,
    sequence_start_frame: int,
    sequence_frame_count: int,
    json_output_stem: str | None = None,
    rig_setup_pose_mode: A1RigSetupPoseMode = (
        A1RigSetupPoseMode.PRESERVE_COMPOSITION
    ),
) -> A1SingleObjectExportSettings:
    """Compatibility helper retained for focused bridge tests and external callers."""

    scene_profile = _capture_scene_profile(
        scene,
        output_directory=output_directory,
        texture_size=texture_size,
        images_relative_path=images_relative_path,
    )
    object_profile = _capture_object_profile(
        obj,
        sequence_start_frame=sequence_start_frame,
        sequence_frame_count=sequence_frame_count,
        connect_enabled=_connect_enabled(obj),
    )
    return _settings_from_profiles(
        object_profile,
        scene_profile,
        json_output_stem=json_output_stem,
        rig_setup_pose_mode=rig_setup_pose_mode,
    )


def _build_multi_object_settings(
    obj: Any,
    scene: Any,
    *,
    output_directory: Path,
    texture_size: int,
    images_relative_path: str,
) -> A1SingleObjectExportSettings:
    bake = getattr(obj, "spine2d_bake_settings", None)
    return _common_object_settings(
        obj,
        scene,
        output_directory=output_directory,
        texture_size=texture_size,
        images_relative_path=images_relative_path,
        sequence_start_frame=int(getattr(bake, "bake_frame_start", 0)),
        sequence_frame_count=int(getattr(bake, "frames_for_render", 0)),
        rig_setup_pose_mode=A1RigSetupPoseMode.PRESERVE_COMPOSITION,
    )


def _build_single_object_settings(
    obj: Any,
    scene: Any,
    *,
    output_directory: Path,
    texture_size: int,
    images_relative_path: str,
) -> A1SingleObjectExportSettings:
    object_name = _object_name(obj)
    return _common_object_settings(
        obj,
        scene,
        output_directory=output_directory,
        texture_size=texture_size,
        images_relative_path=images_relative_path,
        sequence_start_frame=int(
            getattr(scene, "spine2d_bake_frame_start", 0)
        ),
        sequence_frame_count=int(
            getattr(scene, "spine2d_frames_for_render", 0)
        ),
        json_output_stem=f"{sanitize_filename_stem(object_name)}_merged",
        rig_setup_pose_mode=A1RigSetupPoseMode.NORMALIZED_SINGLE,
    )


def _build_sources_from_profiles(
    objects: Tuple[_ObjectExportProfile, ...],
    scene: _SceneExportProfile,
) -> Tuple[A1MultiObjectSource, ...]:
    if not isinstance(objects, tuple) or not objects:
        raise ValueError("objects must be a non-empty tuple")
    if not all(isinstance(item, _ObjectExportProfile) for item in objects):
        raise TypeError("objects must contain _ObjectExportProfile values")
    if not isinstance(scene, _SceneExportProfile):
        raise TypeError("scene must be _SceneExportProfile")
    return tuple(
        A1MultiObjectSource(
            source_object=obj.source_object,
            component_id=f"object_{index}:{obj.object_name}",
            animation_namespace=f"object_{index}",
            settings=_settings_from_profiles(
                obj,
                scene,
                rig_setup_pose_mode=A1RigSetupPoseMode.PRESERVE_COMPOSITION,
            ),
        )
        for index, obj in enumerate(objects, start=1)
    )


def _build_sources(
    objects: Tuple[Any, ...],
    scene: Any,
    *,
    output_directory: Path,
    texture_size: int,
    images_relative_path: str,
) -> Tuple[A1MultiObjectSource, ...]:
    """Compatibility helper that captures one Scene snapshot for every source."""

    if not isinstance(objects, tuple) or not objects:
        raise ValueError("objects must be a non-empty tuple")
    scene_profile = _capture_scene_profile(
        scene,
        output_directory=output_directory,
        texture_size=texture_size,
        images_relative_path=images_relative_path,
    )
    profiles = tuple(
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
        for obj in objects
    )
    return _build_sources_from_profiles(profiles, scene_profile)


__all__ = [
    "_build_multi_object_settings",
    "_build_single_object_settings",
    "_build_sources",
    "_build_sources_from_profiles",
    "_common_object_settings",
    "_settings_from_profiles",
]
