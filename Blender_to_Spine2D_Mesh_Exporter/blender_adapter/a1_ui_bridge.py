"""Translate Blender UI state into immutable A1 export contracts.

The bridge is the only production boundary that reads Scene/Object RNA for Rewrite exports.
It captures mutable Blender properties once, builds typed application settings, and routes the
request to the post-render output services. Preparation modules are never used as output
entry-points here.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Tuple

from ..application import (
    A1GeometryPreparationSettings,
    A1MultiObjectExportSettings,
    A1MultiObjectMode,
    A1SingleObjectExportSettings,
    A1SourceGeometryMode,
    ExportResult,
    ExportSettings,
)
from ..domain.baking import BakeExecutionSettings, sanitize_filename_stem
from ..domain.geometry import A1AngularMode
from ..domain.uv import UvUnwrapSettings
from .a1_mixed_object_output import export_a1_mixed_object
from .a1_multi_object_export import A1MultiObjectSource
from .a1_multi_object_output import export_a1_multi_object
from .a1_single_object_export import export_a1_single_object

logger = logging.getLogger(__name__)

_DEFAULT_PROJECTION_ALPHA_THRESHOLD = 1.0 / 255.0
_DEFAULT_BAKE_MARGIN = 4
_DEFAULT_UV_LAYER_NAME = "SpineBakeUV"


@dataclass(frozen=True, slots=True)
class _SceneExportProfile:
    """One immutable snapshot of all Scene-level Rewrite export settings."""

    output_directory: Path
    images_relative_path: str
    texture_size: int
    seam_mode: str
    angle_limit_degrees: float
    geometry: A1GeometryPreparationSettings
    bake_execution: BakeExecutionSettings
    include_control_icons: bool
    include_preview_animation: bool

    def __post_init__(self) -> None:
        if not isinstance(self.output_directory, Path):
            raise TypeError("output_directory must be pathlib.Path")
        if not isinstance(self.images_relative_path, str) or not self.images_relative_path:
            raise ValueError("images_relative_path must be a non-empty string")
        if not isinstance(self.texture_size, int) or self.texture_size <= 0:
            raise ValueError("texture_size must be a positive integer")
        if self.seam_mode not in {"AUTO", "CUSTOM"}:
            raise ValueError("seam_mode must be AUTO or CUSTOM")
        if not isinstance(self.angle_limit_degrees, (int, float)):
            raise TypeError("angle_limit_degrees must be numeric")
        if not isinstance(self.geometry, A1GeometryPreparationSettings):
            raise TypeError("geometry must be A1GeometryPreparationSettings")
        if not isinstance(self.bake_execution, BakeExecutionSettings):
            raise TypeError("bake_execution must be BakeExecutionSettings")
        if not isinstance(self.include_control_icons, bool):
            raise TypeError("include_control_icons must be bool")
        if not isinstance(self.include_preview_animation, bool):
            raise TypeError("include_preview_animation must be bool")


@dataclass(frozen=True, slots=True)
class _ObjectExportProfile:
    """Live Blender object handle plus values captured before preparation starts."""

    source_object: Any
    object_name: str
    sequence_start_frame: int
    sequence_frame_count: int
    connect_enabled: bool

    def __post_init__(self) -> None:
        if self.source_object is None:
            raise ValueError("source_object cannot be None")
        if not isinstance(self.object_name, str) or not self.object_name.strip():
            raise ValueError("object_name must be a non-empty string")
        if not isinstance(self.sequence_start_frame, int) or self.sequence_start_frame < 0:
            raise ValueError("sequence_start_frame must be a non-negative integer")
        if not isinstance(self.sequence_frame_count, int) or self.sequence_frame_count < 0:
            raise ValueError("sequence_frame_count must be a non-negative integer")
        if not isinstance(self.connect_enabled, bool):
            raise TypeError("connect_enabled must be bool")


def _load_bpy() -> Any:
    try:
        import bpy
    except Exception as exc:
        raise RuntimeError("Blender bpy module is unavailable") from exc
    return bpy


def _object_name(obj: Any) -> str:
    value = str(
        getattr(obj, "name_full", None)
        or getattr(obj, "name", None)
        or ""
    ).strip()
    if not value:
        raise ValueError("Selected mesh object has an empty name")
    return value


def _rna_identity(value: Any) -> tuple[str, object]:
    """Return stable identity across transient Blender RNA wrapper instances."""

    pointer = getattr(value, "as_pointer", None)
    if callable(pointer):
        try:
            resolved = int(pointer())
            if resolved:
                return ("RNA_POINTER", resolved)
        except Exception:
            logger.debug("Unable to read Blender RNA pointer", exc_info=True)
    name = str(
        getattr(value, "name_full", None)
        or getattr(value, "name", None)
        or ""
    ).strip()
    if name:
        return ("RNA_NAME", name)
    return ("PYTHON_ID", id(value))


def _active_mesh(context: Any) -> Any:
    obj = getattr(context, "active_object", None)
    if obj is None:
        raise ValueError("There is no active object")
    if getattr(obj, "type", None) != "MESH":
        raise ValueError(f"The active object '{_object_name(obj)}' is not a Mesh")
    if getattr(obj, "data", None) is None:
        raise ValueError(f"The active Mesh object '{_object_name(obj)}' has no data")
    return obj


def _ordered_selected_meshes(context: Any) -> Tuple[Any, ...]:
    """Return active Mesh first and all remaining Mesh objects in deterministic order."""

    raw_selected = tuple(
        obj
        for obj in getattr(context, "selected_objects", ())
        if getattr(obj, "type", None) == "MESH"
    )
    unique_by_identity: dict[tuple[str, object], Any] = {}
    for obj in raw_selected:
        unique_by_identity.setdefault(_rna_identity(obj), obj)
    selected = tuple(unique_by_identity.values())
    if len(selected) < 2:
        raise ValueError("Select at least two Mesh objects for multi-export")

    active = getattr(context, "active_object", None)
    active_identity = None if active is None else _rna_identity(active)
    active_match = next(
        (obj for obj in selected if _rna_identity(obj) == active_identity),
        None,
    )
    ordered: list[Any] = []
    if active_match is not None:
        ordered.append(active_match)
    ordered.extend(
        sorted(
            (
                obj
                for obj in selected
                if active_match is None
                or _rna_identity(obj) != _rna_identity(active_match)
            ),
            key=lambda obj: (_object_name(obj).casefold(), _object_name(obj)),
        )
    )
    return tuple(ordered)


def _resolve_output_directory(scene: Any) -> Path:
    bpy = _load_bpy()
    raw = str(getattr(scene, "spine2d_json_path", "") or "").strip()
    resolved = str(bpy.path.abspath(raw) if raw else "").strip()
    if not resolved:
        from ..config import get_default_output_dir

        resolved = str(get_default_output_dir()).strip()
    if not resolved:
        raise ValueError("Output directory is empty; save the .blend file first")
    return Path(resolved).expanduser().resolve(strict=False)


def _resolve_images_relative_path(scene: Any) -> str:
    value = str(getattr(scene, "spine2d_images_path", "images") or "images")
    normalized = value.replace("\\", "/").strip()
    while normalized.startswith("./"):
        normalized = normalized[2:]
    normalized = normalized.strip("/")
    return normalized or "images"


def _texture_size(scene: Any) -> int:
    value = int(getattr(scene, "spine2d_texture_size", 1024))
    if value <= 0:
        raise ValueError(f"Texture size must be positive, got {value}")
    return value


def _projection_alpha_threshold(scene: Any) -> float:
    property_name = "spine2d_projection_alpha_threshold"
    raw_value = getattr(scene, property_name, None)
    if raw_value is None:
        getter = getattr(scene, "get", None)
        if callable(getter):
            raw_value = getter(property_name, _DEFAULT_PROJECTION_ALPHA_THRESHOLD)
    if raw_value is None:
        raw_value = _DEFAULT_PROJECTION_ALPHA_THRESHOLD
    if isinstance(raw_value, bool):
        raise ValueError(
            "spine2d_projection_alpha_threshold must be numeric, not bool"
        )
    try:
        return float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "spine2d_projection_alpha_threshold must be numeric"
        ) from exc


def _connect_enabled(obj: Any) -> bool:
    settings = getattr(obj, "spine2d_connect_settings", None)
    return bool(settings is not None and getattr(settings, "enabled", False))


def _resolve_geometry_settings(scene: Any) -> A1GeometryPreparationSettings:
    raw_mode = str(
        getattr(
            scene,
            "spine2d_angular_mode",
            A1AngularMode.LEGACY_SEED_CONE.value,
        )
        or A1AngularMode.LEGACY_SEED_CONE.value
    ).strip().upper()
    try:
        angular_mode = A1AngularMode(raw_mode)
    except ValueError as exc:
        supported = tuple(mode.value for mode in A1AngularMode)
        raise ValueError(
            f"Unsupported Spine2D angular mode {raw_mode!r}; supported={supported}"
        ) from exc

    if angular_mode is A1AngularMode.LEGACY_SEED_CONE:
        local_angle_limit = None
    else:
        raw_local_limit = getattr(scene, "spine2d_local_angle_limit", None)
        local_angle_limit = (
            None
            if raw_local_limit is None or raw_local_limit == ""
            else float(raw_local_limit)
        )
    return A1GeometryPreparationSettings(
        angular_mode=angular_mode,
        local_angle_limit_degrees=local_angle_limit,
    )


def _capture_scene_profile(
    scene: Any,
    *,
    output_directory: Path | None = None,
    texture_size: int | None = None,
    images_relative_path: str | None = None,
) -> _SceneExportProfile:
    """Capture every mutable Scene setting exactly once for one export request."""

    seam_mode = str(
        getattr(scene, "spine2d_seam_maker_mode", "AUTO") or "AUTO"
    ).strip().upper()
    angle_limit = float(getattr(scene, "spine2d_angle_limit", 30.0))
    render_engine = str(
        getattr(getattr(scene, "render", None), "engine", "CYCLES") or "CYCLES"
    )
    return _SceneExportProfile(
        output_directory=(
            output_directory
            if output_directory is not None
            else _resolve_output_directory(scene)
        ),
        images_relative_path=(
            images_relative_path
            if images_relative_path is not None
            else _resolve_images_relative_path(scene)
        ),
        texture_size=(
            texture_size if texture_size is not None else _texture_size(scene)
        ),
        seam_mode=seam_mode,
        angle_limit_degrees=angle_limit,
        geometry=_resolve_geometry_settings(scene),
        bake_execution=BakeExecutionSettings(
            render_engine=render_engine,
            projection_alpha_threshold=_projection_alpha_threshold(scene),
        ),
        include_control_icons=bool(
            getattr(scene, "spine2d_control_icons", True)
        ),
        include_preview_animation=bool(
            getattr(scene, "spine2d_export_preview_animation", True)
        ),
    )


def _capture_object_profile(
    obj: Any,
    *,
    sequence_start_frame: int,
    sequence_frame_count: int,
    connect_enabled: bool,
) -> _ObjectExportProfile:
    return _ObjectExportProfile(
        source_object=obj,
        object_name=_object_name(obj),
        sequence_start_frame=max(0, int(sequence_start_frame)),
        sequence_frame_count=max(0, int(sequence_frame_count)),
        connect_enabled=bool(connect_enabled),
    )


def _settings_from_profiles(
    obj: _ObjectExportProfile,
    scene: _SceneExportProfile,
    *,
    json_output_stem: str | None = None,
) -> A1SingleObjectExportSettings:
    return A1SingleObjectExportSettings(
        export=ExportSettings(
            texture_width=scene.texture_size,
            texture_height=scene.texture_size,
            output_directory=scene.output_directory,
            images_relative_path=scene.images_relative_path,
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
        sequence_start_frame=int(getattr(scene, "spine2d_bake_frame_start", 0)),
        sequence_frame_count=int(getattr(scene, "spine2d_frames_for_render", 0)),
        json_output_stem=f"{sanitize_filename_stem(object_name)}_merged",
    )


def _build_sources_from_profiles(
    objects: Tuple[_ObjectExportProfile, ...],
    scene: _SceneExportProfile,
) -> Tuple[A1MultiObjectSource, ...]:
    result: list[A1MultiObjectSource] = []
    for index, obj in enumerate(objects, start=1):
        result.append(
            A1MultiObjectSource(
                source_object=obj.source_object,
                component_id=f"object_{index}:{obj.object_name}",
                animation_namespace=f"object_{index}",
                settings=_settings_from_profiles(obj, scene),
            )
        )
    return tuple(result)


def _build_sources(
    objects: Tuple[Any, ...],
    scene: Any,
    *,
    output_directory: Path,
    texture_size: int,
    images_relative_path: str,
) -> Tuple[A1MultiObjectSource, ...]:
    """Compatibility helper that captures one Scene snapshot for every source."""

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
                getattr(getattr(obj, "spine2d_bake_settings", None), "bake_frame_start", 0)
            ),
            sequence_frame_count=int(
                getattr(getattr(obj, "spine2d_bake_settings", None), "frames_for_render", 0)
            ),
            connect_enabled=_connect_enabled(obj),
        )
        for obj in objects
    )
    return _build_sources_from_profiles(profiles, scene_profile)


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
        sequence_start_frame=int(getattr(scene, "spine2d_bake_frame_start", 0)),
        sequence_frame_count=int(getattr(scene, "spine2d_frames_for_render", 0)),
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
    object_profiles = tuple(
        _capture_object_profile(
            obj,
            sequence_start_frame=int(
                getattr(getattr(obj, "spine2d_bake_settings", None), "bake_frame_start", 0)
            ),
            sequence_frame_count=int(
                getattr(getattr(obj, "spine2d_bake_settings", None), "frames_for_render", 0)
            ),
            connect_enabled=_connect_enabled(obj),
        )
        for obj in ordered_objects
    )
    sources = _build_sources_from_profiles(object_profiles, scene_profile)

    connected = tuple(
        source
        for source, profile in zip(sources, object_profiles)
        if profile.connect_enabled
    )
    standalone = tuple(
        source
        for source, profile in zip(sources, object_profiles)
        if not profile.connect_enabled
    )

    if len(connected) == 1:
        logger.warning(
            "One selected object has Connect enabled; connected export requires at "
            "least two objects, so all selected objects will be exported standalone"
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
        return export_a1_mixed_object(
            connected,
            standalone,
            settings,
            context=context,
            scene=scene,
        )

    mode = A1MultiObjectMode.CONNECTED if connected else A1MultiObjectMode.STANDALONE
    settings = A1MultiObjectExportSettings(
        output_directory=scene_profile.output_directory,
        output_stem=output_stem,
        mode=mode,
        anchor_component_id=(connected[0].component_id if connected else None),
    )
    return export_a1_multi_object(
        connected or standalone,
        settings,
        context=context,
        scene=scene,
    )


__all__ = [
    "export_active_object_a1",
    "export_selected_objects_a1",
]
