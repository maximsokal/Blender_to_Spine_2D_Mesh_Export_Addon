"""Capture immutable Rewrite export profiles from Blender RNA state."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Tuple

from ..application import A1GeometryPreparationSettings
from ..domain.baking import BakeExecutionSettings
from ..domain.geometry import A1AngularMode


logger = logging.getLogger(__name__)
_DEFAULT_PROJECTION_ALPHA_THRESHOLD = 1.0 / 255.0


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
    """Return active Mesh first and remaining unique meshes in deterministic order."""

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


__all__ = [
    "_ObjectExportProfile",
    "_SceneExportProfile",
    "_active_mesh",
    "_capture_object_profile",
    "_capture_scene_profile",
    "_connect_enabled",
    "_object_name",
    "_ordered_selected_meshes",
    "_projection_alpha_threshold",
    "_resolve_geometry_settings",
    "_resolve_images_relative_path",
    "_resolve_output_directory",
    "_rna_identity",
    "_texture_size",
]
