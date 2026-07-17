"""Translate the existing Blender UI properties into rewritten A1 export requests."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Tuple

from ..application import (
    A1MultiObjectExportSettings,
    A1MultiObjectMode,
    A1SingleObjectExportSettings,
    A1SourceGeometryMode,
    ExportResult,
    ExportSettings,
)
from ..domain.baking import BakeExecutionSettings, sanitize_filename_stem
from ..domain.uv import UvUnwrapSettings
from .a1_mixed_object_export import export_a1_mixed_object
from .a1_multi_object_export import A1MultiObjectSource, export_a1_multi_object
from .a1_single_object_export import export_a1_single_object


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
    selected = tuple(
        obj for obj in getattr(context, "selected_objects", ()) if obj.type == "MESH"
    )
    if len(selected) < 2:
        raise ValueError("Select at least two Mesh objects for multi-export")
    active = getattr(context, "active_object", None)
    ordered: list[Any] = []
    if active is not None and active in selected and active.type == "MESH":
        ordered.append(active)
    ordered.extend(
        sorted(
            (obj for obj in selected if obj is not active),
            key=lambda obj: _object_name(obj).casefold(),
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


def _connect_enabled(obj: Any) -> bool:
    settings = getattr(obj, "spine2d_connect_settings", None)
    return bool(settings is not None and getattr(settings, "enabled", False))


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
    object_name = _object_name(obj)
    seam_mode = str(getattr(scene, "spine2d_seam_maker_mode", "AUTO"))
    angle_limit = float(getattr(scene, "spine2d_angle_limit", 30.0))
    render_engine = str(
        getattr(getattr(scene, "render", None), "engine", "CYCLES") or "CYCLES"
    )
    return A1SingleObjectExportSettings(
        export=ExportSettings(
            texture_width=texture_size,
            texture_height=texture_size,
            output_directory=output_directory,
            images_relative_path=images_relative_path,
            seam_mode=seam_mode,
            angle_limit_degrees=angle_limit,
            bake_margin=4,
            sequence_start_frame=max(0, int(sequence_start_frame)),
            sequence_frame_count=max(0, int(sequence_frame_count)),
        ),
        prefix=object_name,
        output_stem=sanitize_filename_stem(object_name),
        json_output_stem=json_output_stem,
        source_geometry_mode=A1SourceGeometryMode.ORIGINAL,
        uv=UvUnwrapSettings(layer_name="SpineBakeUV"),
        bake_execution=BakeExecutionSettings(render_engine=render_engine),
        include_control_icons=bool(
            getattr(scene, "spine2d_control_icons", True)
        ),
        include_preview_animation=bool(
            getattr(scene, "spine2d_export_preview_animation", True)
        ),
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


def _build_sources(
    objects: Tuple[Any, ...],
    scene: Any,
    *,
    output_directory: Path,
    texture_size: int,
    images_relative_path: str,
) -> Tuple[A1MultiObjectSource, ...]:
    result: list[A1MultiObjectSource] = []
    for index, obj in enumerate(objects, start=1):
        object_name = _object_name(obj)
        result.append(
            A1MultiObjectSource(
                source_object=obj,
                component_id=f"object_{index}:{object_name}",
                animation_namespace=f"object_{index}",
                settings=_build_multi_object_settings(
                    obj,
                    scene,
                    output_directory=output_directory,
                    texture_size=texture_size,
                    images_relative_path=images_relative_path,
                ),
            )
        )
    return tuple(result)


def export_active_object_a1(context: Any) -> ExportResult:
    """Export the active Mesh through the complete single-object A1 service."""

    if context is None:
        raise ValueError("context cannot be None")
    scene = getattr(context, "scene", None)
    if scene is None:
        raise ValueError("context.scene is missing")
    obj = _active_mesh(context)
    output_directory = _resolve_output_directory(scene)
    settings = _build_single_object_settings(
        obj,
        scene,
        output_directory=output_directory,
        texture_size=_texture_size(scene),
        images_relative_path=_resolve_images_relative_path(scene),
    )
    return export_a1_single_object(
        obj,
        settings,
        context=context,
        scene=scene,
    )


def export_selected_objects_a1(context: Any) -> ExportResult:
    """Export selected meshes through standalone, connected, or mixed A1."""

    if context is None:
        raise ValueError("context cannot be None")
    scene = getattr(context, "scene", None)
    if scene is None:
        raise ValueError("context.scene is missing")
    objects = _ordered_selected_meshes(context)
    output_directory = _resolve_output_directory(scene)
    texture_size = _texture_size(scene)
    images_relative_path = _resolve_images_relative_path(scene)
    sources = _build_sources(
        objects,
        scene,
        output_directory=output_directory,
        texture_size=texture_size,
        images_relative_path=images_relative_path,
    )
    connected = tuple(
        source for source, obj in zip(sources, objects) if _connect_enabled(obj)
    )
    standalone = tuple(
        source for source, obj in zip(sources, objects) if not _connect_enabled(obj)
    )

    if len(connected) < 2:
        connected = ()
        standalone = sources

    base_name = sanitize_filename_stem(_object_name(objects[0]))
    output_stem = f"{base_name}_plus_{len(objects) - 1}_objects"
    if connected and standalone:
        settings = A1MultiObjectExportSettings(
            output_directory=output_directory,
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
        output_directory=output_directory,
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
