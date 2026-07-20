"""Capture source-object, Light, Camera, and color-management snapshots."""

from __future__ import annotations

from typing import Any

from ..domain.baking.context import (
    CameraBakeSnapshot,
    ColorManagementSnapshot,
    LightBakeSnapshot,
    ObjectBakeContext,
)
from .scene_bake_error import SceneBakeAnalysisError
from .scene_bake_rna import (
    animated,
    color_tuple,
    finite_float,
    matrix_tuple,
    name,
    non_negative_float,
    positive_float,
    visible_boolean,
)


def analyse_object_bake_context(obj: Any) -> ObjectBakeContext:
    if obj is None or getattr(obj, "type", None) != "MESH":
        raise SceneBakeAnalysisError("obj must be a Blender MESH object")
    object_id = name(obj)
    if not object_id:
        raise SceneBakeAnalysisError("Source object name is empty")
    try:
        collections = tuple(getattr(obj, "users_collection", ()))
    except Exception as exc:
        raise SceneBakeAnalysisError(f"Unable to inspect collections of source object '{object_id}'") from exc
    collection_names = tuple(sorted({collection_name for collection in collections if (collection_name := name(collection))}, key=str.casefold))
    data = getattr(obj, "data", None)
    try:
        return ObjectBakeContext(
            source_object_id=object_id,
            object_type=str(getattr(obj, "type", "MESH") or "MESH"),
            world_matrix=matrix_tuple(getattr(obj, "matrix_world", None)),
            collection_names=collection_names,
            hide_render=bool(getattr(obj, "hide_render", False)),
            visible_camera=visible_boolean(obj, "visible_camera", True),
            visible_shadow=visible_boolean(obj, "visible_shadow", True),
            animated=animated(obj, data),
        )
    except SceneBakeAnalysisError:
        raise
    except (TypeError, ValueError, OverflowError) as exc:
        raise SceneBakeAnalysisError(f"Unable to build source object snapshot for '{object_id}': {exc}") from exc


def analyse_light(obj: Any) -> LightBakeSnapshot:
    data = getattr(obj, "data", None)
    object_id = name(obj)
    if not object_id or data is None:
        raise SceneBakeAnalysisError("Visible light object is missing name or data")
    try:
        return LightBakeSnapshot(
            object_id=object_id,
            light_type=str(getattr(data, "type", "POINT") or "POINT"),
            energy=non_negative_float(getattr(data, "energy", 0.0) or 0.0, label=f"Light '{object_id}' energy"),
            color=color_tuple(getattr(data, "color", (1.0, 1.0, 1.0)), default=(1.0, 1.0, 1.0), label=f"Light '{object_id}' color"),
            world_matrix=matrix_tuple(getattr(obj, "matrix_world", None)),
            use_shadow=bool(getattr(data, "use_shadow", True)),
            animated=animated(obj, data),
        )
    except SceneBakeAnalysisError:
        raise
    except (TypeError, ValueError, OverflowError) as exc:
        raise SceneBakeAnalysisError(f"Unable to build Light snapshot for '{object_id}': {exc}") from exc


def analyse_camera(scene: Any) -> CameraBakeSnapshot | None:
    obj = getattr(scene, "camera", None)
    if obj is None:
        return None
    data = getattr(obj, "data", None)
    object_id = name(obj)
    if not object_id or data is None or getattr(obj, "type", None) != "CAMERA":
        raise SceneBakeAnalysisError("scene.camera is not a valid Blender Camera object")
    lens = positive_float(getattr(data, "lens", 50.0) or 50.0, label=f"Camera '{object_id}' lens", minimum=1e-8)
    ortho_scale = positive_float(getattr(data, "ortho_scale", 6.0) or 6.0, label=f"Camera '{object_id}' ortho_scale", minimum=1e-8)
    clip_start = positive_float(getattr(data, "clip_start", 0.1) or 0.1, label=f"Camera '{object_id}' clip_start", minimum=1e-8)
    clip_end = positive_float(getattr(data, "clip_end", 1000.0) or 1000.0, label=f"Camera '{object_id}' clip_end", minimum=1e-7)
    if clip_end <= clip_start:
        raise SceneBakeAnalysisError(f"Camera '{object_id}' clip_end must be greater than clip_start")
    try:
        return CameraBakeSnapshot(
            object_id=object_id,
            camera_type=str(getattr(data, "type", "PERSP") or "PERSP"),
            world_matrix=matrix_tuple(getattr(obj, "matrix_world", None)),
            lens=lens,
            ortho_scale=ortho_scale,
            clip_start=clip_start,
            clip_end=clip_end,
            animated=animated(obj, data),
        )
    except SceneBakeAnalysisError:
        raise
    except (TypeError, ValueError, OverflowError) as exc:
        raise SceneBakeAnalysisError(f"Unable to build Camera snapshot for '{object_id}': {exc}") from exc


def analyse_color_management(scene: Any) -> ColorManagementSnapshot:
    view = getattr(scene, "view_settings", None)
    if view is None:
        raise SceneBakeAnalysisError("Scene view_settings are unavailable")
    gamma_value = getattr(view, "gamma", 1.0)
    if gamma_value is None:
        gamma_value = 1.0
    try:
        return ColorManagementSnapshot(
            view_transform=str(getattr(view, "view_transform", "Standard") or "Standard"),
            look=str(getattr(view, "look", "") or ""),
            exposure=finite_float(getattr(view, "exposure", 0.0) or 0.0, label="Scene color-management exposure"),
            gamma=finite_float(gamma_value, label="Scene color-management gamma"),
        )
    except SceneBakeAnalysisError:
        raise
    except (TypeError, ValueError, OverflowError) as exc:
        raise SceneBakeAnalysisError(f"Unable to build color-management snapshot: {exc}") from exc


__all__ = ["analyse_camera", "analyse_color_management", "analyse_light", "analyse_object_bake_context"]
