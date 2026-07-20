"""Low-level Blender-compatible reads used by scene-bake capture."""

from __future__ import annotations

from math import isfinite
from typing import Any, Tuple

from .scene_bake_error import SceneBakeAnalysisError


def load_bpy() -> Any:
    """Import Blender lazily only when no explicit Scene or Context was supplied."""

    try:
        import bpy
    except Exception as exc:
        raise SceneBakeAnalysisError("Blender bpy module is unavailable") from exc
    return bpy


def name(value: Any, *, fallback: str = "") -> str:
    return str(
        getattr(value, "name_full", None)
        or getattr(value, "name", None)
        or fallback
        or ""
    ).strip()


def matrix_tuple(value: Any) -> Tuple[float, ...]:
    try:
        result = tuple(
            float(value[row][column])
            for row in range(4)
            for column in range(4)
        )
    except Exception as exc:
        raise SceneBakeAnalysisError("Unable to read a 4x4 Blender matrix") from exc
    if not all(isfinite(item) for item in result):
        raise SceneBakeAnalysisError("Blender matrix contains non-finite values")
    return result


def color_tuple(
    value: Any,
    *,
    default: Tuple[float, float, float],
    label: str = "Blender color",
) -> Tuple[float, float, float]:
    try:
        result = float(value[0]), float(value[1]), float(value[2])
    except Exception:
        try:
            result = float(default[0]), float(default[1]), float(default[2])
        except Exception as exc:
            raise SceneBakeAnalysisError(f"{label} fallback is invalid") from exc
    if not all(isfinite(item) for item in result):
        raise SceneBakeAnalysisError(f"{label} contains non-finite values")
    return result


def finite_float(value: Any, *, label: str) -> float:
    try:
        result = float(value)
    except Exception as exc:
        raise SceneBakeAnalysisError(f"{label} is not numeric") from exc
    if not isfinite(result):
        raise SceneBakeAnalysisError(f"{label} must be finite")
    return result


def non_negative_float(value: Any, *, label: str) -> float:
    return max(0.0, finite_float(value, label=label))


def positive_float(value: Any, *, label: str, minimum: float) -> float:
    if not isinstance(minimum, (int, float)) or not isfinite(float(minimum)):
        raise TypeError("minimum must be a finite number")
    if float(minimum) <= 0.0:
        raise ValueError("minimum must be positive")
    return max(float(minimum), finite_float(value, label=label))


def animated(*datablocks: Any) -> bool:
    for datablock in datablocks:
        if datablock is None:
            continue
        animation_data = getattr(datablock, "animation_data", None)
        if animation_data is None:
            continue
        if getattr(animation_data, "action", None) is not None:
            return True
        try:
            if len(getattr(animation_data, "drivers", ())) > 0:
                return True
        except Exception:
            return True
    return False


def visible_boolean(obj: Any, property_name: str, default: bool) -> bool:
    if not isinstance(property_name, str) or not property_name:
        raise ValueError("property_name must be a non-empty string")
    if not isinstance(default, bool):
        raise TypeError("default must be bool")
    try:
        return bool(getattr(obj, property_name))
    except Exception:
        return default


def object_render_visible(obj: Any) -> bool:
    try:
        return not bool(getattr(obj, "hide_render", False))
    except Exception:
        return True


def resolve_scene_inputs(
    *,
    scene: Any | None = None,
    context: Any | None = None,
) -> tuple[Any | None, Any | None, Any]:
    """Resolve a Scene without importing bpy when explicit inputs are sufficient."""

    if scene is not None:
        return None, context, scene
    if context is not None:
        resolved_scene = getattr(context, "scene", None)
        if resolved_scene is None:
            raise SceneBakeAnalysisError("A Blender Scene is required")
        return None, context, resolved_scene

    bpy_module = load_bpy()
    resolved_context = getattr(bpy_module, "context", None)
    resolved_scene = getattr(resolved_context, "scene", None)
    if resolved_scene is None:
        raise SceneBakeAnalysisError("A Blender Scene is required")
    return bpy_module, resolved_context, resolved_scene


__all__ = [
    "animated",
    "color_tuple",
    "finite_float",
    "load_bpy",
    "matrix_tuple",
    "name",
    "non_negative_float",
    "object_render_visible",
    "positive_float",
    "resolve_scene_inputs",
    "visible_boolean",
]
