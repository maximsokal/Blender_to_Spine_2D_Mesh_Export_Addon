"""Blender 5.2 UV coordinate access through MeshUVLoopLayer attribute collections."""

from __future__ import annotations

from math import isfinite
from typing import Any, Iterable


class MeshUvAttributeError(RuntimeError):
    """Raised when a Blender 5.2 UV layer exposes malformed attribute data."""


def _layer_name(layer: Any) -> str:
    value = str(getattr(layer, "name", "") or "").strip()
    if not value:
        raise MeshUvAttributeError("UV layer name is empty")
    return value


def _uv_collection(layer: Any) -> Any:
    collection = getattr(layer, "uv", None)
    if collection is None:
        raise MeshUvAttributeError(
            f"Blender 5.2 UV layer '{_layer_name(layer)}' has no uv collection"
        )
    return collection


def _validate_length(layer: Any, expected_length: int) -> Any:
    if not isinstance(expected_length, int) or isinstance(expected_length, bool):
        raise TypeError("expected_length must be int")
    if expected_length < 0:
        raise ValueError("expected_length must be non-negative")
    collection = _uv_collection(layer)
    try:
        actual_length = len(collection)
    except Exception as exc:
        raise MeshUvAttributeError(
            f"Unable to inspect UV layer '{_layer_name(layer)}' length"
        ) from exc
    if actual_length != expected_length:
        raise MeshUvAttributeError(
            f"UV layer '{_layer_name(layer)}' contains {actual_length} values for "
            f"{expected_length} mesh loops"
        )
    return collection


def _coordinate_tuple(value: Any, *, label: str) -> tuple[float, float]:
    try:
        coordinate = (float(value[0]), float(value[1]))
    except Exception as exc:
        raise MeshUvAttributeError(f"Unable to read {label} as a 2D vector") from exc
    if not all(isfinite(component) for component in coordinate):
        raise MeshUvAttributeError(f"{label} contains a non-finite component")
    return coordinate


def read_uv_coordinate(
    layer: Any,
    loop_index: int,
    *,
    expected_length: int,
) -> tuple[float, float]:
    """Read one Blender 5.2 UV coordinate by mesh-loop index."""

    if not isinstance(loop_index, int) or isinstance(loop_index, bool):
        raise TypeError("loop_index must be int")
    collection = _validate_length(layer, expected_length)
    if loop_index < 0 or loop_index >= expected_length:
        raise MeshUvAttributeError(
            f"UV loop index {loop_index} is outside [0, {expected_length})"
        )
    try:
        vector = collection[loop_index].vector
    except Exception as exc:
        raise MeshUvAttributeError(
            f"Unable to access UV layer '{_layer_name(layer)}' value {loop_index}"
        ) from exc
    return _coordinate_tuple(
        vector,
        label=f"UV layer '{_layer_name(layer)}' value {loop_index}",
    )


def read_uv_coordinates(
    layer: Any,
    *,
    expected_length: int,
) -> tuple[tuple[float, float], ...]:
    """Read all Blender 5.2 UV coordinates in mesh-loop order."""

    collection = _validate_length(layer, expected_length)
    resolved: list[tuple[float, float]] = []
    for index in range(expected_length):
        try:
            vector = collection[index].vector
        except Exception as exc:
            raise MeshUvAttributeError(
                f"Unable to access UV layer '{_layer_name(layer)}' value {index}"
            ) from exc
        resolved.append(
            _coordinate_tuple(
                vector,
                label=f"UV layer '{_layer_name(layer)}' value {index}",
            )
        )
    return tuple(resolved)


def write_uv_coordinate(
    layer: Any,
    loop_index: int,
    coordinate: Iterable[float],
    *,
    expected_length: int,
) -> None:
    """Write one Blender 5.2 UV coordinate by mesh-loop index."""

    if not isinstance(loop_index, int) or isinstance(loop_index, bool):
        raise TypeError("loop_index must be int")
    collection = _validate_length(layer, expected_length)
    if loop_index < 0 or loop_index >= expected_length:
        raise MeshUvAttributeError(
            f"UV loop index {loop_index} is outside [0, {expected_length})"
        )
    try:
        raw = tuple(coordinate)
    except Exception as exc:
        raise TypeError("coordinate must be iterable") from exc
    if len(raw) != 2:
        raise MeshUvAttributeError("UV coordinate must contain exactly two values")
    resolved = _coordinate_tuple(raw, label="UV coordinate")
    try:
        collection[loop_index].vector = resolved
    except Exception as exc:
        raise MeshUvAttributeError(
            f"Unable to write UV layer '{_layer_name(layer)}' value {loop_index}"
        ) from exc


__all__ = [
    "MeshUvAttributeError",
    "read_uv_coordinate",
    "read_uv_coordinates",
    "write_uv_coordinate",
]
