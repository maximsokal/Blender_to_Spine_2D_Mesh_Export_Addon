"""Strict interpolation-curve contract for Spine 4.2 animation timelines."""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from typing import Any

from .spine_json_contract import json_path_key


_SLOT_CURVE_CHANNELS: dict[str, int] = {
    "rgba": 4,
    "rgb": 3,
    "alpha": 1,
    "rgba2": 7,
    "rgb2": 6,
}
_BONE_CURVE_CHANNELS: dict[str, int] = {
    "rotate": 1,
    "translate": 2,
    "translatex": 1,
    "translatey": 1,
    "scale": 2,
    "scalex": 1,
    "scaley": 1,
    "shear": 2,
    "shearx": 1,
    "sheary": 1,
}
_PATH_CURVE_CHANNELS: dict[str, int] = {
    "position": 1,
    "spacing": 1,
    "mix": 3,
}
_PHYSICS_CURVE_CHANNELS: dict[str, int] = {
    "inertia": 1,
    "strength": 1,
    "damping": 1,
    "mass": 1,
    "wind": 1,
    "gravity": 1,
    "mix": 1,
}
_NESTED_CURVE_SECTIONS: dict[str, dict[str, int]] = {
    "slots": _SLOT_CURVE_CHANNELS,
    "bones": _BONE_CURVE_CHANNELS,
    "path": _PATH_CURVE_CHANNELS,
    "physics": _PHYSICS_CURVE_CHANNELS,
}
_DIRECT_CURVE_SECTIONS: dict[str, int] = {
    "ik": 2,
    "transform": 6,
}


def _mapping_key_path(path: str, key: object) -> str:
    if not isinstance(key, str):
        raise TypeError(f"{path} keys must be str")
    return json_path_key(path, key)


def _require_finite_number(value: object, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a finite number")
    if isinstance(value, float) and not isfinite(value):
        raise ValueError(f"{field_name} must be finite")


def _validate_curve_value(
    curve: object,
    *,
    channel_count: int,
    path: str,
) -> None:
    """Validate one Spine curve without rewriting absolute control points."""

    if curve == "stepped":
        return

    if isinstance(curve, str):
        raise ValueError(
            f'{path} must be exactly "stepped" or a Bezier number sequence'
        )

    if not isinstance(curve, (list, tuple)):
        raise TypeError(
            f'{path} must be exactly "stepped" or a list or tuple'
        )

    expected_length = channel_count * 4
    if len(curve) != expected_length:
        raise ValueError(
            f"{path} must contain exactly {expected_length} Bezier numbers "
            f"for {channel_count} interpolation channel(s)"
        )

    for value_index, value in enumerate(curve):
        _require_finite_number(value, f"{path}[{value_index}]")


def _validate_curve_timeline(
    timeline: object,
    *,
    channel_count: int,
    path: str,
) -> None:
    if not isinstance(timeline, (list, tuple)):
        raise TypeError(f"{path} must be a list or tuple")
    if not timeline:
        raise ValueError(f"{path} cannot be empty")

    previous_time: float | int | None = None
    last_keyframe_index = len(timeline) - 1
    for keyframe_index, keyframe in enumerate(timeline):
        keyframe_path = f"{path}[{keyframe_index}]"
        if not isinstance(keyframe, Mapping):
            raise TypeError(f"{keyframe_path} must be a mapping")

        time_value = keyframe.get("time", 0)
        _require_finite_number(time_value, f"{keyframe_path}.time")
        if previous_time is not None and time_value < previous_time:
            raise ValueError(
                f"{keyframe_path}.time must be greater than or equal to "
                f"the previous timeline time {previous_time}"
            )
        previous_time = time_value

        # Spine reads a curve only when the current frame has a following frame.
        # A terminal curve is inert metadata and is preserved for compatibility.
        if "curve" in keyframe and keyframe_index < last_keyframe_index:
            _validate_curve_value(
                keyframe["curve"],
                channel_count=channel_count,
                path=f"{keyframe_path}.curve",
            )


def _validate_nested_curve_section(
    section: object,
    *,
    timeline_channels: Mapping[str, int],
    path: str,
) -> None:
    if not isinstance(section, Mapping):
        raise TypeError(f"{path} must be a mapping")

    for owner_name, owner_timelines in section.items():
        owner_path = _mapping_key_path(path, owner_name)
        if not isinstance(owner_timelines, Mapping):
            raise TypeError(f"{owner_path} must be a mapping")

        for timeline_name, channel_count in timeline_channels.items():
            if timeline_name not in owner_timelines:
                continue
            _validate_curve_timeline(
                owner_timelines[timeline_name],
                channel_count=channel_count,
                path=f"{owner_path}.{timeline_name}",
            )


def _validate_direct_curve_section(
    section: object,
    *,
    channel_count: int,
    path: str,
) -> None:
    if not isinstance(section, Mapping):
        raise TypeError(f"{path} must be a mapping")

    for owner_name, timeline in section.items():
        _validate_curve_timeline(
            timeline,
            channel_count=channel_count,
            path=_mapping_key_path(path, owner_name),
        )


def validate_animation_curves(
    animations: Mapping[str, Any],
    *,
    path: str,
) -> None:
    """Validate known slot, bone, and constraint curve-bearing timelines.

    Omitted curves remain linear. Exact ``"stepped"`` strings and absolute
    Bezier control arrays are preserved byte-for-byte. Curves on terminal
    keyframes are preserved as inert metadata, matching Spine runtime parsing.
    Discrete timelines, deform/sequence attachment timelines, and unknown
    future timeline kinds are intentionally outside this contract.
    """

    if not isinstance(animations, Mapping):
        raise TypeError("animations must be a mapping")
    if not isinstance(path, str) or not path:
        raise ValueError("path must be a non-empty string")

    for animation_name, animation_metadata in animations.items():
        animation_path = _mapping_key_path(path, animation_name)
        if not isinstance(animation_metadata, Mapping):
            raise TypeError(f"{animation_path} must be a mapping")

        for section_name, timeline_channels in _NESTED_CURVE_SECTIONS.items():
            if section_name not in animation_metadata:
                continue
            _validate_nested_curve_section(
                animation_metadata[section_name],
                timeline_channels=timeline_channels,
                path=f"{animation_path}.{section_name}",
            )

        for section_name, channel_count in _DIRECT_CURVE_SECTIONS.items():
            if section_name not in animation_metadata:
                continue
            _validate_direct_curve_section(
                animation_metadata[section_name],
                channel_count=channel_count,
                path=f"{animation_path}.{section_name}",
            )


__all__ = ["validate_animation_curves"]
