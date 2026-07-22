"""Strict output contract for Spine slot color timelines."""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from re import fullmatch
from typing import Any

from .setup_slot_contract import SetupSlotIndex, resolve_setup_slot_index
from .spine_json_contract import json_path_key


_SLOT_COLOR_FIELDS: dict[str, tuple[tuple[str, int], ...]] = {
    "rgba": (("color", 8),),
    "rgb": (("color", 6),),
    "rgba2": (("light", 8), ("dark", 6)),
    "rgb2": (("light", 6), ("dark", 6)),
}
_SLOT_COLOR_TIMELINE_NAMES = frozenset((*_SLOT_COLOR_FIELDS, "alpha"))


def _is_finite_number(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return isinstance(value, int) or isfinite(value)


def _require_hex_color(
    value: object,
    *,
    field_name: str,
    digits: int,
) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be str")

    if len(value) != digits or fullmatch(
        rf"[0-9A-Fa-f]{{{digits}}}",
        value,
    ) is None:
        color_kind = "RGBA" if digits == 8 else "RGB"
        raise ValueError(
            f"{field_name} must contain exactly {digits} hexadecimal "
            f"{color_kind} digits"
        )


def _require_finite_number(value: object, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a finite number")
    if not _is_finite_number(value):
        raise ValueError(f"{field_name} must be finite")


def validate_animation_slot_color_timelines(
    animations: Mapping[str, Any],
    *,
    slot_names: tuple[str, ...],
    path: str,
    setup_slot_index: SetupSlotIndex | None = None,
) -> None:
    """Validate known Spine slot color timelines without normalizing payloads."""

    if not isinstance(animations, Mapping):
        raise TypeError("animations must be a mapping")
    if not isinstance(slot_names, tuple):
        raise TypeError("slot_names must be tuple")
    if not isinstance(path, str) or not path:
        raise ValueError("path must be a non-empty string")

    slot_index = resolve_setup_slot_index(slot_names, setup_slot_index)

    for animation_name, animation_metadata in animations.items():
        animation_path = json_path_key(path, animation_name)
        if not isinstance(animation_metadata, Mapping):
            raise TypeError(f"{animation_path} must be a mapping")

        if "slots" not in animation_metadata:
            continue

        slot_timelines = animation_metadata["slots"]
        slots_path = f"{animation_path}.slots"
        if not isinstance(slot_timelines, Mapping):
            raise TypeError(f"{slots_path} must be a mapping")

        for slot_name, slot_metadata in slot_timelines.items():
            slot_path = json_path_key(slots_path, slot_name)
            slot_index.require(slot_name, path=slot_path)
            if not isinstance(slot_metadata, Mapping):
                raise TypeError(f"{slot_path} must be a mapping")

            for timeline_name in _SLOT_COLOR_TIMELINE_NAMES:
                if timeline_name not in slot_metadata:
                    continue

                timeline = slot_metadata[timeline_name]
                timeline_path = f"{slot_path}.{timeline_name}"
                if not isinstance(timeline, (list, tuple)):
                    raise TypeError(
                        f"{timeline_path} must be a list or tuple"
                    )
                if not timeline:
                    raise ValueError(f"{timeline_path} cannot be empty")

                previous_time: float | int | None = None
                for keyframe_index, keyframe in enumerate(timeline):
                    keyframe_path = f"{timeline_path}[{keyframe_index}]"
                    if not isinstance(keyframe, Mapping):
                        raise TypeError(f"{keyframe_path} must be a mapping")

                    time_value = keyframe.get("time", 0)
                    _require_finite_number(
                        time_value,
                        f"{keyframe_path}.time",
                    )
                    if (
                        previous_time is not None
                        and time_value < previous_time
                    ):
                        raise ValueError(
                            f"{keyframe_path}.time must be greater than or "
                            f"equal to the previous {timeline_name} time "
                            f"{previous_time}"
                        )
                    previous_time = time_value

                    if timeline_name == "alpha":
                        if "value" in keyframe:
                            _require_finite_number(
                                keyframe["value"],
                                f"{keyframe_path}.value",
                            )
                        continue

                    for field_name, digits in _SLOT_COLOR_FIELDS[timeline_name]:
                        if field_name not in keyframe:
                            raise ValueError(
                                f"{keyframe_path}.{field_name} is required"
                            )
                        _require_hex_color(
                            keyframe[field_name],
                            field_name=f"{keyframe_path}.{field_name}",
                            digits=digits,
                        )


__all__ = ["validate_animation_slot_color_timelines"]
