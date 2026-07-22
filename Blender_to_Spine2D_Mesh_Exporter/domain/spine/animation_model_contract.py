"""Shared construction-time and pre-serialization Spine animation validation."""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from typing import Any

from .setup_attachment_contract import (
    SetupAttachmentNameIndex,
    resolve_setup_attachment_name_index,
)
from .setup_slot_contract import SetupSlotIndex, resolve_setup_slot_index
from .spine_json_contract import json_path_key, validate_json_mapping


_EVENT_STRING_FIELDS = ("string", "audio")
_EVENT_TIMELINE_STRING_FIELDS = ("string",)
_EVENT_NUMBER_FIELDS = ("float", "volume", "balance")
_EVENT_INT_MIN = -(2**31)
_EVENT_INT_MAX = 2**31 - 1


def _require_name(value: object, field_name: str = "name") -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be str")
    if not value.strip():
        raise ValueError(f"{field_name} cannot be empty")
    return value


def _is_finite_number(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return isinstance(value, int) or isfinite(value)


def _validate_event_definitions(
    events: Mapping[str, Any],
    *,
    path: str,
) -> None:
    """Validate setup-pose event definitions without inserting runtime defaults."""

    for event_name, event_metadata in events.items():
        event_path = json_path_key(path, event_name)
        _require_name(event_name, f"{event_path} event name")
        if not isinstance(event_metadata, Mapping):
            raise TypeError(f"{event_path} must be a mapping")

        if "int" in event_metadata:
            int_value = event_metadata["int"]
            if isinstance(int_value, bool) or not isinstance(int_value, int):
                raise TypeError(f"{event_path}.int must be int")
            if int_value < _EVENT_INT_MIN or int_value > _EVENT_INT_MAX:
                raise ValueError(
                    f"{event_path}.int must be inside signed 32-bit range "
                    f"[{_EVENT_INT_MIN}, {_EVENT_INT_MAX}]"
                )

        for field_name in _EVENT_STRING_FIELDS:
            if field_name in event_metadata and not isinstance(
                event_metadata[field_name], str
            ):
                raise TypeError(f"{event_path}.{field_name} must be str")

        for field_name in _EVENT_NUMBER_FIELDS:
            if field_name not in event_metadata:
                continue
            value = event_metadata[field_name]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{event_path}.{field_name} must be a finite number")
            if not _is_finite_number(value):
                raise ValueError(f"{event_path}.{field_name} must be finite")


def _validate_animation_event_timelines(
    animations: Mapping[str, Any],
    *,
    event_definitions: Mapping[str, Any],
    path: str,
) -> None:
    """Validate animation event keyframes and their setup-event references."""

    event_names = frozenset(event_definitions)
    for animation_name, animation_metadata in animations.items():
        animation_path = json_path_key(path, animation_name)
        if not isinstance(animation_metadata, Mapping):
            raise TypeError(f"{animation_path} must be a mapping")

        if "events" not in animation_metadata:
            continue

        timeline = animation_metadata["events"]
        timeline_path = f"{animation_path}.events"
        if not isinstance(timeline, (list, tuple)):
            raise TypeError(f"{timeline_path} must be a list or tuple")
        if not timeline:
            raise ValueError(f"{timeline_path} cannot be empty")

        previous_time: float | int | None = None
        for keyframe_index, keyframe in enumerate(timeline):
            keyframe_path = f"{timeline_path}[{keyframe_index}]"
            if not isinstance(keyframe, Mapping):
                raise TypeError(f"{keyframe_path} must be a mapping")

            if "name" not in keyframe:
                raise ValueError(f"{keyframe_path}.name is required")
            event_name = keyframe["name"]
            _require_name(event_name, f"{keyframe_path}.name")
            if event_name not in event_names:
                raise ValueError(
                    f"{keyframe_path}.name references undefined event "
                    f"'{event_name}'"
                )

            time_value = keyframe.get("time", 0)
            if isinstance(time_value, bool) or not isinstance(
                time_value,
                (int, float),
            ):
                raise TypeError(f"{keyframe_path}.time must be a finite number")
            if not _is_finite_number(time_value):
                raise ValueError(f"{keyframe_path}.time must be finite")
            if previous_time is not None and time_value < previous_time:
                raise ValueError(
                    f"{keyframe_path}.time must be greater than or equal to "
                    f"the previous event time {previous_time}"
                )
            previous_time = time_value

            if "int" in keyframe:
                int_value = keyframe["int"]
                if isinstance(int_value, bool) or not isinstance(int_value, int):
                    raise TypeError(f"{keyframe_path}.int must be int")
                if int_value < _EVENT_INT_MIN or int_value > _EVENT_INT_MAX:
                    raise ValueError(
                        f"{keyframe_path}.int must be inside signed 32-bit range "
                        f"[{_EVENT_INT_MIN}, {_EVENT_INT_MAX}]"
                    )

            for field_name in _EVENT_TIMELINE_STRING_FIELDS:
                if field_name in keyframe and not isinstance(
                    keyframe[field_name],
                    str,
                ):
                    raise TypeError(f"{keyframe_path}.{field_name} must be str")

            for field_name in _EVENT_NUMBER_FIELDS:
                if field_name not in keyframe:
                    continue
                value = keyframe[field_name]
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise TypeError(
                        f"{keyframe_path}.{field_name} must be a finite number"
                    )
                if not _is_finite_number(value):
                    raise ValueError(f"{keyframe_path}.{field_name} must be finite")


def _validate_animation_draw_order_timelines(
    animations: Mapping[str, Any],
    *,
    setup_slot_index: SetupSlotIndex,
    path: str,
) -> None:
    """Validate draw-order keyframes as deterministic slot permutations."""

    if not isinstance(setup_slot_index, SetupSlotIndex):
        raise TypeError("setup_slot_index must be SetupSlotIndex")

    slot_count = len(setup_slot_index.slot_names)
    for animation_name, animation_metadata in animations.items():
        animation_path = json_path_key(path, animation_name)
        if not isinstance(animation_metadata, Mapping):
            raise TypeError(f"{animation_path} must be a mapping")

        if "drawOrder" not in animation_metadata:
            continue

        timeline = animation_metadata["drawOrder"]
        timeline_path = f"{animation_path}.drawOrder"
        if not isinstance(timeline, (list, tuple)):
            raise TypeError(f"{timeline_path} must be a list or tuple")
        if not timeline:
            raise ValueError(f"{timeline_path} cannot be empty")

        previous_time: float | int | None = None
        for keyframe_index, keyframe in enumerate(timeline):
            keyframe_path = f"{timeline_path}[{keyframe_index}]"
            if not isinstance(keyframe, Mapping):
                raise TypeError(f"{keyframe_path} must be a mapping")

            time_value = keyframe.get("time", 0)
            if isinstance(time_value, bool) or not isinstance(
                time_value,
                (int, float),
            ):
                raise TypeError(f"{keyframe_path}.time must be a finite number")
            if not _is_finite_number(time_value):
                raise ValueError(f"{keyframe_path}.time must be finite")
            if previous_time is not None and time_value < previous_time:
                raise ValueError(
                    f"{keyframe_path}.time must be greater than or equal to "
                    f"the previous draw order time {previous_time}"
                )
            previous_time = time_value

            if "offsets" not in keyframe:
                continue

            offsets = keyframe["offsets"]
            offsets_path = f"{keyframe_path}.offsets"
            if not isinstance(offsets, (list, tuple)):
                raise TypeError(f"{offsets_path} must be a list or tuple")

            previous_source_index = -1
            seen_slot_names: set[str] = set()
            target_to_entry_index: dict[int, int] = {}
            for offset_index, offset_entry in enumerate(offsets):
                entry_path = f"{offsets_path}[{offset_index}]"
                if not isinstance(offset_entry, Mapping):
                    raise TypeError(f"{entry_path} must be a mapping")

                if "slot" not in offset_entry:
                    raise ValueError(f"{entry_path}.slot is required")
                slot_name = offset_entry["slot"]
                _require_name(slot_name, f"{entry_path}.slot")
                if slot_name in seen_slot_names:
                    raise ValueError(
                        f"{entry_path}.slot duplicates slot '{slot_name}' "
                        "in the same draw order keyframe"
                    )
                seen_slot_names.add(slot_name)

                source_index = setup_slot_index.require(
                    slot_name,
                    path=f"{entry_path}.slot",
                )
                if source_index <= previous_source_index:
                    raise ValueError(
                        f"{entry_path}.slot must follow setup slot order"
                    )
                previous_source_index = source_index

                if "offset" not in offset_entry:
                    raise ValueError(f"{entry_path}.offset is required")
                offset_value = offset_entry["offset"]
                if isinstance(offset_value, bool) or not isinstance(offset_value, int):
                    raise TypeError(f"{entry_path}.offset must be int")

                target_index = source_index + offset_value
                if target_index < 0 or target_index >= slot_count:
                    raise ValueError(
                        f"{entry_path}.offset moves slot '{slot_name}' outside "
                        f"draw order range [0, {slot_count})"
                    )

                previous_entry_index = target_to_entry_index.get(target_index)
                if previous_entry_index is not None:
                    raise ValueError(
                        f"{entry_path}.offset targets draw order index "
                        f"{target_index}, already used by "
                        f"{offsets_path}[{previous_entry_index}]"
                    )
                target_to_entry_index[target_index] = offset_index


def _validate_animation_slot_attachment_timelines(
    animations: Mapping[str, Any],
    *,
    setup_slot_index: SetupSlotIndex,
    setup_attachment_index: SetupAttachmentNameIndex,
    path: str,
) -> None:
    """Validate slot attachment timelines and their setup attachment references."""

    if not isinstance(setup_slot_index, SetupSlotIndex):
        raise TypeError("setup_slot_index must be SetupSlotIndex")
    if not isinstance(setup_attachment_index, SetupAttachmentNameIndex):
        raise TypeError(
            "setup_attachment_index must be SetupAttachmentNameIndex"
        )

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
            _require_name(slot_name, f"{slot_path} slot name")
            setup_slot_index.require(slot_name, path=slot_path)
            if not isinstance(slot_metadata, Mapping):
                raise TypeError(f"{slot_path} must be a mapping")

            if "attachment" not in slot_metadata:
                continue

            timeline = slot_metadata["attachment"]
            timeline_path = f"{slot_path}.attachment"
            if not isinstance(timeline, (list, tuple)):
                raise TypeError(f"{timeline_path} must be a list or tuple")
            if not timeline:
                raise ValueError(f"{timeline_path} cannot be empty")

            previous_time: float | int | None = None
            for keyframe_index, keyframe in enumerate(timeline):
                keyframe_path = f"{timeline_path}[{keyframe_index}]"
                if not isinstance(keyframe, Mapping):
                    raise TypeError(f"{keyframe_path} must be a mapping")

                time_value = keyframe.get("time", 0)
                if isinstance(time_value, bool) or not isinstance(
                    time_value,
                    (int, float),
                ):
                    raise TypeError(
                        f"{keyframe_path}.time must be a finite number"
                    )
                if not _is_finite_number(time_value):
                    raise ValueError(f"{keyframe_path}.time must be finite")
                if previous_time is not None and time_value < previous_time:
                    raise ValueError(
                        f"{keyframe_path}.time must be greater than or equal to "
                        f"the previous attachment time {previous_time}"
                    )
                previous_time = time_value

                if "name" not in keyframe or keyframe["name"] is None:
                    continue

                attachment_name = keyframe["name"]
                _require_name(attachment_name, f"{keyframe_path}.name")
                setup_attachment_index.require(
                    slot_name,
                    attachment_name,
                    path=f"{keyframe_path}.name",
                )


def validate_animation_model_contracts(
    animations: Mapping[str, Any],
    *,
    events: Mapping[str, Any],
    slot_names: tuple[str, ...],
    skin_attachments: tuple[Mapping[str, Mapping[str, Any]], ...],
    path: str = "document.animations",
    events_path: str = "document.events",
    setup_slot_index: SetupSlotIndex | None = None,
    setup_attachment_index: SetupAttachmentNameIndex | None = None,
) -> None:
    """Validate model-level animation payloads without mutation."""

    if not isinstance(path, str) or not path:
        raise ValueError("path must be a non-empty string")
    if not isinstance(events_path, str) or not events_path:
        raise ValueError("events_path must be a non-empty string")

    validate_json_mapping(animations, path=path)
    validate_json_mapping(events, path=events_path)
    _validate_event_definitions(events, path=events_path)
    _validate_animation_event_timelines(
        animations,
        event_definitions=events,
        path=path,
    )

    slot_index = resolve_setup_slot_index(slot_names, setup_slot_index)
    _validate_animation_draw_order_timelines(
        animations,
        setup_slot_index=slot_index,
        path=path,
    )

    attachment_index = resolve_setup_attachment_name_index(
        skin_attachments,
        setup_attachment_index,
    )
    _validate_animation_slot_attachment_timelines(
        animations,
        setup_slot_index=slot_index,
        setup_attachment_index=attachment_index,
        path=path,
    )


__all__ = ["validate_animation_model_contracts"]
