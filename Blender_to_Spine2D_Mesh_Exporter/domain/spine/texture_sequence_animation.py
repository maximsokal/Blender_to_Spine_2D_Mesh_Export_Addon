"""Finalize baked texture animation for each supported Spine target.

Spine 4.1+ uses one native sequence attachment and a compact looping sequence timeline.
Spine 3.8/4.0 receive one ordinary mesh attachment per baked image plus a stepped slot
attachment timeline. The canonical document is never patched after JSON serialization.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace
from typing import Any

from ..baking import TextureSequenceTiming
from .attachment_sequence_animation import apply_attachment_sequence_animations
from .model import MeshAttachment, Skin, Slot, SpineDocument
from .validator import SpineValidator
from .version_target import (
    SpineJsonTarget,
    SpineTextureAnimationEncoding,
    resolve_spine_json_target,
)


class TextureSequenceFinalizationError(ValueError):
    """Raised when canonical sequence data cannot map safely to one target."""


def _sequence_mapping(
    attachment: MeshAttachment | Mapping[str, Any],
) -> Mapping[str, Any] | None:
    if isinstance(attachment, MeshAttachment):
        return attachment.sequence
    if not isinstance(attachment, Mapping):
        return None
    value = attachment.get("sequence")
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError("raw attachment sequence must be a mapping")
    return value


def _sequence_integer(
    sequence: Mapping[str, Any],
    field_name: str,
    *,
    minimum: int,
) -> int:
    if field_name not in sequence:
        raise TextureSequenceFinalizationError(
            f"sequence.{field_name} is required"
        )
    value = sequence[field_name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"sequence.{field_name} must be int")
    if value < minimum:
        raise TextureSequenceFinalizationError(
            f"sequence.{field_name} must be greater than or equal to {minimum}"
        )
    return value


def _frame_token(frame_number: int, digits: int) -> str:
    if isinstance(frame_number, bool) or not isinstance(frame_number, int):
        raise TypeError("frame_number must be int")
    if isinstance(digits, bool) or not isinstance(digits, int):
        raise TypeError("digits must be int")
    if digits < 0:
        raise ValueError("digits must be non-negative")
    return str(frame_number) if digits == 0 else f"{frame_number:0{digits}d}"


def _merge_legacy_slot_timeline(
    animations: dict[str, Any],
    *,
    animation_name: str,
    slot_name: str,
    keyframes: tuple[dict[str, object], ...],
) -> None:
    animation = animations.setdefault(animation_name, {})
    if not isinstance(animation, dict):
        raise TextureSequenceFinalizationError(
            f"animations[{animation_name!r}] must be a mutable mapping"
        )
    slots = animation.setdefault("slots", {})
    if not isinstance(slots, dict):
        raise TextureSequenceFinalizationError(
            f"animations[{animation_name!r}].slots must be a mapping"
        )
    slot = slots.setdefault(slot_name, {})
    if not isinstance(slot, dict):
        raise TextureSequenceFinalizationError(
            f"slot animation {slot_name!r} must be a mapping"
        )
    serialized = [dict(keyframe) for keyframe in keyframes]
    existing = slot.get("attachment")
    if existing is None:
        slot["attachment"] = serialized
        return
    if existing != serialized and existing != keyframes:
        raise TextureSequenceFinalizationError(
            "Refusing to overwrite a different slot attachment timeline; "
            f"animation={animation_name!r}, slot={slot_name!r}"
        )


def _legacy_attachment_timeline(
    frame_names: tuple[str, ...],
    timing: TextureSequenceTiming,
) -> tuple[dict[str, object], ...]:
    if not isinstance(frame_names, tuple) or not frame_names:
        raise ValueError("frame_names must be a non-empty tuple")
    if not all(isinstance(value, str) and value for value in frame_names):
        raise TypeError("frame_names must contain non-empty strings")
    if not isinstance(timing, TextureSequenceTiming):
        raise TypeError("timing must be TextureSequenceTiming")

    keyframes = tuple(
        {
            "time": timing.time_for_frame_index(index),
            "name": attachment_name,
        }
        for index, attachment_name in enumerate(frame_names)
    )
    # Slot timelines have no native loop mode. A boundary key gives the animation its
    # complete duration and returns to frame zero when the animation itself is looped.
    return (
        *keyframes,
        {
            "time": timing.duration_for_frame_count(len(frame_names)),
            "name": frame_names[0],
        },
    )


def _with_sequence_fps(
    document: SpineDocument,
    timing: TextureSequenceTiming,
) -> SpineDocument:
    skeleton = dict(document.skeleton)
    skeleton["fps"] = round(timing.resolved_fps, 6)
    return replace(document, skeleton=skeleton)


def _finalize_attachment_swap_sequences(
    document: SpineDocument,
    timing: TextureSequenceTiming,
    *,
    animation_name: str,
) -> SpineDocument:
    slots_by_name = {slot.name: slot for slot in document.slots}
    if len(slots_by_name) != len(document.slots):
        raise TextureSequenceFinalizationError("document contains duplicate slot names")

    sequence_frames: dict[tuple[str, str], tuple[str, ...]] = {}
    generated_primary: dict[tuple[str, str], str] = {}
    expanded_skins: list[Skin] = []

    for skin in document.skins:
        expanded_groups: dict[str, dict[str, MeshAttachment | Mapping[str, Any]]] = {}
        for slot_name, attachments in skin.attachments.items():
            expanded: dict[str, MeshAttachment | Mapping[str, Any]] = {}
            for attachment_name, attachment in attachments.items():
                sequence = _sequence_mapping(attachment)
                if sequence is None:
                    if attachment_name in expanded:
                        raise TextureSequenceFinalizationError(
                            f"duplicate attachment name {attachment_name!r}"
                        )
                    expanded[attachment_name] = attachment
                    continue
                if not isinstance(attachment, MeshAttachment):
                    raise TextureSequenceFinalizationError(
                        "Legacy attachment-swap finalization requires typed "
                        f"MeshAttachment values; skin={skin.name!r}, "
                        f"slot={slot_name!r}, attachment={attachment_name!r}"
                    )

                count = _sequence_integer(sequence, "count", minimum=1)
                start = _sequence_integer(sequence, "start", minimum=0)
                digits = _sequence_integer(sequence, "digits", minimum=0)
                if attachment.path is None or not attachment.path:
                    raise TextureSequenceFinalizationError(
                        f"sequence attachment {attachment_name!r} has no image path"
                    )
                path_prefix = attachment.path
                frame_names: list[str] = []
                for frame_index in range(count):
                    token = _frame_token(start + frame_index, digits)
                    frame_name = f"{attachment_name}_{token}"
                    frame_path = f"{path_prefix}{token}"
                    if frame_name in expanded:
                        raise TextureSequenceFinalizationError(
                            "generated frame attachment collides with an existing name: "
                            f"{frame_name!r}"
                        )
                    expanded[frame_name] = replace(
                        attachment,
                        name=frame_name,
                        path=frame_path,
                        sequence=None,
                    )
                    frame_names.append(frame_name)

                key = (str(slot_name), str(attachment_name))
                resolved_names = tuple(frame_names)
                existing_names = sequence_frames.get(key)
                if existing_names is not None and existing_names != resolved_names:
                    raise TextureSequenceFinalizationError(
                        "sequence frame names differ across skins for "
                        f"slot={slot_name!r}, attachment={attachment_name!r}"
                    )
                sequence_frames[key] = resolved_names
                generated_primary[key] = resolved_names[0]
            expanded_groups[str(slot_name)] = expanded
        expanded_skins.append(replace(skin, attachments=expanded_groups))

    if not sequence_frames:
        return document

    resolved_slots: list[Slot] = []
    animated_keys: set[tuple[str, str]] = set()
    for slot in document.slots:
        if slot.attachment is None:
            resolved_slots.append(slot)
            continue
        key = (slot.name, slot.attachment)
        primary = generated_primary.get(key)
        if primary is None:
            resolved_slots.append(slot)
            continue
        resolved_slots.append(replace(slot, attachment=primary))
        animated_keys.add(key)

    orphaned = tuple(
        sorted(
            set(sequence_frames) - animated_keys,
            key=lambda value: (value[0].casefold(), value[1].casefold()),
        )
    )
    if orphaned:
        raise TextureSequenceFinalizationError(
            "Sequence attachments must be active setup attachments before legacy "
            f"conversion; orphaned={orphaned}"
        )

    animations = deepcopy(dict(document.animations))
    for slot_name, attachment_name in sorted(
        animated_keys,
        key=lambda value: (value[0].casefold(), value[1].casefold()),
    ):
        _merge_legacy_slot_timeline(
            animations,
            animation_name=animation_name,
            slot_name=slot_name,
            keyframes=_legacy_attachment_timeline(
                sequence_frames[(slot_name, attachment_name)],
                timing,
            ),
        )

    result = replace(
        document,
        slots=tuple(resolved_slots),
        skins=tuple(expanded_skins),
        animations=animations,
    )
    result = _with_sequence_fps(result, timing)
    SpineValidator().validate_or_raise(result)
    return result


def finalize_texture_sequence_animation(
    document: SpineDocument,
    *,
    target: object,
    timing: TextureSequenceTiming,
    animation_name: str = "animation",
) -> SpineDocument:
    """Return a target-safe looping texture animation without mutating ``document``."""

    if not isinstance(document, SpineDocument):
        raise TypeError("document must be SpineDocument")
    if not isinstance(timing, TextureSequenceTiming):
        raise TypeError("timing must be TextureSequenceTiming")
    if not isinstance(animation_name, str) or not animation_name.strip():
        raise ValueError("animation_name must be a non-empty string")

    resolved_target: SpineJsonTarget = resolve_spine_json_target(target)
    has_sequence = any(
        _sequence_mapping(attachment) is not None
        for skin in document.skins
        for attachments in skin.attachments.values()
        for attachment in attachments.values()
    )
    if not has_sequence:
        return document

    encoding = resolved_target.texture_animation_encoding
    if encoding is SpineTextureAnimationEncoding.NATIVE_SEQUENCE:
        result = apply_attachment_sequence_animations(
            document,
            animation_name=animation_name.strip(),
            frame_delay=timing.frame_duration,
        )
        result = _with_sequence_fps(result, timing)
        SpineValidator().validate_or_raise(result)
        return result
    if encoding is SpineTextureAnimationEncoding.ATTACHMENT_SWAP:
        return _finalize_attachment_swap_sequences(
            document,
            timing,
            animation_name=animation_name.strip(),
        )
    raise AssertionError(f"Unhandled texture animation encoding: {encoding}")


__all__ = [
    "TextureSequenceFinalizationError",
    "finalize_texture_sequence_animation",
]
