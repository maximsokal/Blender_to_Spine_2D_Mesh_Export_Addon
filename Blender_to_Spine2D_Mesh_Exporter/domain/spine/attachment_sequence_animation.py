"""Build deterministic Spine animation timelines for sequence attachments.

Canonical assembly and a few compatibility callers still require the historical
per-frame timeline. Target finalization for Spine 4.1+ uses a compact looping timeline
with one duration boundary. Both modes share the same validation and merge pipeline.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace
from math import isfinite
from typing import Any

from .model import MeshAttachment, SpineDocument
from .sequence_timeline_contract import validate_animation_sequence_timelines
from .validator import SpineValidator


DEFAULT_SEQUENCE_FRAME_DELAY = 1.0 / 30.0
_SEQUENCE_TIME_DECIMALS = 6
_LEGACY_SEQUENCE_TIME_DECIMALS = 4


class AttachmentSequenceAnimationError(ValueError):
    """Raised when sequence setup data cannot produce one unambiguous timeline."""


def _sequence_mapping(attachment: object) -> Mapping[str, Any] | None:
    if isinstance(attachment, MeshAttachment):
        return attachment.sequence
    if isinstance(attachment, Mapping):
        if attachment.get("parent"):
            return None
        sequence = attachment.get("sequence")
        if sequence is None:
            return None
        if not isinstance(sequence, Mapping):
            raise TypeError("raw attachment sequence must be a mapping")
        return sequence
    return None


def _sequence_count(sequence: Mapping[str, Any], *, path: str) -> int:
    if "count" not in sequence:
        raise AttachmentSequenceAnimationError(f"{path}.count is required")
    value = sequence["count"]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{path}.count must be int")
    if value < 1:
        raise AttachmentSequenceAnimationError(
            f"{path}.count must be greater than or equal to 1"
        )
    return value


def _validated_frame_delay(frame_delay: float) -> float:
    if (
        isinstance(frame_delay, bool)
        or not isinstance(frame_delay, (int, float))
        or not isfinite(float(frame_delay))
        or float(frame_delay) <= 0.0
    ):
        raise ValueError("frame_delay must be a finite number greater than zero")
    return float(frame_delay)


def _build_legacy_per_frame_timeline(
    count: int,
    frame_delay: float,
) -> tuple[dict[str, object], ...]:
    """Return the historical one-key-per-frame Loop representation.

    Legacy callers first quantized the frame delay to four decimals and then advanced
    every key by that quantized value. Preserve that exact contract here; production
    target finalization uses ``TextureSequenceTiming`` and never relies on this branch.
    """

    delay = round(frame_delay, _LEGACY_SEQUENCE_TIME_DECIMALS)
    if delay <= 0.0:
        raise ValueError("frame_delay rounds to zero at legacy timeline precision")

    keyframes: list[dict[str, object]] = [
        {
            "mode": "loop",
            "delay": delay,
        }
    ]
    for frame_index in range(1, count):
        keyframes.append(
            {
                "time": round(
                    delay * frame_index,
                    _LEGACY_SEQUENCE_TIME_DECIMALS,
                ),
                "mode": "loop",
                "index": frame_index,
            }
        )
    return tuple(keyframes)


def build_attachment_sequence_timeline(
    count: int,
    *,
    frame_delay: float = DEFAULT_SEQUENCE_FRAME_DELAY,
    legacy_per_frame: bool = False,
) -> tuple[dict[str, object], ...]:
    """Build either compact target output or the historical compatibility timeline."""

    if isinstance(count, bool) or not isinstance(count, int):
        raise TypeError("count must be int")
    if count < 1:
        raise ValueError("count must be greater than or equal to 1")
    if not isinstance(legacy_per_frame, bool):
        raise TypeError("legacy_per_frame must be bool")

    resolved_delay = _validated_frame_delay(frame_delay)
    if legacy_per_frame:
        return _build_legacy_per_frame_timeline(count, resolved_delay)

    delay = round(resolved_delay, _SEQUENCE_TIME_DECIMALS)
    if delay <= 0.0:
        raise ValueError("frame_delay rounds to zero at Spine timeline precision")
    duration = round(resolved_delay * count, _SEQUENCE_TIME_DECIMALS)
    if duration <= 0.0:
        raise ValueError("sequence duration rounds to zero")

    return (
        {
            "time": 0.0,
            "mode": "loop",
            "index": 0,
            "delay": delay,
        },
        {
            "time": duration,
            "mode": "loop",
            "index": 0,
            "delay": delay,
        },
    )


def _merge_sequence_timeline(
    animations: dict[str, Any],
    *,
    animation_name: str,
    skin_name: str,
    slot_name: str,
    attachment_name: str,
    timeline: tuple[dict[str, object], ...],
) -> None:
    animation = animations.setdefault(animation_name, {})
    if not isinstance(animation, dict):
        raise AttachmentSequenceAnimationError(
            f"animations[{animation_name!r}] must be a mutable mapping"
        )
    attachments = animation.setdefault("attachments", {})
    if not isinstance(attachments, dict):
        raise AttachmentSequenceAnimationError(
            f"animations[{animation_name!r}].attachments must be a mapping"
        )
    skin_timelines = attachments.setdefault(skin_name, {})
    if not isinstance(skin_timelines, dict):
        raise AttachmentSequenceAnimationError(
            f"animations[{animation_name!r}].attachments[{skin_name!r}] "
            "must be a mapping"
        )
    slot_timelines = skin_timelines.setdefault(slot_name, {})
    if not isinstance(slot_timelines, dict):
        raise AttachmentSequenceAnimationError(
            f"sequence slot timeline {skin_name!r}/{slot_name!r} must be a mapping"
        )
    attachment_timeline = slot_timelines.setdefault(attachment_name, {})
    if not isinstance(attachment_timeline, dict):
        raise AttachmentSequenceAnimationError(
            "sequence attachment timeline must be a mapping; "
            f"skin={skin_name!r}, slot={slot_name!r}, attachment={attachment_name!r}"
        )

    existing = attachment_timeline.get("sequence")
    serialized_timeline = [dict(keyframe) for keyframe in timeline]
    if existing is None:
        attachment_timeline["sequence"] = serialized_timeline
        return
    if existing != serialized_timeline and existing != timeline:
        raise AttachmentSequenceAnimationError(
            "Refusing to overwrite a different attachment sequence timeline; "
            f"animation={animation_name!r}, skin={skin_name!r}, slot={slot_name!r}, "
            f"attachment={attachment_name!r}"
        )


def _resolved_slot_filter(
    slot_names: tuple[str, ...] | None,
) -> frozenset[str] | None:
    if slot_names is None:
        return None
    if not isinstance(slot_names, tuple):
        raise TypeError("slot_names must be a tuple or None")
    if not slot_names:
        raise ValueError("slot_names cannot be empty when supplied")
    if not all(isinstance(value, str) and value.strip() for value in slot_names):
        raise TypeError("slot_names must contain non-empty strings")
    normalized = tuple(value.strip() for value in slot_names)
    if len(normalized) != len(set(normalized)):
        raise ValueError("slot_names cannot contain duplicates")
    return frozenset(normalized)


def apply_attachment_sequence_animations(
    document: SpineDocument,
    *,
    animation_name: str = "animation",
    frame_delay: float = DEFAULT_SEQUENCE_FRAME_DELAY,
    slot_names: tuple[str, ...] | None = None,
    legacy_per_frame: bool = False,
) -> SpineDocument:
    """Add validated Loop timelines for selected sequence attachments.

    Existing equal timelines are retained, different timelines fail explicitly, and
    documents without matching sequence attachments are returned unchanged.
    ``legacy_per_frame`` is reserved for canonical compatibility boundaries; production
    target finalization uses the compact timeline.
    """

    if not isinstance(document, SpineDocument):
        raise TypeError("document must be SpineDocument")
    if not isinstance(animation_name, str) or not animation_name.strip():
        raise ValueError("animation_name must be a non-empty string")
    if not isinstance(legacy_per_frame, bool):
        raise TypeError("legacy_per_frame must be bool")
    slot_filter = _resolved_slot_filter(slot_names)
    if slot_filter is not None:
        setup_slot_names = {slot.name for slot in document.slots}
        unknown = tuple(sorted(slot_filter - setup_slot_names, key=str.casefold))
        if unknown:
            raise AttachmentSequenceAnimationError(
                f"slot_names reference unknown setup slots: {unknown}"
            )

    targets: list[tuple[str, str, str, int]] = []
    for skin in document.skins:
        for slot_name, attachments in skin.attachments.items():
            resolved_slot_name = str(slot_name)
            if slot_filter is not None and resolved_slot_name not in slot_filter:
                continue
            for attachment_name, attachment in attachments.items():
                sequence = _sequence_mapping(attachment)
                if sequence is None:
                    continue
                path = (
                    f"skins[{skin.name!r}].attachments[{resolved_slot_name!r}]"
                    f"[{str(attachment_name)!r}].sequence"
                )
                targets.append(
                    (
                        skin.name,
                        resolved_slot_name,
                        str(attachment_name),
                        _sequence_count(sequence, path=path),
                    )
                )

    if not targets:
        return document

    animations = deepcopy(dict(document.animations))
    for skin_name, slot_name, attachment_name, count in targets:
        _merge_sequence_timeline(
            animations,
            animation_name=animation_name.strip(),
            skin_name=skin_name,
            slot_name=slot_name,
            attachment_name=attachment_name,
            timeline=build_attachment_sequence_timeline(
                count,
                frame_delay=frame_delay,
                legacy_per_frame=legacy_per_frame,
            ),
        )

    result = replace(document, animations=animations)
    SpineValidator().validate_or_raise(result)
    validate_animation_sequence_timelines(
        result.animations,
        skins=result.skins,
        slot_names=tuple(slot.name for slot in result.slots),
        path="document.animations",
    )
    return result


__all__ = [
    "AttachmentSequenceAnimationError",
    "DEFAULT_SEQUENCE_FRAME_DELAY",
    "apply_attachment_sequence_animations",
    "build_attachment_sequence_timeline",
]
