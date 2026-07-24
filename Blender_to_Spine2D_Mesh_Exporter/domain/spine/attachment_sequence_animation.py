"""Build deterministic Spine 4.2 animation timelines for sequence attachments.

A setup ``sequence`` mapping tells Spine how image files are named, but it does not by
itself animate the displayed sequence index. The legacy exporter generated one
``animations.attachments`` timeline for every sequence mesh. Rewrite keeps that visible
contract while making ownership and conflict handling explicit and Blender-independent.
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


DEFAULT_SEQUENCE_FRAME_DELAY = 0.0333


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


def build_attachment_sequence_timeline(
    count: int,
    *,
    frame_delay: float = DEFAULT_SEQUENCE_FRAME_DELAY,
) -> tuple[dict[str, object], ...]:
    """Return the deterministic v0.23 loop timeline for one sequence attachment."""

    if isinstance(count, bool) or not isinstance(count, int):
        raise TypeError("count must be int")
    if count < 1:
        raise ValueError("count must be greater than or equal to 1")
    if (
        isinstance(frame_delay, bool)
        or not isinstance(frame_delay, (int, float))
        or not isfinite(float(frame_delay))
        or float(frame_delay) <= 0.0
    ):
        raise ValueError("frame_delay must be a finite number greater than zero")

    delay = round(float(frame_delay), 4)
    if delay <= 0.0:
        raise ValueError("frame_delay rounds to zero at Spine timeline precision")
    keyframes: list[dict[str, object]] = [
        {"mode": "loop", "delay": delay},
    ]
    keyframes.extend(
        {
            "time": round(delay * frame_index, 4),
            "mode": "loop",
            "index": frame_index,
        }
        for frame_index in range(1, count)
    )
    return tuple(keyframes)


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


def apply_attachment_sequence_animations(
    document: SpineDocument,
    *,
    animation_name: str = "animation",
    frame_delay: float = DEFAULT_SEQUENCE_FRAME_DELAY,
) -> SpineDocument:
    """Add missing sequence timelines for every setup sequence attachment.

    Existing equal timelines are retained, different timelines fail explicitly, and
    documents without sequence attachments are returned unchanged.
    """

    if not isinstance(document, SpineDocument):
        raise TypeError("document must be SpineDocument")
    if not isinstance(animation_name, str) or not animation_name.strip():
        raise ValueError("animation_name must be a non-empty string")

    targets: list[tuple[str, str, str, int]] = []
    for skin in document.skins:
        for slot_name, attachments in skin.attachments.items():
            for attachment_name, attachment in attachments.items():
                sequence = _sequence_mapping(attachment)
                if sequence is None:
                    continue
                path = (
                    f"skins[{skin.name!r}].attachments[{str(slot_name)!r}]"
                    f"[{str(attachment_name)!r}].sequence"
                )
                targets.append(
                    (
                        skin.name,
                        str(slot_name),
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
