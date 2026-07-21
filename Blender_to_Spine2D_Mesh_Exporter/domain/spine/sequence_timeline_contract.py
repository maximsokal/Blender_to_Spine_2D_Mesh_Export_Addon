"""Strict Spine 4.2 attachment sequence timeline contract."""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from typing import Any

from .model import MeshAttachment, Skin
from .spine_json_contract import json_path_key


_SEQUENCE_MODES = frozenset(
    {
        "hold",
        "once",
        "loop",
        "pingpong",
        "onceReverse",
        "loopReverse",
        "pingpongReverse",
    }
)
_TEXTURE_REGION_ATTACHMENT_TYPES = frozenset({"region", "mesh", "linkedmesh"})

# SequenceTimeline packs ``mode | (index << 4)`` into single-precision frame
# storage. Every integer through 2**24 is represented exactly, so this bound
# preserves both the index and the low four mode bits across Spine runtimes.
_SEQUENCE_INDEX_MAX = ((1 << 24) - 1) >> 4


def _mapping_key_path(path: str, key: object) -> str:
    if not isinstance(key, str):
        raise TypeError(f"{path} keys must be str")
    return json_path_key(path, key)


def _require_name(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be str")
    if not value.strip():
        raise ValueError(f"{field_name} cannot be empty")
    return value


def _require_finite_number(value: object, field_name: str) -> float | int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a finite number")
    if isinstance(value, float) and not isfinite(value):
        raise ValueError(f"{field_name} must be finite")
    return value


def _build_skin_index(
    skins: tuple[Skin, ...],
) -> tuple[dict[str, Skin], set[str]]:
    if not isinstance(skins, tuple):
        raise TypeError("skins must be tuple")

    skin_by_name: dict[str, Skin] = {}
    ambiguous_skin_names: set[str] = set()
    for skin_index, skin in enumerate(skins):
        if not isinstance(skin, Skin):
            raise TypeError(f"skins[{skin_index}] must be Skin")
        if skin.name in skin_by_name:
            ambiguous_skin_names.add(skin.name)
        else:
            skin_by_name[skin.name] = skin
    return skin_by_name, ambiguous_skin_names


def _resolve_setup_attachment(
    *,
    skin_by_name: Mapping[str, Skin],
    ambiguous_skin_names: set[str],
    skin_name: str,
    slot_name: str,
    attachment_name: str,
    path: str,
) -> MeshAttachment | Mapping[str, Any]:
    if skin_name in ambiguous_skin_names:
        raise ValueError(f"{path} references duplicated skin '{skin_name}'")
    skin = skin_by_name.get(skin_name)
    if skin is None:
        raise ValueError(f"{path} references undefined skin '{skin_name}'")

    slot_attachments = skin.attachments.get(slot_name)
    if slot_attachments is None:
        raise ValueError(
            f"{path} references slot '{slot_name}' without attachments "
            f"in skin '{skin_name}'"
        )
    attachment = slot_attachments.get(attachment_name)
    if attachment is None:
        raise ValueError(
            f"{path} references undefined attachment '{attachment_name}' "
            f"for slot '{slot_name}' in skin '{skin_name}'"
        )
    if not isinstance(attachment, (MeshAttachment, Mapping)):
        raise TypeError(f"{path} setup attachment has an unsupported value type")
    return attachment


def _resolve_setup_sequence(
    attachment: MeshAttachment | Mapping[str, Any],
    *,
    path: str,
) -> tuple[Mapping[str, Any], int]:
    if isinstance(attachment, MeshAttachment):
        attachment_type = "mesh"
        sequence = attachment.sequence
    else:
        attachment_type = attachment.get("type", "region")
        if not isinstance(attachment_type, str):
            raise TypeError(f"{path}.type must be str")
        sequence = attachment.get("sequence")

    if attachment_type not in _TEXTURE_REGION_ATTACHMENT_TYPES:
        raise ValueError(
            f"{path} has non-sequence attachment type '{attachment_type}'"
        )
    if sequence is None:
        raise ValueError(f"{path}.sequence is required for a sequence timeline")
    if not isinstance(sequence, Mapping):
        raise TypeError(f"{path}.sequence must be a mapping")
    if "count" not in sequence:
        raise ValueError(f"{path}.sequence.count is required")

    count = sequence["count"]
    if isinstance(count, bool) or not isinstance(count, int):
        raise TypeError(f"{path}.sequence.count must be int")
    if count < 1:
        raise ValueError(
            f"{path}.sequence.count must be greater than or equal to 1"
        )
    return sequence, count


def _validate_sequence_timeline(
    timeline: object,
    *,
    path: str,
) -> None:
    if not isinstance(timeline, (list, tuple)):
        raise TypeError(f"{path} must be a list or tuple")
    if not timeline:
        raise ValueError(f"{path} cannot be empty")

    previous_time: float | int | None = None
    last_delay: float | int = 0

    for keyframe_index, keyframe in enumerate(timeline):
        keyframe_path = f"{path}[{keyframe_index}]"
        if not isinstance(keyframe, Mapping):
            raise TypeError(f"{keyframe_path} must be a mapping")

        time_value = _require_finite_number(
            keyframe.get("time", 0),
            f"{keyframe_path}.time",
        )
        if previous_time is not None and time_value < previous_time:
            raise ValueError(
                f"{keyframe_path}.time must be greater than or equal to "
                f"the previous sequence time {previous_time}"
            )
        previous_time = time_value

        mode = keyframe.get("mode", "hold")
        if not isinstance(mode, str):
            raise TypeError(f"{keyframe_path}.mode must be str")
        if mode not in _SEQUENCE_MODES:
            allowed = ", ".join(sorted(_SEQUENCE_MODES))
            raise ValueError(
                f"{keyframe_path}.mode must be one of: {allowed}"
            )

        index = keyframe.get("index", 0)
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError(f"{keyframe_path}.index must be int")
        if index < 0:
            raise ValueError(f"{keyframe_path}.index must be non-negative")
        if index > _SEQUENCE_INDEX_MAX:
            raise ValueError(
                f"{keyframe_path}.index must be less than or equal to "
                f"{_SEQUENCE_INDEX_MAX} for exact runtime frame packing"
            )

        if "delay" in keyframe:
            delay = _require_finite_number(
                keyframe["delay"],
                f"{keyframe_path}.delay",
            )
            if delay < 0:
                raise ValueError(
                    f"{keyframe_path}.delay must be non-negative"
                )
        else:
            delay = last_delay

        if mode != "hold" and delay <= 0:
            raise ValueError(
                f"{keyframe_path}.delay must resolve to a value greater than "
                f"0 for sequence mode '{mode}'"
            )
        last_delay = delay


def validate_animation_sequence_timelines(
    animations: Mapping[str, Any],
    *,
    skins: tuple[Skin, ...],
    slot_names: tuple[str, ...],
    path: str,
) -> None:
    """Validate Spine 4.2 ``animations.attachments`` sequence timelines.

    Runtime defaults and inherited delays are evaluated for validation only.
    Source mappings, omitted fields, unknown attachment timelines, and inert
    future fields are preserved without normalization.
    """

    if not isinstance(animations, Mapping):
        raise TypeError("animations must be a mapping")
    if not isinstance(slot_names, tuple):
        raise TypeError("slot_names must be tuple")
    if not isinstance(path, str) or not path:
        raise ValueError("path must be a non-empty string")

    skin_by_name, ambiguous_skin_names = _build_skin_index(skins)

    known_slot_names: set[str] = set()
    ambiguous_slot_names: set[str] = set()
    for slot_index, slot_name in enumerate(slot_names):
        _require_name(slot_name, f"slot_names[{slot_index}]")
        if slot_name in known_slot_names:
            ambiguous_slot_names.add(slot_name)
        known_slot_names.add(slot_name)

    for animation_name, animation_metadata in animations.items():
        animation_path = _mapping_key_path(path, animation_name)
        if not isinstance(animation_metadata, Mapping):
            raise TypeError(f"{animation_path} must be a mapping")
        if "attachments" not in animation_metadata:
            continue

        skin_timelines = animation_metadata["attachments"]
        attachments_path = f"{animation_path}.attachments"
        if not isinstance(skin_timelines, Mapping):
            raise TypeError(f"{attachments_path} must be a mapping")

        for skin_name, skin_metadata in skin_timelines.items():
            skin_path = _mapping_key_path(attachments_path, skin_name)
            _require_name(skin_name, f"{skin_path} skin name")
            if skin_name in ambiguous_skin_names:
                raise ValueError(
                    f"{skin_path} references duplicated skin '{skin_name}'"
                )
            if skin_name not in skin_by_name:
                raise ValueError(
                    f"{skin_path} references undefined skin '{skin_name}'"
                )
            if not isinstance(skin_metadata, Mapping):
                raise TypeError(f"{skin_path} must be a mapping")

            for slot_name, slot_metadata in skin_metadata.items():
                slot_path = _mapping_key_path(skin_path, slot_name)
                _require_name(slot_name, f"{slot_path} slot name")
                if slot_name in ambiguous_slot_names:
                    raise ValueError(
                        f"{slot_path} references duplicated setup slot "
                        f"'{slot_name}'"
                    )
                if slot_name not in known_slot_names:
                    raise ValueError(
                        f"{slot_path} references undefined slot '{slot_name}'"
                    )
                if not isinstance(slot_metadata, Mapping):
                    raise TypeError(f"{slot_path} must be a mapping")

                for attachment_name, attachment_metadata in slot_metadata.items():
                    attachment_path = _mapping_key_path(
                        slot_path,
                        attachment_name,
                    )
                    _require_name(
                        attachment_name,
                        f"{attachment_path} attachment name",
                    )
                    if not isinstance(attachment_metadata, Mapping):
                        raise TypeError(f"{attachment_path} must be a mapping")
                    if "sequence" not in attachment_metadata:
                        continue

                    attachment = _resolve_setup_attachment(
                        skin_by_name=skin_by_name,
                        ambiguous_skin_names=ambiguous_skin_names,
                        skin_name=skin_name,
                        slot_name=slot_name,
                        attachment_name=attachment_name,
                        path=attachment_path,
                    )
                    _resolve_setup_sequence(
                        attachment,
                        path=attachment_path,
                    )
                    _validate_sequence_timeline(
                        attachment_metadata["sequence"],
                        path=f"{attachment_path}.sequence",
                    )


__all__ = ["validate_animation_sequence_timelines"]
