"""Strict Spine 4.2 deform timeline and attachment-capacity contract."""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from typing import Any

from .curve_timeline_contract import validate_curve_value
from .model import MeshAttachment, Skin
from .spine_json_contract import json_path_key
from .weighted_vertices import decode_weighted_vertices


_VERTEX_ATTACHMENT_TYPES = frozenset({"mesh", "linkedmesh", "boundingbox", "path", "clipping"})
_VERTEX_COUNT_ATTACHMENT_TYPES = frozenset({"boundingbox", "path", "clipping"})


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


def _require_finite_number(value: object, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a finite number")
    if isinstance(value, float) and not isfinite(value):
        raise ValueError(f"{field_name} must be finite")


def _require_number_sequence(value: object, field_name: str) -> list | tuple:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{field_name} must be a list or tuple")
    for value_index, component in enumerate(value):
        _require_finite_number(component, f"{field_name}[{value_index}]")
    return value


def _require_non_negative_even_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be int")
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")
    if value % 2:
        raise ValueError(f"{field_name} must preserve X/Y pair alignment")
    return value


def _deform_capacity_from_vertices(
    vertices: object,
    *,
    expected_coordinate_count: int,
    path: str,
) -> int:
    if (
        isinstance(expected_coordinate_count, bool)
        or not isinstance(expected_coordinate_count, int)
        or expected_coordinate_count < 0
        or expected_coordinate_count % 2
    ):
        raise ValueError(
            f"{path}: expected coordinate count must be a non-negative even integer"
        )

    stream = _require_number_sequence(vertices, path)
    if len(stream) == expected_coordinate_count:
        return expected_coordinate_count

    decoded = decode_weighted_vertices(
        stream,
        expected_vertex_count=expected_coordinate_count // 2,
    )
    return sum(len(vertex.influences) for vertex in decoded) * 2


def _raw_attachment_expected_coordinate_count(
    attachment: Mapping[str, Any],
    *,
    attachment_type: str,
    path: str,
) -> int:
    if attachment_type in {"mesh", "linkedmesh"}:
        if "uvs" not in attachment:
            raise ValueError(f"{path}.uvs is required for an unlinked mesh")
        uvs = _require_number_sequence(attachment["uvs"], f"{path}.uvs")
        if len(uvs) % 2:
            raise ValueError(f"{path}.uvs must contain U/V pairs")
        return len(uvs)

    if attachment_type in _VERTEX_COUNT_ATTACHMENT_TYPES:
        if "vertexCount" not in attachment:
            raise ValueError(f"{path}.vertexCount is required")
        vertex_count = attachment["vertexCount"]
        if isinstance(vertex_count, bool) or not isinstance(vertex_count, int):
            raise TypeError(f"{path}.vertexCount must be int")
        if vertex_count < 0:
            raise ValueError(f"{path}.vertexCount must be non-negative")
        return vertex_count * 2

    raise ValueError(
        f"{path} has non-deformable attachment type '{attachment_type}'"
    )


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


def _resolve_attachment(
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


def _resolve_deform_capacity(
    *,
    skin_by_name: Mapping[str, Skin],
    ambiguous_skin_names: set[str],
    skin_name: str,
    slot_name: str,
    attachment_name: str,
    path: str,
    cache: dict[tuple[str, str, str], int],
    resolving: set[tuple[str, str, str]],
) -> int:
    key = (skin_name, slot_name, attachment_name)
    cached = cache.get(key)
    if cached is not None:
        return cached
    if key in resolving:
        raise ValueError(f"{path} participates in a linked mesh parent cycle")

    resolving.add(key)
    try:
        attachment = _resolve_attachment(
            skin_by_name=skin_by_name,
            ambiguous_skin_names=ambiguous_skin_names,
            skin_name=skin_name,
            slot_name=slot_name,
            attachment_name=attachment_name,
            path=path,
        )

        if isinstance(attachment, MeshAttachment):
            capacity = _deform_capacity_from_vertices(
                attachment.vertices,
                expected_coordinate_count=len(attachment.uvs),
                path=f"{path}.vertices",
            )
        else:
            attachment_type = attachment.get("type", "region")
            if not isinstance(attachment_type, str):
                raise TypeError(f"{path}.type must be str")
            if attachment_type not in _VERTEX_ATTACHMENT_TYPES:
                raise ValueError(
                    f"{path} has non-deformable attachment type "
                    f"'{attachment_type}'"
                )

            parent = attachment.get("parent")
            if attachment_type in {"mesh", "linkedmesh"} and parent is not None:
                parent_name = _require_name(parent, f"{path}.parent")
                raw_parent_skin = attachment.get("skin")
                if raw_parent_skin in (None, ""):
                    parent_skin_name = "default"
                else:
                    parent_skin_name = _require_name(
                        raw_parent_skin,
                        f"{path}.skin",
                    )
                capacity = _resolve_deform_capacity(
                    skin_by_name=skin_by_name,
                    ambiguous_skin_names=ambiguous_skin_names,
                    skin_name=parent_skin_name,
                    slot_name=slot_name,
                    attachment_name=parent_name,
                    path=(
                        f"{path}.parent[{parent_skin_name!r}, "
                        f"{slot_name!r}, {parent_name!r}]"
                    ),
                    cache=cache,
                    resolving=resolving,
                )
            else:
                expected_coordinate_count = (
                    _raw_attachment_expected_coordinate_count(
                        attachment,
                        attachment_type=attachment_type,
                        path=path,
                    )
                )
                if "vertices" not in attachment:
                    raise ValueError(f"{path}.vertices is required")
                capacity = _deform_capacity_from_vertices(
                    attachment["vertices"],
                    expected_coordinate_count=expected_coordinate_count,
                    path=f"{path}.vertices",
                )

        cache[key] = capacity
        return capacity
    finally:
        resolving.remove(key)


def validate_animation_deform_timelines(
    animations: Mapping[str, Any],
    *,
    skins: tuple[Skin, ...],
    slot_names: tuple[str, ...],
    path: str,
) -> None:
    """Validate Spine 4.2 ``animations.attachments`` deform timelines.

    The function validates only consumed 4.2 deform data. Unknown attachment
    timeline kinds and inert fields are preserved without normalization.
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

    capacity_cache: dict[tuple[str, str, str], int] = {}
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
                    if "deform" not in attachment_metadata:
                        continue

                    capacity = _resolve_deform_capacity(
                        skin_by_name=skin_by_name,
                        ambiguous_skin_names=ambiguous_skin_names,
                        skin_name=skin_name,
                        slot_name=slot_name,
                        attachment_name=attachment_name,
                        path=attachment_path,
                        cache=capacity_cache,
                        resolving=set(),
                    )

                    timeline = attachment_metadata["deform"]
                    timeline_path = f"{attachment_path}.deform"
                    if not isinstance(timeline, (list, tuple)):
                        raise TypeError(
                            f"{timeline_path} must be a list or tuple"
                        )
                    if not timeline:
                        raise ValueError(f"{timeline_path} cannot be empty")

                    previous_time: float | int | None = None
                    last_keyframe_index = len(timeline) - 1
                    for keyframe_index, keyframe in enumerate(timeline):
                        keyframe_path = f"{timeline_path}[{keyframe_index}]"
                        if not isinstance(keyframe, Mapping):
                            raise TypeError(
                                f"{keyframe_path} must be a mapping"
                            )

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
                                f"equal to the previous deform time "
                                f"{previous_time}"
                            )
                        previous_time = time_value

                        if "vertices" in keyframe:
                            deform_vertices = _require_number_sequence(
                                keyframe["vertices"],
                                f"{keyframe_path}.vertices",
                            )
                            if len(deform_vertices) % 2:
                                raise ValueError(
                                    f"{keyframe_path}.vertices must contain "
                                    "X/Y pairs"
                                )
                            offset = _require_non_negative_even_int(
                                keyframe.get("offset", 0),
                                f"{keyframe_path}.offset",
                            )
                            end = offset + len(deform_vertices)
                            if end > capacity:
                                raise ValueError(
                                    f"{keyframe_path}.vertices range "
                                    f"[{offset}, {end}) exceeds deform capacity "
                                    f"{capacity}"
                                )

                        # Spine consumes a curve only when a next keyframe exists.
                        if (
                            "curve" in keyframe
                            and keyframe_index < last_keyframe_index
                        ):
                            validate_curve_value(
                                keyframe["curve"],
                                channel_count=1,
                                path=f"{keyframe_path}.curve",
                            )


__all__ = ["validate_animation_deform_timelines"]
