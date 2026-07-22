"""Strict Spine 4.2 deform timeline and attachment-capacity contract."""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from typing import Any

from .curve_timeline_contract import validate_curve_value
from .linked_mesh_contract import (
    AttachmentReference,
    LinkedMeshResolver,
    is_linked_mesh_attachment,
    raw_attachment_type,
)
from .model import MeshAttachment, Skin
from .spine_json_contract import json_path_key
from .weighted_vertices import decode_weighted_vertices


_VERTEX_ATTACHMENT_TYPES = frozenset(
    {"mesh", "linkedmesh", "boundingbox", "path", "clipping"}
)
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


def _deform_capacity_for_attachment(
    attachment: MeshAttachment | Mapping[str, Any],
    *,
    path: str,
) -> int:
    if isinstance(attachment, MeshAttachment):
        return _deform_capacity_from_vertices(
            attachment.vertices,
            expected_coordinate_count=len(attachment.uvs),
            path=f"{path}.vertices",
        )

    attachment_type = raw_attachment_type(attachment, path=path)
    if attachment_type not in _VERTEX_ATTACHMENT_TYPES:
        raise ValueError(
            f"{path} has non-deformable attachment type '{attachment_type}'"
        )
    if is_linked_mesh_attachment(attachment, path=path):
        raise RuntimeError(
            f"{path} linked mesh must be resolved before deform capacity calculation"
        )

    expected_coordinate_count = _raw_attachment_expected_coordinate_count(
        attachment,
        attachment_type=attachment_type,
        path=path,
    )
    if "vertices" not in attachment:
        raise ValueError(f"{path}.vertices is required")
    return _deform_capacity_from_vertices(
        attachment["vertices"],
        expected_coordinate_count=expected_coordinate_count,
        path=f"{path}.vertices",
    )


def _resolve_deform_capacity(
    *,
    resolver: LinkedMeshResolver,
    reference: AttachmentReference,
    path: str,
    cache: dict[AttachmentReference, int],
) -> int:
    cached = cache.get(reference)
    if cached is not None:
        return cached

    setup = resolver.get_attachment(reference, path=path)
    attachment = setup.attachment
    setup_path = setup.path
    capacity_reference = reference

    if isinstance(attachment, Mapping) and is_linked_mesh_attachment(
        attachment,
        path=setup.path,
    ):
        resolved = resolver.resolve(reference)
        attachment = resolved.terminal_attachment
        setup_path = resolved.terminal_path
        capacity_reference = resolved.terminal

        terminal_cached = cache.get(capacity_reference)
        if terminal_cached is not None:
            cache[reference] = terminal_cached
            return terminal_cached

    capacity = _deform_capacity_for_attachment(
        attachment,
        path=setup_path,
    )
    cache[capacity_reference] = capacity
    cache[reference] = capacity
    return capacity


def validate_animation_deform_timelines(
    animations: Mapping[str, Any],
    *,
    skins: tuple[Skin, ...],
    slot_names: tuple[str, ...],
    path: str,
    linked_mesh_resolver: LinkedMeshResolver | None = None,
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

    if linked_mesh_resolver is None:
        resolver = LinkedMeshResolver(skins, path="document.skins")
    else:
        if not isinstance(linked_mesh_resolver, LinkedMeshResolver):
            raise TypeError(
                "linked_mesh_resolver must be LinkedMeshResolver or None"
            )
        if linked_mesh_resolver.skins is not skins:
            raise ValueError(
                "linked_mesh_resolver must be built from the exact skins tuple"
            )
        resolver = linked_mesh_resolver

    known_slot_names: set[str] = set()
    ambiguous_slot_names: set[str] = set()
    for slot_index, slot_name in enumerate(slot_names):
        _require_name(slot_name, f"slot_names[{slot_index}]")
        if slot_name in known_slot_names:
            ambiguous_slot_names.add(slot_name)
        known_slot_names.add(slot_name)

    capacity_cache: dict[AttachmentReference, int] = {}
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
            resolver.require_skin(skin_name, path=skin_path)
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

                    reference = AttachmentReference(
                        skin_name=skin_name,
                        slot_name=slot_name,
                        attachment_name=attachment_name,
                    )
                    capacity = _resolve_deform_capacity(
                        resolver=resolver,
                        reference=reference,
                        path=attachment_path,
                        cache=capacity_cache,
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
