"""Reusable model-level Spine animation validation before serialization."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .model import (
    _validate_animation_draw_order_timelines,
    _validate_animation_event_timelines,
    _validate_animation_slot_attachment_timelines,
    _validate_event_definitions,
)
from .setup_attachment_contract import (
    SetupAttachmentNameIndex,
    resolve_setup_attachment_name_index,
)
from .setup_slot_contract import SetupSlotIndex, resolve_setup_slot_index
from .spine_json_contract import validate_json_mapping


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
    """Revalidate mutable model-level animation payloads without mutation.

    ``SpineDocument`` validates these contracts during construction, but its nested
    mappings and sequences may remain mutable. Serializer callers use this boundary
    to reject semantic changes made after ``__post_init__`` while reusing the exact
    setup indexes already built for the output pipeline.
    """

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
