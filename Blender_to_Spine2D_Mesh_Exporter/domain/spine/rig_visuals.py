"""Profile-aware control attachments and preview animations."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import Mapping, Tuple

from .legacy_visuals import (
    apply_legacy_visual_options,
    build_legacy_control_slots_and_attachments,
)
from .model import Skin, Slot, SpineDocument
from .rig_profiles import A1RigProfile, resolve_a1_rig_profile
from .validator import SpineValidator


_SCALE_CONTROL_VERTICES = (
    -30.0,
    -30.0,
    30.0,
    -30.0,
    30.0,
    30.0,
    -30.0,
    30.0,
)


def build_two_axis_scale_control_slots_and_attachments(
    prefix: str,
) -> tuple[Tuple[Slot, ...], Mapping[str, Mapping[str, Mapping[str, object]]]]:
    """Reuse stable X/Y/main shapes and add one dedicated scale square."""

    if not isinstance(prefix, str) or not prefix.strip():
        raise ValueError("prefix must be a non-empty string")
    normalized = prefix.strip()
    legacy_slots, legacy_attachments = build_legacy_control_slots_and_attachments(
        normalized
    )
    selected_names = (
        f"{normalized}_rotation_X",
        f"{normalized}_rotation_Y",
        f"{normalized}_main",
    )
    slot_by_name = {slot.name: slot for slot in legacy_slots}
    missing = tuple(name for name in selected_names if name not in slot_by_name)
    if missing:
        raise ValueError(f"Legacy control shapes are missing required entries: {missing}")

    scale_name = f"{normalized}_scale"
    slots = (
        slot_by_name[selected_names[0]],
        slot_by_name[selected_names[1]],
        Slot(name=scale_name, bone=scale_name, attachment=scale_name),
        slot_by_name[selected_names[2]],
    )
    attachments = {
        name: deepcopy(dict(legacy_attachments[name])) for name in selected_names
    }
    attachments[scale_name] = {
        scale_name: {
            "type": "boundingbox",
            "vertexCount": 4,
            "vertices": list(_SCALE_CONTROL_VERTICES),
            "color": "abe323ff",
        }
    }
    return slots, attachments


def build_two_axis_scale_preview_animation(prefix: str) -> Mapping[str, object]:
    """Return one deterministic preview covering X, Y, and uniform scale together."""

    if not isinstance(prefix, str) or not prefix.strip():
        raise ValueError("prefix must be a non-empty string")
    normalized = prefix.strip()
    return {
        "bones": {
            f"{normalized}_rotation_X": {
                "rotate": [
                    {},
                    {"time": 2, "value": 360},
                ]
            },
            f"{normalized}_rotation_Y": {
                "rotate": [
                    {},
                    {"time": 2, "value": 360},
                ]
            },
            f"{normalized}_scale": {
                "scale": [
                    {"x": 1.0, "y": 1.0},
                    {"time": 0.5, "x": 1.5, "y": 1.5},
                    {"time": 1.0, "x": 0.75, "y": 0.75},
                    {"time": 1.5, "x": 1.5, "y": 1.5},
                    {"time": 2.0, "x": 1.0, "y": 1.0},
                ]
            },
        }
    }


def _apply_visual_payload(
    document: SpineDocument,
    *,
    control_slots: Tuple[Slot, ...],
    control_attachments: Mapping[str, Mapping[str, Mapping[str, object]]],
    preview_animation: Mapping[str, object],
    include_control_icons: bool,
    include_preview_animation: bool,
) -> SpineDocument:
    if not isinstance(document, SpineDocument):
        raise TypeError("document must be SpineDocument")
    for field_name, value in (
        ("include_control_icons", include_control_icons),
        ("include_preview_animation", include_preview_animation),
    ):
        if not isinstance(value, bool):
            raise TypeError(f"{field_name} must be bool")

    result = document
    if include_control_icons:
        existing_slot_names = {slot.name for slot in result.slots}
        collisions = tuple(
            slot.name for slot in control_slots if slot.name in existing_slot_names
        )
        if collisions:
            raise ValueError(f"Control slot names collide with mesh slots: {collisions}")

        matching_skins = tuple(skin for skin in result.skins if skin.name == "default")
        if len(matching_skins) != 1:
            raise ValueError(
                f"Expected one default skin for control icons, found {len(matching_skins)}"
            )
        default_skin = matching_skins[0]
        merged_attachments = deepcopy(dict(control_attachments))
        for slot_name, slot_attachments in default_skin.attachments.items():
            if slot_name in merged_attachments:
                raise ValueError(
                    f"Control attachment slot '{slot_name}' collides with mesh data"
                )
            merged_attachments[slot_name] = deepcopy(dict(slot_attachments))
        replacement_skin = replace(default_skin, attachments=merged_attachments)
        result = replace(
            result,
            slots=control_slots + result.slots,
            skins=tuple(
                replacement_skin if skin is default_skin else skin
                for skin in result.skins
            ),
        )

    if include_preview_animation:
        animations = deepcopy(dict(result.animations))
        animations["preview"] = deepcopy(dict(preview_animation))
        result = replace(result, animations=animations)

    SpineValidator().validate_or_raise(result)
    return result


def apply_rig_visual_options(
    document: SpineDocument,
    *,
    prefix: str,
    rig_profile: A1RigProfile | str,
    include_control_icons: bool,
    include_preview_animation: bool,
) -> SpineDocument:
    """Apply visual extras matching only controls generated by the selected rig."""

    profile = resolve_a1_rig_profile(rig_profile)
    if profile is A1RigProfile.THREE_AXIS_ROTATION:
        return apply_legacy_visual_options(
            document,
            prefix=prefix,
            include_control_icons=include_control_icons,
            include_preview_animation=include_preview_animation,
        )

    if profile is A1RigProfile.TWO_AXIS_ROTATION_SCALE:
        slots, attachments = build_two_axis_scale_control_slots_and_attachments(prefix)
        return _apply_visual_payload(
            document,
            control_slots=slots,
            control_attachments=attachments,
            preview_animation=build_two_axis_scale_preview_animation(prefix),
            include_control_icons=include_control_icons,
            include_preview_animation=include_preview_animation,
        )

    raise AssertionError(f"Unhandled rig profile: {profile}")


__all__ = [
    "apply_rig_visual_options",
    "build_two_axis_scale_control_slots_and_attachments",
    "build_two_axis_scale_preview_animation",
]
