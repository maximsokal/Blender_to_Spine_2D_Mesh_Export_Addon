"""Legacy A1 control bounding boxes and preview animation.

These values are part of the public v0.23 Spine contract. They are static control-shape
geometry, not model-specific mesh data. Keeping them in the Blender-independent Spine
domain allows Rewrite exports to honor the existing UI switches without importing or
mutating legacy JSON dictionaries.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import Mapping, Tuple

from .model import Skin, Slot, SpineDocument
from .validator import SpineValidator


_MAIN_VERTICES = (
    -21.11, 20.72, -20.96, 68.4, -43.49, 68.1, -1.72, 116.85,
    43.32, 68.33, 20.42, 68.33, 20.56, 20.66, 68.38, 20.83,
    68.11, 42.63, 117.11, -1.92, 68.53, -43.12, 68.19, -20.49,
    20.8, -20.65, 20.65, -68.22, 42.87, -68.6, -1.65, -116.52,
    -42.84, -68.86, -20.52, -68.5, -20.68, -21.13, -68.0, -21.09,
    -68.23, -43.28, -117.01, -1.99, -68.32, 43.2, -68.17, 20.58,
)

_ROTATION_X_VERTICES = (
    33.35, -86.18, 19.36, -57.86, 6.36, -62.25, -4.44, -64.09,
    -18.35, -63.9, -33.54, -59.51, -45.63, -52.92, -55.88, -43.95,
    -63.57, -33.51, -69.78, -20.21, -72.89, -6.3, -72.71, 7.43,
    -69.78, 21.89, -62.09, 35.44, -52.19, 47.35, -39.92, 56.13,
    -25.83, 61.81, -8.98, 63.46, 9.22, 61.63, 25.33, 53.75,
    37.77, 43.14, 48.39, 27.94, 53.34, 12.38, 54.98, -0.62,
    27.89, 25.93, 24.23, 25.93, 22.4, 23.18, 22.95, -10.5,
    23.86, -14.53, 65.6, -56.82, 69.63, -58.65, 73.47, -58.65,
    75.78, -56.46, 116.42, -15.08, 118.44, -11.24, 118.99, 23.55,
    116.42, 26.11, 112.21, 25.93, 86.4, -0.25, 86.09, 9.86,
    84.81, 22.13, 80.05, 35.13, 73.82, 48.31, 65.58, 60.94,
    53.88, 72.75, 39.97, 82.27, 25.87, 89.23, 8.3, 93.81,
    -9.28, 95.82, -26.57, 94.17, -44.33, 89.23, -61.54, 80.26,
    -78.02, 67.08, -91.56, 49.32, -101.17, 27.7, -105.19, 5.18,
    -103.91, -18.8, -96.22, -42.97, -85.53, -59.08, -69.42, -75.0,
    -52.44, -86.57, -24.8, -95.17, 0.46, -96.63, 24.45, -90.78,
)

_ROTATION_Y_VERTICES = (
    33.45, -86.17, 19.46, -57.86, 6.46, -62.25, -4.34, -64.08,
    -18.25, -63.9, -33.45, -59.5, -45.53, -52.91, -55.78, -43.94,
    -63.47, -33.51, -69.68, -20.21, -72.79, -6.29, -72.61, 7.44,
    -69.68, 21.9, -61.99, 35.45, -52.09, 47.35, -39.83, 56.14,
    -25.73, 61.81, -8.89, 63.46, 9.31, 61.63, 25.42, 53.76,
    37.87, 43.14, 48.49, 27.95, 53.43, 12.38, 55.08, -0.61,
    27.99, 25.93, 24.32, 25.93, 22.49, 23.19, 23.04, -10.5,
    23.96, -14.53, 65.7, -56.82, 69.73, -58.65, 73.57, -58.65,
    75.88, -56.45, 116.52, -15.08, 118.53, -11.23, 119.08, 23.55,
    116.52, 26.11, 112.31, 25.93, 86.5, -0.25, 86.19, 9.87,
    84.91, 22.13, 80.15, 35.13, 73.92, 48.31, 65.68, 60.94,
    53.98, 72.76, 40.07, 82.28, 25.97, 89.23, 8.4, 93.81,
    -9.18, 95.82, -26.48, 94.18, -44.23, 89.23, -61.44, 80.26,
    -77.92, 67.08, -91.47, 49.32, -101.07, 27.7, -105.09, 5.19,
    -103.81, -18.8, -96.12, -42.96, -85.43, -59.07, -69.32, -75.0,
    -52.35, -86.56, -24.7, -95.17, 0.56, -96.63, 24.54, -90.77,
)

_ROTATION_Z_VERTICES = (
    33.45, -86.17, 19.46, -57.86, 6.46, -62.25, -4.34, -64.08,
    -18.26, -63.9, -33.45, -59.5, -45.53, -52.91, -55.79, -43.94,
    -63.48, -33.51, -69.68, -20.21, -72.79, -6.29, -72.61, 7.44,
    -69.68, 21.9, -61.99, 35.45, -52.1, 47.35, -39.83, 56.14,
    -25.73, 61.81, -8.89, 63.46, 9.31, 61.63, 25.42, 53.76,
    37.87, 43.14, 48.49, 27.95, 53.43, 12.38, 55.08, -0.61,
    27.98, 25.93, 24.32, 25.93, 22.49, 23.19, 23.04, -10.5,
    23.96, -14.53, 65.7, -56.82, 69.72, -58.65, 73.57, -58.65,
    75.87, -56.45, 116.52, -15.08, 118.53, -11.23, 119.08, 23.55,
    116.52, 26.11, 112.31, 25.93, 86.49, -0.25, 86.18, 9.87,
    84.9, 22.13, 80.14, 35.13, 73.92, 48.31, 65.68, 60.94,
    53.98, 72.76, 40.06, 82.28, 25.97, 89.23, 8.39, 93.81,
    -9.18, 95.82, -26.48, 94.18, -44.24, 89.23, -61.45, 80.26,
    -77.92, 67.08, -91.47, 49.32, -101.07, 27.7, -105.1, 5.19,
    -103.82, -18.8, -96.13, -42.96, -85.44, -59.07, -69.33, -75.0,
    -52.35, -86.56, -24.71, -95.17, 0.56, -96.63, 24.54, -90.77,
)

_CONTROL_SHAPES: Mapping[str, Tuple[int, Tuple[float, ...], str]] = {
    "rotation_X": (64, _ROTATION_X_VERTICES, "ff0000ff"),
    "rotation_Z": (64, _ROTATION_Z_VERTICES, "002cffff"),
    "rotation_Y": (64, _ROTATION_Y_VERTICES, "00ff18ff"),
    "main": (24, _MAIN_VERTICES, "df00ffff"),
}


def build_legacy_control_slots_and_attachments(
    prefix: str,
) -> tuple[Tuple[Slot, ...], Mapping[str, Mapping[str, Mapping[str, object]]]]:
    """Build the exact v0.23 control slot order and bounding-box payloads."""

    if not isinstance(prefix, str) or not prefix.strip():
        raise ValueError("prefix must be a non-empty string")
    normalized = prefix.strip()
    slots: list[Slot] = []
    attachments: dict[str, dict[str, Mapping[str, object]]] = {}
    for suffix, (vertex_count, vertices, color) in _CONTROL_SHAPES.items():
        name = f"{normalized}_{suffix}"
        slots.append(Slot(name=name, bone=name, attachment=name))
        attachments[name] = {
            name: {
                "type": "boundingbox",
                "vertexCount": vertex_count,
                "vertices": list(vertices),
                "color": color,
            }
        }
    return tuple(slots), attachments


def build_legacy_preview_animation(prefix: str) -> Mapping[str, object]:
    """Return the exact deterministic v0.23 preview control timelines."""

    if not isinstance(prefix, str) or not prefix.strip():
        raise ValueError("prefix must be a non-empty string")
    normalized = prefix.strip()
    return {
        "bones": {
            f"{normalized}_rotation_Y": {
                "rotate": [
                    {"curve": [0.667, 0, 1.333, -360]},
                    {
                        "time": 2,
                        "value": -360,
                        "curve": [2.667, -360, 3.333, -360],
                    },
                    {
                        "time": 4,
                        "value": -360,
                        "curve": [4.667, -360, 5.333, 0],
                    },
                    {"time": 6},
                ]
            },
            f"{normalized}_rotation_Z": {
                "rotate": [
                    {"time": 2, "curve": [2.667, 0, 3.333, 360]},
                    {
                        "time": 4,
                        "value": 360,
                        "curve": [4.667, 360, 5.333, 0],
                    },
                    {"time": 6},
                ]
            },
            f"{normalized}_rotation_X": {
                "rotate": [
                    {"value": -360, "curve": [0.667, -360, 1.333, 0]},
                    {"time": 2, "curve": "stepped"},
                    {"time": 4, "curve": [4.667, 0, 5.333, -360]},
                    {
                        "time": 6,
                        "value": -360,
                        "curve": [6.667, -360, 7.333, -360],
                    },
                    {"time": 8, "value": -360},
                ]
            },
        }
    }


def apply_legacy_visual_options(
    document: SpineDocument,
    *,
    prefix: str,
    include_control_icons: bool,
    include_preview_animation: bool,
) -> SpineDocument:
    """Return a validated document with the requested legacy visual extras."""

    if not isinstance(document, SpineDocument):
        raise TypeError("document must be SpineDocument")
    if not isinstance(include_control_icons, bool):
        raise TypeError("include_control_icons must be bool")
    if not isinstance(include_preview_animation, bool):
        raise TypeError("include_preview_animation must be bool")

    result = document
    if include_control_icons:
        control_slots, control_attachments = (
            build_legacy_control_slots_and_attachments(prefix)
        )
        existing_slot_names = {slot.name for slot in document.slots}
        collisions = tuple(
            slot.name for slot in control_slots if slot.name in existing_slot_names
        )
        if collisions:
            raise ValueError(f"Control slot names collide with mesh slots: {collisions}")

        matching_skins = tuple(skin for skin in document.skins if skin.name == "default")
        if len(matching_skins) != 1:
            raise ValueError(
                f"Expected one default skin for control icons, found {len(matching_skins)}"
            )
        default_skin = matching_skins[0]
        merged_attachments = deepcopy(control_attachments)
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
        animations["preview"] = deepcopy(build_legacy_preview_animation(prefix))
        result = replace(result, animations=animations)

    SpineValidator().validate_or_raise(result)
    return result
