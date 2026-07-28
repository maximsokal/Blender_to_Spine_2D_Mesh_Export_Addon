"""Supported selectable Spine rig profiles.

The enum is Blender-independent and is the single source of truth used by the UI,
application settings, rig router, documentation tests, and serializers.
"""

from __future__ import annotations

from enum import Enum


class A1RigProfile(str, Enum):
    """Stable profile identifiers persisted in Blender Scene settings."""

    THREE_AXIS_ROTATION = "LEGACY_ROTATABLE_MESH"
    TWO_AXIS_ROTATION_SCALE = "TWO_AXIS_ROTATION_SCALE"

    @property
    def label(self) -> str:
        if self is A1RigProfile.THREE_AXIS_ROTATION:
            return "3-Axis Rotation"
        return "2-Axis Rotation + Scale"

    @property
    def description(self) -> str:
        if self is A1RigProfile.THREE_AXIS_ROTATION:
            return "Current X/Y/Z pseudo-rotation rig"
        return "X/Y pseudo-rotation with an independent uniform scale control"


def resolve_a1_rig_profile(value: object) -> A1RigProfile:
    """Resolve one enum or persisted string without accepting silent fallbacks."""

    if isinstance(value, A1RigProfile):
        return value
    if not isinstance(value, str):
        raise TypeError("rig profile must be A1RigProfile or str")
    normalized = value.strip().upper()
    if not normalized:
        raise ValueError("rig profile cannot be empty")
    try:
        return A1RigProfile(normalized)
    except ValueError as exc:
        supported = tuple(profile.value for profile in A1RigProfile)
        raise ValueError(
            f"Unsupported rig profile {value!r}; supported={supported}"
        ) from exc


__all__ = ["A1RigProfile", "resolve_a1_rig_profile"]
