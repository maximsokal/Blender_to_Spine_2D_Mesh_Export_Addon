"""Supported selectable Spine rig profiles and setup-pose policies.

The enums are Blender-independent and are the single source of truth used by the UI,
application settings, rig router, builders, documentation tests, and serializers.
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


class A1RigSetupPoseMode(str, Enum):
    """Control how generated controls and internal depth layers evaluate at setup.

    ``NORMALIZED_SINGLE`` is safe only when one object owns the whole Spine document.
    It keeps the visible main and X/Y controls neutral while moving the existing object
    placement into internal rig coordinates.

    ``PRESERVE_COMPOSITION`` retains the historical model-space setup used by ordinary
    Normal / UV Segments exports and connected composition.

    ``PREPROJECTED_SCREEN`` is reserved for geometry that has already been projected to
    final camera-screen X/Y. Its default transform-constraint offsets evaluate to an
    identity screen-space pose while retaining the same live X/Y/Scale controls for later
    authoring. It must not be selected for ordinary model-space geometry.
    """

    NORMALIZED_SINGLE = "NORMALIZED_SINGLE"
    PRESERVE_COMPOSITION = "PRESERVE_COMPOSITION"
    PREPROJECTED_SCREEN = "PREPROJECTED_SCREEN"


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


__all__ = [
    "A1RigProfile",
    "A1RigSetupPoseMode",
    "resolve_a1_rig_profile",
]
