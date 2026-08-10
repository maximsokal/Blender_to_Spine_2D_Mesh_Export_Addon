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


class A1CameraLayerProjectionKind(str, Enum):
    """Projection semantics for one rigid camera-relative object layer.

    Perspective layers retain depth-dependent whole-object foreshortening. Orthographic
    layers retain camera-relative translation/parallax but explicitly disable automatic
    depth scale so object size remains independent of camera distance.
    """

    PERSPECTIVE = "PERSPECTIVE"
    ORTHOGRAPHIC = "ORTHOGRAPHIC"


class A1RigSetupPoseMode(str, Enum):
    """Control how generated controls and internal depth layers evaluate at setup.

    ``NORMALIZED_SINGLE`` is safe only when one object owns the whole Spine document.
    It keeps the visible main and X/Y controls neutral while moving the existing object
    placement into internal rig coordinates.

    ``PRESERVE_COMPOSITION`` retains the historical model-space setup used by connected
    composition and compatibility paths.

    ``PROJECTED_AXIS_NORMAL`` is used only for standalone signed-axis Normal / UV
    Segments. Source geometry and Blender Object Origin have already been projected into
    canonical U/V/depth space. The generated rig therefore keeps the ordinary deformable
    model-space hierarchy, IK schedule, scale targets, and depth mapping, while X/Y setup
    rotation calibration starts from the already-projected view. No camera-specific
    inverse bones are inserted, and vertex bones remain direct children of their ordinary
    depth bones so later X/Y/scale controls retain the historical deformation graph.

    ``CAMERA_VIEW_NORMAL`` is used only when Normal / UV Segments takes its setup view
    from Blender's active camera. The geometry has already been oriented and projected
    around Blender Object Origin, so historical non-zero setup rotation and depth-scale
    offsets must be neutral. Unlike ``PREPROJECTED_SCREEN``, this mode keeps the ordinary
    model-space hierarchy, the Object Origin pivot, and every per-vertex depth group.
    Active Camera additionally owns inverse setup children required by its camera-facing
    setup contract.

    ``PREPROJECTED_SCREEN`` is reserved for camera-projected geometry. It represents one
    complete object as a rigid camera-relative layer: camera space is zero, projected
    Blender Object Origin is stored on the internal base, and one Object-Origin depth
    group drives layer motion. ``A1CameraLayerProjectionKind`` determines whether
    depth-dependent whole-layer scaling is Perspective or disabled for Orthographic.
    It must never be used with per-vertex depth groups or ordinary model-space geometry.

    ``CAMERA_DEPTH_SURFACE`` is the multi-depth companion used only by Depth Camera
    Projection. The mesh is already camera-facing in setup, but each generated vertex
    retains an absolute camera-distance group for pseudo-3D deformation. Consequently,
    the historical model-space hierarchy is retained while legacy non-zero setup
    rotations and scale offsets are removed. This mode must not be used for ordinary
    Normal / UV Segments geometry or a rigid one-group camera layer.
    """

    NORMALIZED_SINGLE = "NORMALIZED_SINGLE"
    PRESERVE_COMPOSITION = "PRESERVE_COMPOSITION"
    PROJECTED_AXIS_NORMAL = "PROJECTED_AXIS_NORMAL"
    CAMERA_VIEW_NORMAL = "CAMERA_VIEW_NORMAL"
    PREPROJECTED_SCREEN = "PREPROJECTED_SCREEN"
    CAMERA_DEPTH_SURFACE = "CAMERA_DEPTH_SURFACE"


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
    "A1CameraLayerProjectionKind",
    "A1RigProfile",
    "A1RigSetupPoseMode",
    "resolve_a1_rig_profile",
]
