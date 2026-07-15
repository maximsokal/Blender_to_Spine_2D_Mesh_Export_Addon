"""Typed, Blender-independent model of the Spine JSON subset used by the add-on."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Tuple

JsonMapping = Mapping[str, Any]


def _require_name(value: str, field_name: str = "name") -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be str")
    if not value.strip():
        raise ValueError(f"{field_name} cannot be empty")


@dataclass(frozen=True, slots=True)
class Bone:
    name: str
    parent: str | None = None
    length: float | None = None
    x: float | None = None
    y: float | None = None
    rotation: float | None = None
    scale_x: float | None = None
    scale_y: float | None = None
    color: str | None = None
    icon: str | None = None
    extras: JsonMapping = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_name(self.name)
        if self.parent is not None:
            _require_name(self.parent, "parent")
        if self.parent == self.name:
            raise ValueError(f"Bone '{self.name}' cannot parent itself")


@dataclass(frozen=True, slots=True)
class Slot:
    name: str
    bone: str
    attachment: str | None = None
    color: str | None = None
    blend: str | None = None
    extras: JsonMapping = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_name(self.name)
        _require_name(self.bone, "bone")


@dataclass(frozen=True, slots=True)
class IKConstraint:
    name: str
    order: int
    bones: Tuple[str, ...]
    target: str
    extras: JsonMapping = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_name(self.name)
        if not isinstance(self.order, int) or self.order < 0:
            raise ValueError("IK constraint order must be a non-negative integer")
        if not self.bones:
            raise ValueError(f"IK constraint '{self.name}' must reference at least one bone")
        for bone in self.bones:
            _require_name(bone, "bones item")
        _require_name(self.target, "target")


@dataclass(frozen=True, slots=True)
class TransformConstraint:
    name: str
    order: int
    bones: Tuple[str, ...]
    target: str
    extras: JsonMapping = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_name(self.name)
        if not isinstance(self.order, int) or self.order < 0:
            raise ValueError("Transform constraint order must be a non-negative integer")
        if not self.bones:
            raise ValueError(
                f"Transform constraint '{self.name}' must reference at least one bone"
            )
        for bone in self.bones:
            _require_name(bone, "bones item")
        _require_name(self.target, "target")


@dataclass(frozen=True, slots=True)
class MeshAttachment:
    name: str
    uvs: Tuple[float, ...]
    triangles: Tuple[int, ...]
    vertices: Tuple[float | int, ...]
    hull: int
    path: str | None = None
    edges: Tuple[int, ...] = ()
    width: float | None = None
    height: float | None = None
    sequence: JsonMapping | None = None
    extras: JsonMapping = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_name(self.name)
        if len(self.uvs) % 2 != 0:
            raise ValueError(f"Mesh attachment '{self.name}' has an odd UV array length")
        if len(self.triangles) % 3 != 0:
            raise ValueError(
                f"Mesh attachment '{self.name}' triangle array length must be divisible by 3"
            )
        if not isinstance(self.hull, int) or self.hull < 0:
            raise ValueError("hull must be a non-negative integer")
        if len(self.edges) % 2 != 0:
            raise ValueError("edges must contain vertex index pairs")


@dataclass(frozen=True, slots=True)
class Skin:
    name: str
    attachments: Mapping[str, Mapping[str, MeshAttachment | JsonMapping]]
    bones: Tuple[str, ...] = ()
    constraints: Tuple[str, ...] = ()
    extras: JsonMapping = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_name(self.name)
        for slot_name, slot_attachments in self.attachments.items():
            _require_name(slot_name, "skin slot name")
            if not isinstance(slot_attachments, Mapping):
                raise TypeError("skin attachment groups must be mappings")


@dataclass(frozen=True, slots=True)
class SpineDocument:
    skeleton: JsonMapping
    bones: Tuple[Bone, ...]
    slots: Tuple[Slot, ...]
    skins: Tuple[Skin, ...]
    ik: Tuple[IKConstraint, ...] = ()
    transform: Tuple[TransformConstraint, ...] = ()
    animations: JsonMapping = field(default_factory=dict)
    events: JsonMapping = field(default_factory=dict)
    extras: JsonMapping = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.skeleton, Mapping):
            raise TypeError("skeleton must be a mapping")
        if not self.bones:
            raise ValueError("SpineDocument must contain at least one bone")
