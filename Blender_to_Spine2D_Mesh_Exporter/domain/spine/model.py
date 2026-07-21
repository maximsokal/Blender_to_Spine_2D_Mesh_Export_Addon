"""Typed, Blender-independent model of the Spine JSON subset used by the add-on."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from re import fullmatch
from typing import Any, Mapping, Tuple

from .spine_json_contract import validate_json_mapping

JsonMapping = Mapping[str, Any]


def _require_name(value: str, field_name: str = "name") -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be str")
    if not value.strip():
        raise ValueError(f"{field_name} cannot be empty")


def _require_optional_string(value: str | None, field_name: str) -> None:
    if value is not None and not isinstance(value, str):
        raise TypeError(f"{field_name} must be str or None")


def _is_finite_number(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return isinstance(value, int) or isfinite(value)


def _require_finite_number(value: float | int | None, field_name: str) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a finite number or None")
    if not _is_finite_number(value):
        raise ValueError(f"{field_name} must be finite")


def _require_non_negative_int(value: int, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be int")
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")


def _require_tuple(value: object, field_name: str) -> tuple:
    if not isinstance(value, tuple):
        raise TypeError(f"{field_name} must be tuple")
    return value


def _validate_extras(
    extras: JsonMapping,
    *,
    path: str,
    known_fields: tuple[str, ...],
) -> None:
    validate_json_mapping(extras, path=path)
    collisions = tuple(sorted(set(known_fields).intersection(extras)))
    if collisions:
        raise ValueError(
            f"{path}: extras cannot overwrite known fields: {', '.join(collisions)}"
        )


def _validate_attachment_metadata(
    metadata: Mapping[str, Any],
    *,
    path: str,
) -> None:
    """Validate common optional attachment fields consumed as strings by runtimes."""

    if "name" in metadata:
        _require_name(metadata["name"], f"{path}.name")

    if "path" in metadata and not isinstance(metadata["path"], str):
        raise TypeError(f"{path}.path must be str")

    if "color" in metadata:
        color = metadata["color"]
        if not isinstance(color, str):
            raise TypeError(f"{path}.color must be str")
        normalized = color[1:] if color.startswith("#") else color
        if (
            len(normalized) not in (6, 8)
            or fullmatch(r"[0-9A-Fa-f]+", normalized) is None
        ):
            raise ValueError(
                f"{path}.color must contain 6 or 8 hexadecimal digits"
            )


def _validate_finite_sequence(values: tuple, field_name: str) -> None:
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{field_name}[{index}] must be a finite number")
        if not _is_finite_number(value):
            raise ValueError(f"{field_name}[{index}] must be finite")


def _validate_index_sequence(values: tuple, field_name: str) -> None:
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{field_name}[{index}] must be int")
        if value < 0:
            raise ValueError(f"{field_name}[{index}] must be non-negative")


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
        for field_name in (
            "length",
            "x",
            "y",
            "rotation",
            "scale_x",
            "scale_y",
        ):
            _require_finite_number(getattr(self, field_name), field_name)
        _require_optional_string(self.color, "color")
        _require_optional_string(self.icon, "icon")
        _validate_extras(
            self.extras,
            path="bone.extras",
            known_fields=(
                "name",
                "parent",
                "length",
                "x",
                "y",
                "rotation",
                "scaleX",
                "scaleY",
                "color",
                "icon",
            ),
        )


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
        _require_optional_string(self.attachment, "attachment")
        _require_optional_string(self.color, "color")
        _require_optional_string(self.blend, "blend")
        _validate_extras(
            self.extras,
            path="slot.extras",
            known_fields=("name", "bone", "attachment", "color", "blend"),
        )


@dataclass(frozen=True, slots=True)
class IKConstraint:
    name: str
    order: int
    bones: Tuple[str, ...]
    target: str
    extras: JsonMapping = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_name(self.name)
        _require_non_negative_int(self.order, "IK constraint order")
        _require_tuple(self.bones, "bones")
        if not self.bones:
            raise ValueError(
                f"IK constraint '{self.name}' must reference at least one bone"
            )
        for bone in self.bones:
            _require_name(bone, "bones item")
        _require_name(self.target, "target")
        _validate_extras(
            self.extras,
            path="ik.extras",
            known_fields=("name", "order", "bones", "target"),
        )


@dataclass(frozen=True, slots=True)
class TransformConstraint:
    name: str
    order: int
    bones: Tuple[str, ...]
    target: str
    extras: JsonMapping = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_name(self.name)
        _require_non_negative_int(self.order, "Transform constraint order")
        _require_tuple(self.bones, "bones")
        if not self.bones:
            raise ValueError(
                f"Transform constraint '{self.name}' must reference at least one bone"
            )
        for bone in self.bones:
            _require_name(bone, "bones item")
        _require_name(self.target, "target")
        _validate_extras(
            self.extras,
            path="transform.extras",
            known_fields=("name", "order", "bones", "target"),
        )


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
        _require_tuple(self.uvs, "uvs")
        _require_tuple(self.triangles, "triangles")
        _require_tuple(self.vertices, "vertices")
        _require_tuple(self.edges, "edges")
        if len(self.uvs) % 2 != 0:
            raise ValueError(
                f"Mesh attachment '{self.name}' has an odd UV array length"
            )
        if len(self.triangles) % 3 != 0:
            raise ValueError(
                f"Mesh attachment '{self.name}' triangle array length must be "
                "divisible by 3"
            )
        if len(self.edges) % 2 != 0:
            raise ValueError("edges must contain vertex index pairs")
        _validate_finite_sequence(self.uvs, "uvs")
        _validate_index_sequence(self.triangles, "triangles")
        _validate_finite_sequence(self.vertices, "vertices")
        _validate_index_sequence(self.edges, "edges")
        _require_non_negative_int(self.hull, "hull")
        _require_optional_string(self.path, "path")
        _require_finite_number(self.width, "width")
        _require_finite_number(self.height, "height")
        if self.sequence is not None:
            validate_json_mapping(self.sequence, path="mesh.sequence")
        _validate_extras(
            self.extras,
            path="mesh.extras",
            known_fields=(
                "type",
                "uvs",
                "triangles",
                "vertices",
                "hull",
                "path",
                "edges",
                "width",
                "height",
                "sequence",
            ),
        )
        _validate_attachment_metadata(self.extras, path="mesh.extras")


@dataclass(frozen=True, slots=True)
class Skin:
    name: str
    attachments: Mapping[str, Mapping[str, MeshAttachment | JsonMapping]]
    bones: Tuple[str, ...] = ()
    constraints: Tuple[str, ...] = ()
    extras: JsonMapping = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_name(self.name)
        if not isinstance(self.attachments, Mapping):
            raise TypeError("attachments must be a mapping")
        for slot_name, slot_attachments in self.attachments.items():
            _require_name(slot_name, "skin slot name")
            if not isinstance(slot_attachments, Mapping):
                raise TypeError("skin attachment groups must be mappings")
            for attachment_name, attachment in slot_attachments.items():
                _require_name(attachment_name, "skin attachment name")
                if isinstance(attachment, MeshAttachment):
                    continue
                if not isinstance(attachment, Mapping):
                    raise TypeError(
                        "skin attachments must be MeshAttachment values or mappings"
                    )
                attachment_path = (
                    f"skin.attachments.{slot_name}.{attachment_name}"
                )
                validate_json_mapping(attachment, path=attachment_path)
                _validate_attachment_metadata(attachment, path=attachment_path)
        _require_tuple(self.bones, "bones")
        _require_tuple(self.constraints, "constraints")
        for bone_name in self.bones:
            _require_name(bone_name, "skin bones item")
        for constraint_name in self.constraints:
            _require_name(constraint_name, "skin constraints item")
        _validate_extras(
            self.extras,
            path="skin.extras",
            known_fields=("name", "attachments", "bones", "constraints"),
        )


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
        validate_json_mapping(self.skeleton, path="document.skeleton")
        for field_name, item_type in (
            ("bones", Bone),
            ("slots", Slot),
            ("skins", Skin),
            ("ik", IKConstraint),
            ("transform", TransformConstraint),
        ):
            values = _require_tuple(getattr(self, field_name), field_name)
            if not all(isinstance(item, item_type) for item in values):
                raise TypeError(
                    f"{field_name} must contain only {item_type.__name__} values"
                )
        if not self.bones:
            raise ValueError("SpineDocument must contain at least one bone")
        validate_json_mapping(self.animations, path="document.animations")
        validate_json_mapping(self.events, path="document.events")
        _validate_extras(
            self.extras,
            path="document.extras",
            known_fields=(
                "skeleton",
                "bones",
                "slots",
                "skins",
                "ik",
                "transform",
                "events",
                "animations",
            ),
        )
