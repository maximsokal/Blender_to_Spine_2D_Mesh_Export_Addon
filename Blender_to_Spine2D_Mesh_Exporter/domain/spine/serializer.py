"""Deterministic serializer for :mod:`domain.spine.model`."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .model import (
    Bone,
    IKConstraint,
    MeshAttachment,
    Skin,
    Slot,
    SpineDocument,
    TransformConstraint,
)


def _merge_known_and_extras(known: dict[str, Any], extras: Mapping[str, Any]) -> dict[str, Any]:
    collision = set(known).intersection(extras)
    if collision:
        raise ValueError("extras cannot overwrite known fields: " + ", ".join(sorted(collision)))
    known.update(extras)
    return known


def _put_optional(target: dict[str, Any], key: str, value: Any) -> None:
    if value is not None:
        target[key] = value


class SpineSerializer:
    """Serialize a validated document while preserving declared sequence order."""

    def bone_to_dict(self, bone: Bone) -> dict[str, Any]:
        data: dict[str, Any] = {"name": bone.name}
        _put_optional(data, "parent", bone.parent)
        _put_optional(data, "length", bone.length)
        _put_optional(data, "x", bone.x)
        _put_optional(data, "y", bone.y)
        _put_optional(data, "rotation", bone.rotation)
        _put_optional(data, "scaleX", bone.scale_x)
        _put_optional(data, "scaleY", bone.scale_y)
        _put_optional(data, "color", bone.color)
        _put_optional(data, "icon", bone.icon)
        return _merge_known_and_extras(data, bone.extras)

    def slot_to_dict(self, slot: Slot) -> dict[str, Any]:
        data: dict[str, Any] = {"name": slot.name, "bone": slot.bone}
        _put_optional(data, "attachment", slot.attachment)
        _put_optional(data, "color", slot.color)
        _put_optional(data, "blend", slot.blend)
        return _merge_known_and_extras(data, slot.extras)

    def ik_to_dict(self, constraint: IKConstraint) -> dict[str, Any]:
        return _merge_known_and_extras(
            {
                "name": constraint.name,
                "order": constraint.order,
                "bones": list(constraint.bones),
                "target": constraint.target,
            },
            constraint.extras,
        )

    def transform_to_dict(self, constraint: TransformConstraint) -> dict[str, Any]:
        return _merge_known_and_extras(
            {
                "name": constraint.name,
                "order": constraint.order,
                "bones": list(constraint.bones),
                "target": constraint.target,
            },
            constraint.extras,
        )

    def attachment_to_dict(self, attachment: MeshAttachment | Mapping[str, Any]) -> dict[str, Any]:
        if isinstance(attachment, Mapping):
            return dict(attachment)
        data: dict[str, Any] = {
            "type": "mesh",
            "uvs": list(attachment.uvs),
            "triangles": list(attachment.triangles),
            "vertices": list(attachment.vertices),
            "hull": attachment.hull,
        }
        _put_optional(data, "path", attachment.path)
        if attachment.edges:
            data["edges"] = list(attachment.edges)
        _put_optional(data, "width", attachment.width)
        _put_optional(data, "height", attachment.height)
        _put_optional(data, "sequence", dict(attachment.sequence) if attachment.sequence else None)
        return _merge_known_and_extras(data, attachment.extras)

    def skin_to_dict(self, skin: Skin) -> dict[str, Any]:
        attachments: dict[str, dict[str, Any]] = {}
        for slot_name, slot_attachments in skin.attachments.items():
            attachments[slot_name] = {
                attachment_name: self.attachment_to_dict(attachment)
                for attachment_name, attachment in slot_attachments.items()
            }
        data: dict[str, Any] = {"name": skin.name, "attachments": attachments}
        if skin.bones:
            data["bones"] = list(skin.bones)
        if skin.constraints:
            data["constraints"] = list(skin.constraints)
        return _merge_known_and_extras(data, skin.extras)

    def to_dict(self, document: SpineDocument) -> dict[str, Any]:
        if not isinstance(document, SpineDocument):
            raise TypeError("document must be SpineDocument")
        data: dict[str, Any] = {
            "skeleton": dict(document.skeleton),
            "bones": [self.bone_to_dict(bone) for bone in document.bones],
            "slots": [self.slot_to_dict(slot) for slot in document.slots],
            "skins": [self.skin_to_dict(skin) for skin in document.skins],
        }
        if document.ik:
            data["ik"] = [self.ik_to_dict(item) for item in document.ik]
        if document.transform:
            data["transform"] = [
                self.transform_to_dict(item) for item in document.transform
            ]
        if document.events:
            data["events"] = dict(document.events)
        data["animations"] = dict(document.animations)
        return _merge_known_and_extras(data, document.extras)

    def to_json(self, document: SpineDocument, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(document), ensure_ascii=False, indent=indent)

    def write_json(self, document: SpineDocument, output_path: Path, *, indent: int = 2) -> Path:
        if not isinstance(output_path, Path):
            raise TypeError("output_path must be pathlib.Path")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_json(document, indent=indent), encoding="utf-8")
        return output_path
