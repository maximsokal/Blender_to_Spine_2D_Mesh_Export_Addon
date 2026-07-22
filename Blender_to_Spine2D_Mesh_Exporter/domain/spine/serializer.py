"""Deterministic serializer for :mod:`domain.spine.model`."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .curve_timeline_contract import validate_animation_curves
from .deform_timeline_contract import validate_animation_deform_timelines
from .linked_mesh_contract import validate_setup_linked_meshes
from .model import (
    Bone,
    IKConstraint,
    MeshAttachment,
    Skin,
    Slot,
    SpineDocument,
    TransformConstraint,
)
from .sequence_timeline_contract import validate_animation_sequence_timelines
from .slot_color_timeline_contract import (
    validate_animation_slot_color_timelines,
)
from .validator import SpineValidator


def _merge_known_and_extras(
    known: dict[str, Any],
    extras: Mapping[str, Any],
    *,
    path: str,
) -> dict[str, Any]:
    collision = tuple(sorted(set(known).intersection(extras)))
    if collision:
        raise ValueError(
            f"{path}: extras cannot overwrite known fields: {', '.join(collision)}"
        )
    known.update(extras)
    return known


def _put_optional(target: dict[str, Any], key: str, value: Any) -> None:
    if value is not None:
        target[key] = value


def _validate_indent(indent: int) -> int:
    if isinstance(indent, bool) or not isinstance(indent, int):
        raise TypeError("indent must be a non-negative integer")
    if indent < 0:
        raise ValueError("indent must be a non-negative integer")
    return indent


class SpineSerializer:
    """Serialize only documents that satisfy the complete Spine output contract."""

    def __init__(self, validator: SpineValidator | None = None) -> None:
        if validator is not None and not isinstance(validator, SpineValidator):
            raise TypeError("validator must be SpineValidator or None")
        self._validator = validator or SpineValidator()

    def bone_to_dict(self, bone: Bone, *, path: str = "bone") -> dict[str, Any]:
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
        return _merge_known_and_extras(data, bone.extras, path=f"{path}.extras")

    def slot_to_dict(self, slot: Slot, *, path: str = "slot") -> dict[str, Any]:
        data: dict[str, Any] = {"name": slot.name, "bone": slot.bone}
        _put_optional(data, "attachment", slot.attachment)
        _put_optional(data, "color", slot.color)
        _put_optional(data, "blend", slot.blend)
        return _merge_known_and_extras(data, slot.extras, path=f"{path}.extras")

    def ik_to_dict(
        self,
        constraint: IKConstraint,
        *,
        path: str = "ik",
    ) -> dict[str, Any]:
        return _merge_known_and_extras(
            {
                "name": constraint.name,
                "order": constraint.order,
                "bones": list(constraint.bones),
                "target": constraint.target,
            },
            constraint.extras,
            path=f"{path}.extras",
        )

    def transform_to_dict(
        self,
        constraint: TransformConstraint,
        *,
        path: str = "transform",
    ) -> dict[str, Any]:
        return _merge_known_and_extras(
            {
                "name": constraint.name,
                "order": constraint.order,
                "bones": list(constraint.bones),
                "target": constraint.target,
            },
            constraint.extras,
            path=f"{path}.extras",
        )

    def attachment_to_dict(
        self,
        attachment: MeshAttachment | Mapping[str, Any],
        *,
        path: str = "attachment",
    ) -> dict[str, Any]:
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
        _put_optional(
            data,
            "sequence",
            dict(attachment.sequence) if attachment.sequence else None,
        )
        return _merge_known_and_extras(
            data,
            attachment.extras,
            path=f"{path}.extras",
        )

    def skin_to_dict(self, skin: Skin, *, path: str = "skin") -> dict[str, Any]:
        attachments: dict[str, dict[str, Any]] = {}
        for slot_name, slot_attachments in skin.attachments.items():
            attachments[slot_name] = {
                attachment_name: self.attachment_to_dict(
                    attachment,
                    path=f"{path}.attachments.{slot_name}.{attachment_name}",
                )
                for attachment_name, attachment in slot_attachments.items()
            }
        data: dict[str, Any] = {"name": skin.name, "attachments": attachments}
        if skin.bones:
            data["bones"] = list(skin.bones)
        if skin.constraints:
            data["constraints"] = list(skin.constraints)
        return _merge_known_and_extras(
            data,
            skin.extras,
            path=f"{path}.extras",
        )

    def to_dict(self, document: SpineDocument) -> dict[str, Any]:
        if not isinstance(document, SpineDocument):
            raise TypeError("document must be SpineDocument")
        self._validator.validate_or_raise(document)
        linked_mesh_resolver = validate_setup_linked_meshes(
            document.skins,
            path="document.skins",
        )
        validate_animation_slot_color_timelines(
            document.animations,
            slot_names=tuple(slot.name for slot in document.slots),
            path="document.animations",
        )
        validate_animation_curves(
            document.animations,
            path="document.animations",
        )
        validate_animation_deform_timelines(
            document.animations,
            skins=document.skins,
            slot_names=tuple(slot.name for slot in document.slots),
            path="document.animations",
            linked_mesh_resolver=linked_mesh_resolver,
        )
        validate_animation_sequence_timelines(
            document.animations,
            skins=document.skins,
            slot_names=tuple(slot.name for slot in document.slots),
            path="document.animations",
            linked_mesh_resolver=linked_mesh_resolver,
        )

        data: dict[str, Any] = {
            "skeleton": dict(document.skeleton),
            "bones": [
                self.bone_to_dict(bone, path=f"bones[{index}]")
                for index, bone in enumerate(document.bones)
            ],
            "slots": [
                self.slot_to_dict(slot, path=f"slots[{index}]")
                for index, slot in enumerate(document.slots)
            ],
            "skins": [
                self.skin_to_dict(skin, path=f"skins[{index}]")
                for index, skin in enumerate(document.skins)
            ],
        }
        if document.ik:
            data["ik"] = [
                self.ik_to_dict(item, path=f"ik[{index}]")
                for index, item in enumerate(document.ik)
            ]
        if document.transform:
            data["transform"] = [
                self.transform_to_dict(item, path=f"transform[{index}]")
                for index, item in enumerate(document.transform)
            ]
        if document.events:
            data["events"] = dict(document.events)
        data["animations"] = dict(document.animations)
        return _merge_known_and_extras(
            data,
            document.extras,
            path="document.extras",
        )

    def to_json(self, document: SpineDocument, *, indent: int = 2) -> str:
        resolved_indent = _validate_indent(indent)
        return json.dumps(
            self.to_dict(document),
            ensure_ascii=False,
            indent=resolved_indent,
            allow_nan=False,
        )

    def write_json(
        self,
        document: SpineDocument,
        output_path: Path,
        *,
        indent: int = 2,
    ) -> Path:
        if not isinstance(output_path, Path):
            raise TypeError("output_path must be pathlib.Path")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            self.to_json(document, indent=indent),
            encoding="utf-8",
        )
        return output_path
