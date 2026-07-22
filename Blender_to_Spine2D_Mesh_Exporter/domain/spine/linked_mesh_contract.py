"""Setup-pose cross-reference contract for Spine 4.2 linked meshes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .model import MeshAttachment, Skin
from .spine_json_contract import json_path_key
from .spine_scalar_contract import require_name as _require_name


_LINKED_MESH_TYPES = frozenset({"linkedmesh"})
_MESH_PARENT_TYPES = frozenset({"mesh", "linkedmesh"})
_DEFAULT_SKIN_NAME = "default"
_HEX_DIGITS = frozenset("0123456789abcdefABCDEF")


@dataclass(frozen=True, slots=True)
class AttachmentReference:
    """One exact setup attachment location."""

    skin_name: str
    slot_name: str
    attachment_name: str


@dataclass(frozen=True, slots=True)
class SetupAttachment:
    """One indexed setup attachment and its exact JSON path."""

    reference: AttachmentReference
    attachment: MeshAttachment | Mapping[str, Any]
    path: str


@dataclass(frozen=True, slots=True)
class ResolvedLinkedMesh:
    """A linked attachment and its terminal non-linked mesh parent."""

    source: AttachmentReference
    terminal: AttachmentReference
    terminal_attachment: MeshAttachment | Mapping[str, Any]
    terminal_path: str


def raw_attachment_type(
    attachment: Mapping[str, Any],
    *,
    path: str,
) -> str:
    """Return the exact Spine attachment type without case normalization."""

    attachment_type = attachment.get("type", "region")
    if not isinstance(attachment_type, str):
        raise TypeError(f"{path}.type must be str")
    if not attachment_type.strip():
        raise ValueError(f"{path}.type cannot be empty")
    return attachment_type


def is_linked_mesh_attachment(
    attachment: Mapping[str, Any],
    *,
    path: str,
) -> bool:
    """Recognize canonical linkedmesh and runtime mesh+truthy-parent spelling."""

    attachment_type = raw_attachment_type(attachment, path=path)
    if attachment_type in _LINKED_MESH_TYPES:
        return True
    return attachment_type == "mesh" and bool(attachment.get("parent"))


# Private aliases retain compatibility for existing architecture assertions while
# production consumers use the explicit public names above.
_raw_attachment_type = raw_attachment_type
_is_linked_raw_attachment = is_linked_mesh_attachment


def _validate_optional_string_metadata(
    attachment: Mapping[str, Any],
    *,
    field_name: str,
    path: str,
) -> None:
    if field_name not in attachment:
        return
    value = attachment[field_name]
    if not isinstance(value, str):
        raise TypeError(f"{path}.{field_name} must be str")


def _validate_optional_color_metadata(
    attachment: Mapping[str, Any],
    *,
    path: str,
) -> None:
    if "color" not in attachment:
        return

    color = attachment["color"]
    if color is None or color == "":
        return
    if not isinstance(color, str):
        raise TypeError(f"{path}.color must be str or None")

    hexadecimal = color[1:] if color.startswith("#") else color
    if len(hexadecimal) not in (6, 8) or any(
        character not in _HEX_DIGITS for character in hexadecimal
    ):
        raise ValueError(
            f"{path}.color must contain 6 or 8 hexadecimal digits, "
            "optionally prefixed by '#'"
        )


def _validate_linked_metadata(record: SetupAttachment) -> None:
    attachment = record.attachment
    if isinstance(attachment, MeshAttachment):
        return

    for field_name in ("name", "path"):
        _validate_optional_string_metadata(
            attachment,
            field_name=field_name,
            path=record.path,
        )

    _validate_optional_color_metadata(attachment, path=record.path)

    if "timelines" in attachment and not isinstance(
        attachment["timelines"],
        bool,
    ):
        raise TypeError(f"{record.path}.timelines must be bool")

    sequence = attachment.get("sequence")
    if sequence is not None and not isinstance(sequence, Mapping):
        raise TypeError(f"{record.path}.sequence must be a mapping or None")


class LinkedMeshResolver:
    """Resolve and validate all setup linked meshes without mutating mappings."""

    def __init__(
        self,
        skins: tuple[Skin, ...],
        *,
        path: str = "document.skins",
    ) -> None:
        if not isinstance(skins, tuple):
            raise TypeError("skins must be tuple")
        if not isinstance(path, str) or not path:
            raise ValueError("path must be a non-empty string")

        self._skins = skins
        self._path = path
        self._skin_by_name: dict[str, Skin] = {}
        self._ambiguous_skin_names: set[str] = set()
        self._records: dict[AttachmentReference, SetupAttachment] = {}
        self._cache: dict[AttachmentReference, ResolvedLinkedMesh] = {}

        for skin_index, skin in enumerate(skins):
            if not isinstance(skin, Skin):
                raise TypeError(f"skins[{skin_index}] must be Skin")
            skin_path = f"{path}[{skin_index}]"
            if skin.name in self._skin_by_name:
                self._ambiguous_skin_names.add(skin.name)
            else:
                self._skin_by_name[skin.name] = skin

            attachments_path = f"{skin_path}.attachments"
            for slot_name, slot_attachments in skin.attachments.items():
                slot_path = json_path_key(attachments_path, slot_name)
                for attachment_name, attachment in slot_attachments.items():
                    if not isinstance(attachment, (MeshAttachment, Mapping)):
                        continue
                    reference = AttachmentReference(
                        skin_name=skin.name,
                        slot_name=slot_name,
                        attachment_name=attachment_name,
                    )
                    self._records[reference] = SetupAttachment(
                        reference=reference,
                        attachment=attachment,
                        path=json_path_key(slot_path, attachment_name),
                    )

    @property
    def skins(self) -> tuple[Skin, ...]:
        """Return the exact immutable skin tuple used to build this index."""

        return self._skins

    def require_skin(self, skin_name: str, *, path: str) -> Skin:
        """Return one unambiguous setup skin or fail with the caller path."""

        _require_name(skin_name, f"{path} skin name")
        if skin_name in self._ambiguous_skin_names:
            raise ValueError(f"{path} references duplicated skin '{skin_name}'")
        skin = self._skin_by_name.get(skin_name)
        if skin is None:
            raise ValueError(f"{path} references undefined skin '{skin_name}'")
        return skin

    def get_attachment(
        self,
        reference: AttachmentReference,
        *,
        path: str,
    ) -> SetupAttachment:
        """Return one exact setup attachment without resolving parent geometry."""

        if not isinstance(reference, AttachmentReference):
            raise TypeError("reference must be AttachmentReference")
        if not isinstance(path, str) or not path:
            raise ValueError("path must be a non-empty string")

        skin = self.require_skin(reference.skin_name, path=path)
        slot_attachments = skin.attachments.get(reference.slot_name)
        if slot_attachments is None:
            raise ValueError(
                f"{path} references slot '{reference.slot_name}' without "
                f"attachments in skin '{reference.skin_name}'"
            )
        if reference.attachment_name not in slot_attachments:
            raise ValueError(
                f"{path} references undefined attachment "
                f"'{reference.attachment_name}' for slot "
                f"'{reference.slot_name}' in skin '{reference.skin_name}'"
            )

        record = self._records.get(reference)
        if record is None:
            raise TypeError(f"{path} setup attachment has an unsupported value type")
        return record

    @staticmethod
    def _parent_reference(record: SetupAttachment) -> AttachmentReference | None:
        attachment = record.attachment
        if isinstance(attachment, MeshAttachment):
            return None

        attachment_type = raw_attachment_type(attachment, path=record.path)
        is_linked = is_linked_mesh_attachment(attachment, path=record.path)
        if not is_linked:
            if attachment_type != "mesh":
                raise ValueError(
                    f"{record.path} resolves to non-mesh attachment type "
                    f"'{attachment_type}'"
                )
            return None

        _validate_linked_metadata(record)

        if "parent" not in attachment or attachment["parent"] is None:
            raise ValueError(f"{record.path}.parent is required for a linked mesh")
        parent_name = _require_name(
            attachment["parent"],
            f"{record.path}.parent",
        )

        raw_skin_name = attachment.get("skin")
        if raw_skin_name in (None, ""):
            parent_skin_name = _DEFAULT_SKIN_NAME
        else:
            parent_skin_name = _require_name(
                raw_skin_name,
                f"{record.path}.skin",
            )

        return AttachmentReference(
            skin_name=parent_skin_name,
            slot_name=record.reference.slot_name,
            attachment_name=parent_name,
        )

    def resolve(
        self,
        reference: AttachmentReference,
    ) -> ResolvedLinkedMesh:
        if not isinstance(reference, AttachmentReference):
            raise TypeError("reference must be AttachmentReference")

        return self._resolve(reference, stack=())

    def _resolve(
        self,
        reference: AttachmentReference,
        *,
        stack: tuple[AttachmentReference, ...],
    ) -> ResolvedLinkedMesh:
        cached = self._cache.get(reference)
        if cached is not None:
            return cached

        if reference in stack:
            cycle_start = stack.index(reference)
            cycle = stack[cycle_start:] + (reference,)
            rendered = " -> ".join(
                f"{item.skin_name}/{item.slot_name}/{item.attachment_name}"
                for item in cycle
            )
            record = self._records.get(reference)
            cycle_path = record.path if record is not None else self._path
            raise ValueError(
                f"{cycle_path} participates in a linked mesh parent cycle: "
                f"{rendered}"
            )

        reference_path = (
            f"linked mesh reference {reference.skin_name!r}/"
            f"{reference.slot_name!r}/{reference.attachment_name!r}"
        )
        record = self.get_attachment(reference, path=reference_path)
        parent_reference = self._parent_reference(record)
        if parent_reference is None:
            result = ResolvedLinkedMesh(
                source=reference,
                terminal=reference,
                terminal_attachment=record.attachment,
                terminal_path=record.path,
            )
            self._cache[reference] = result
            return result

        parent_record = self.get_attachment(
            parent_reference,
            path=f"{record.path}.parent",
        )
        parent_attachment = parent_record.attachment
        if isinstance(parent_attachment, Mapping):
            parent_type = raw_attachment_type(
                parent_attachment,
                path=parent_record.path,
            )
            if parent_type not in _MESH_PARENT_TYPES:
                raise ValueError(
                    f"{record.path}.parent resolves to unsupported attachment "
                    f"type '{parent_type}' at {parent_record.path}"
                )

        terminal = self._resolve(
            parent_reference,
            stack=stack + (reference,),
        )
        result = ResolvedLinkedMesh(
            source=reference,
            terminal=terminal.terminal,
            terminal_attachment=terminal.terminal_attachment,
            terminal_path=terminal.terminal_path,
        )
        self._cache[reference] = result
        return result

    def validate_all(self) -> tuple[ResolvedLinkedMesh, ...]:
        """Resolve every raw linked mesh in deterministic skin order."""

        resolved: list[ResolvedLinkedMesh] = []
        for record in self._records.values():
            attachment = record.attachment
            if not isinstance(attachment, Mapping):
                continue
            if not is_linked_mesh_attachment(attachment, path=record.path):
                continue
            resolved.append(self.resolve(record.reference))
        return tuple(resolved)


def validate_setup_linked_meshes(
    skins: tuple[Skin, ...],
    *,
    path: str = "document.skins",
) -> LinkedMeshResolver:
    """Validate setup links and return the reusable resolver/index."""

    resolver = LinkedMeshResolver(skins, path=path)
    resolver.validate_all()
    return resolver


__all__ = [
    "AttachmentReference",
    "LinkedMeshResolver",
    "ResolvedLinkedMesh",
    "SetupAttachment",
    "is_linked_mesh_attachment",
    "raw_attachment_type",
    "validate_setup_linked_meshes",
]
