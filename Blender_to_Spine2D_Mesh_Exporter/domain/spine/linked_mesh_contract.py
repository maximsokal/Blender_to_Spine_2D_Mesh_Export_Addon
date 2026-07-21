"""Setup-pose cross-reference contract for Spine 4.2 linked meshes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .model import MeshAttachment, Skin
from .spine_json_contract import json_path_key


_LINKED_MESH_TYPES = frozenset({"linkedmesh"})
_MESH_PARENT_TYPES = frozenset({"mesh", "linkedmesh"})
_DEFAULT_SKIN_NAME = "default"


@dataclass(frozen=True, slots=True)
class AttachmentReference:
    """One exact setup attachment location."""

    skin_name: str
    slot_name: str
    attachment_name: str


@dataclass(frozen=True, slots=True)
class ResolvedLinkedMesh:
    """A linked attachment and its terminal non-linked mesh parent."""

    source: AttachmentReference
    terminal: AttachmentReference
    terminal_attachment: MeshAttachment | Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class _AttachmentRecord:
    reference: AttachmentReference
    attachment: MeshAttachment | Mapping[str, Any]
    path: str


def _require_name(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be str")
    if not value.strip():
        raise ValueError(f"{field_name} cannot be empty")
    return value


def _raw_attachment_type(
    attachment: Mapping[str, Any],
    *,
    path: str,
) -> str:
    attachment_type = attachment.get("type", "region")
    if not isinstance(attachment_type, str):
        raise TypeError(f"{path}.type must be str")
    if not attachment_type.strip():
        raise ValueError(f"{path}.type cannot be empty")
    return attachment_type


def _is_linked_raw_attachment(
    attachment: Mapping[str, Any],
    *,
    path: str,
) -> bool:
    """Recognize canonical linkedmesh and the legacy mesh+parent spelling."""

    attachment_type = _raw_attachment_type(attachment, path=path)
    if attachment_type in _LINKED_MESH_TYPES:
        return True
    return attachment_type == "mesh" and attachment.get("parent") is not None


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

        self._path = path
        self._skin_by_name: dict[str, Skin] = {}
        self._skin_path_by_name: dict[str, str] = {}
        self._ambiguous_skin_names: set[str] = set()
        self._records: dict[AttachmentReference, _AttachmentRecord] = {}
        self._cache: dict[AttachmentReference, ResolvedLinkedMesh] = {}

        for skin_index, skin in enumerate(skins):
            if not isinstance(skin, Skin):
                raise TypeError(f"skins[{skin_index}] must be Skin")
            skin_path = f"{path}[{skin_index}]"
            if skin.name in self._skin_by_name:
                self._ambiguous_skin_names.add(skin.name)
            else:
                self._skin_by_name[skin.name] = skin
                self._skin_path_by_name[skin.name] = skin_path

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
                    self._records[reference] = _AttachmentRecord(
                        reference=reference,
                        attachment=attachment,
                        path=json_path_key(slot_path, attachment_name),
                    )

    def _require_skin(self, skin_name: str, *, path: str) -> Skin:
        if skin_name in self._ambiguous_skin_names:
            raise ValueError(f"{path} references duplicated skin '{skin_name}'")
        skin = self._skin_by_name.get(skin_name)
        if skin is None:
            raise ValueError(f"{path} references undefined skin '{skin_name}'")
        return skin

    def _require_record(
        self,
        reference: AttachmentReference,
        *,
        path: str,
    ) -> _AttachmentRecord:
        skin = self._require_skin(reference.skin_name, path=path)
        slot_attachments = skin.attachments.get(reference.slot_name)
        if slot_attachments is None:
            raise ValueError(
                f"{path} references slot '{reference.slot_name}' without "
                f"attachments in skin '{reference.skin_name}'"
            )
        if reference.attachment_name not in slot_attachments:
            raise ValueError(
                f"{path} references undefined parent attachment "
                f"'{reference.attachment_name}' for slot "
                f"'{reference.slot_name}' in skin '{reference.skin_name}'"
            )

        record = self._records.get(reference)
        if record is None:
            raise TypeError(f"{path} parent attachment has an unsupported value type")
        return record

    @staticmethod
    def _parent_reference(record: _AttachmentRecord) -> AttachmentReference | None:
        attachment = record.attachment
        if isinstance(attachment, MeshAttachment):
            return None

        attachment_type = _raw_attachment_type(attachment, path=record.path)
        is_linked = _is_linked_raw_attachment(attachment, path=record.path)
        if not is_linked:
            if attachment_type != "mesh":
                raise ValueError(
                    f"{record.path} resolves to non-mesh attachment type "
                    f"'{attachment_type}'"
                )
            return None

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

        record = self._require_record(
            reference,
            path=(
                f"linked mesh reference "
                f"{reference.skin_name!r}/{reference.slot_name!r}/"
                f"{reference.attachment_name!r}"
            ),
        )
        parent_reference = self._parent_reference(record)
        if parent_reference is None:
            result = ResolvedLinkedMesh(
                source=reference,
                terminal=reference,
                terminal_attachment=record.attachment,
            )
            self._cache[reference] = result
            return result

        parent_record = self._require_record(
            parent_reference,
            path=f"{record.path}.parent",
        )
        parent_attachment = parent_record.attachment
        if isinstance(parent_attachment, Mapping):
            parent_type = _raw_attachment_type(
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
            if not _is_linked_raw_attachment(attachment, path=record.path):
                continue
            resolved.append(self.resolve(record.reference))
        return tuple(resolved)


def validate_setup_linked_meshes(
    skins: tuple[Skin, ...],
    *,
    path: str = "document.skins",
) -> None:
    """Fail when any setup linked mesh cannot resolve to a terminal mesh."""

    LinkedMeshResolver(skins, path=path).validate_all()


__all__ = [
    "AttachmentReference",
    "LinkedMeshResolver",
    "ResolvedLinkedMesh",
    "validate_setup_linked_meshes",
]
