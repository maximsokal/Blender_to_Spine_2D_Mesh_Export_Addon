"""Reusable cross-skin setup attachment-name index for Spine timelines."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any


_EMPTY_ATTACHMENT_NAMES: frozenset[str] = frozenset()


def _require_name(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be str")
    if not value.strip():
        raise ValueError(f"{field_name} cannot be empty")
    return value


@dataclass(frozen=True, slots=True)
class SetupAttachmentNameIndex:
    """Immutable union of setup attachment names grouped by slot across skins."""

    skin_attachments: tuple[Mapping[str, Mapping[str, Any]], ...]
    _names_by_slot: Mapping[str, frozenset[str]] = field(
        init=False,
        repr=False,
        compare=False,
        hash=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.skin_attachments, tuple):
            raise TypeError("skin_attachments must be tuple")

        mutable_names: dict[str, set[str]] = {}
        for skin_index, attachments in enumerate(self.skin_attachments):
            if not isinstance(attachments, Mapping):
                raise TypeError(
                    f"skin_attachments[{skin_index}] must be a mapping"
                )

            for slot_name, slot_attachments in attachments.items():
                resolved_slot_name = _require_name(
                    slot_name,
                    f"skin_attachments[{skin_index}] slot name",
                )
                if not isinstance(slot_attachments, Mapping):
                    raise TypeError(
                        f"skin_attachments[{skin_index}][{resolved_slot_name!r}] "
                        "must be a mapping"
                    )

                names = mutable_names.setdefault(resolved_slot_name, set())
                for attachment_name in slot_attachments:
                    names.add(
                        _require_name(
                            attachment_name,
                            f"skin_attachments[{skin_index}]"
                            f"[{resolved_slot_name!r}] attachment name",
                        )
                    )

        object.__setattr__(
            self,
            "_names_by_slot",
            MappingProxyType(
                {
                    slot_name: frozenset(attachment_names)
                    for slot_name, attachment_names in mutable_names.items()
                }
            ),
        )

    def names_for_slot(self, slot_name: object) -> frozenset[str]:
        """Return the immutable cross-skin attachment-name union for one slot."""

        resolved_slot_name = _require_name(slot_name, "slot_name")
        return self._names_by_slot.get(
            resolved_slot_name,
            _EMPTY_ATTACHMENT_NAMES,
        )

    def require(
        self,
        slot_name: object,
        attachment_name: object,
        *,
        path: str,
    ) -> None:
        """Require one setup attachment name for a slot using caller diagnostics."""

        if not isinstance(path, str) or not path:
            raise ValueError("path must be a non-empty string")
        resolved_slot_name = _require_name(slot_name, "slot_name")
        resolved_attachment_name = _require_name(attachment_name, path)
        if resolved_attachment_name not in self._names_by_slot.get(
            resolved_slot_name,
            _EMPTY_ATTACHMENT_NAMES,
        ):
            raise ValueError(
                f"{path} references undefined attachment "
                f"'{resolved_attachment_name}' for slot '{resolved_slot_name}'"
            )


def resolve_setup_attachment_name_index(
    skin_attachments: tuple[Mapping[str, Mapping[str, Any]], ...],
    setup_attachment_index: SetupAttachmentNameIndex | None,
) -> SetupAttachmentNameIndex:
    """Build a direct-call index or validate reuse of the exact skin snapshot."""

    if setup_attachment_index is None:
        return SetupAttachmentNameIndex(skin_attachments)
    if not isinstance(setup_attachment_index, SetupAttachmentNameIndex):
        raise TypeError(
            "setup_attachment_index must be SetupAttachmentNameIndex or None"
        )
    if setup_attachment_index.skin_attachments is not skin_attachments:
        raise ValueError(
            "setup_attachment_index must be built from the exact "
            "skin_attachments tuple"
        )
    return setup_attachment_index


__all__ = [
    "SetupAttachmentNameIndex",
    "resolve_setup_attachment_name_index",
]
