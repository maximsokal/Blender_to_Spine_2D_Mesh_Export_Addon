"""Reusable setup-slot index for Spine animation output boundaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

from .spine_scalar_contract import require_name as _require_name


@dataclass(frozen=True, slots=True)
class SetupSlotIndex:
    """Immutable name-to-index view of one exact setup slot tuple."""

    slot_names: tuple[str, ...]
    _index_by_name: Mapping[str, int] = field(
        init=False, repr=False, compare=False, hash=False
    )
    _ambiguous_names: frozenset[str] = field(
        init=False, repr=False, compare=False, hash=False
    )

    def __post_init__(self) -> None:
        if not isinstance(self.slot_names, tuple):
            raise TypeError("slot_names must be tuple")

        index_by_name: dict[str, int] = {}
        ambiguous_names: set[str] = set()
        for slot_index, slot_name in enumerate(self.slot_names):
            _require_name(slot_name, f"slot_names[{slot_index}]")
            if slot_name in index_by_name:
                ambiguous_names.add(slot_name)
            else:
                index_by_name[slot_name] = slot_index

        object.__setattr__(
            self,
            "_index_by_name",
            MappingProxyType(index_by_name),
        )
        object.__setattr__(
            self,
            "_ambiguous_names",
            frozenset(ambiguous_names),
        )

    def require(self, slot_name: object, *, path: str) -> int:
        """Return one unambiguous setup slot index or fail with the caller path."""

        if not isinstance(path, str) or not path:
            raise ValueError("path must be a non-empty string")
        resolved_name = _require_name(slot_name, f"{path} slot name")
        if resolved_name not in self._index_by_name:
            raise ValueError(
                f"{path} references undefined slot '{resolved_name}'"
            )
        if resolved_name in self._ambiguous_names:
            raise ValueError(
                f"{path} references duplicated setup slot '{resolved_name}'"
            )
        return self._index_by_name[resolved_name]


def resolve_setup_slot_index(
    slot_names: tuple[str, ...],
    setup_slot_index: SetupSlotIndex | None,
) -> SetupSlotIndex:
    """Build a direct-call index or validate reuse of the exact slot tuple."""

    if setup_slot_index is None:
        return SetupSlotIndex(slot_names)
    if not isinstance(setup_slot_index, SetupSlotIndex):
        raise TypeError("setup_slot_index must be SetupSlotIndex or None")
    if setup_slot_index.slot_names is not slot_names:
        raise ValueError(
            "setup_slot_index must be built from the exact slot_names tuple"
        )
    return setup_slot_index


__all__ = ["SetupSlotIndex", "resolve_setup_slot_index"]
