"""Blender-independent contracts for target-specific Spine JSON codecs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping

from ..model import SpineDocument
from ..validator import SpineValidator
from ..version_target import (
    SpineJsonTarget,
    validate_spine_json_exact_version_for_target,
)


@dataclass(frozen=True, slots=True)
class SpineJsonCodecContext:
    """Immutable dependencies supplied to one target codec invocation."""

    target: SpineJsonTarget
    validator: SpineValidator | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.target, SpineJsonTarget):
            raise TypeError("target must be SpineJsonTarget")
        if self.validator is not None and not isinstance(self.validator, SpineValidator):
            raise TypeError("validator must be SpineValidator or None")


def validate_document_spine_version_for_target(
    document: SpineDocument,
    target: SpineJsonTarget,
) -> str:
    """Validate the document-owned exact project version for one codec family.

    ``SpineDocument.skeleton['spine']`` is the single source of truth for the exact
    Editor/project patch version. Codecs own schema-family translation only and must not
    replace this value with their descriptor default. Failing here also protects direct
    codec callers from producing a JSON body whose declared Spine version belongs to a
    different schema family.
    """

    if not isinstance(document, SpineDocument):
        raise TypeError("document must be SpineDocument")
    if not isinstance(target, SpineJsonTarget):
        raise TypeError("target must be SpineJsonTarget")

    skeleton: Any = document.skeleton
    if not isinstance(skeleton, Mapping):
        raise TypeError("document.skeleton must be a mapping")
    if "spine" not in skeleton:
        raise ValueError("document.skeleton.spine is required for versioned export")

    return validate_spine_json_exact_version_for_target(
        target,
        skeleton["spine"],
    )


class SpineJsonVersionCodec(ABC):
    """Serialize one canonical :class:`SpineDocument` for exactly one target."""

    @property
    @abstractmethod
    def target(self) -> SpineJsonTarget:
        """Return the only target accepted by this codec."""

    @abstractmethod
    def to_json(
        self,
        document: SpineDocument,
        *,
        context: SpineJsonCodecContext,
        indent: int = 2,
    ) -> str:
        """Return deterministic JSON without mutating ``document``."""


__all__ = [
    "SpineJsonCodecContext",
    "SpineJsonVersionCodec",
    "validate_document_spine_version_for_target",
]
