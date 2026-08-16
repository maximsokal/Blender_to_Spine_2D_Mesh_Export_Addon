"""Blender-independent contracts for target-specific Spine JSON codecs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping

from ..model import SpineDocument
from ..validator import SpineValidator
from ..version_target import (
    DEFAULT_SPINE_JSON_TARGET,
    DEFAULT_SPINE_JSON_VERSION,
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
    """Resolve the exact output version accepted by one target codec family.

    Production documents already carry the user-selected exact patch in
    ``SpineDocument.skeleton['spine']``. That same-family value must survive schema
    translation unchanged.

    Historical schema-adapter fixtures and direct canonical callers, however, use the
    rewrite's canonical 4.2 document shape and therefore carry
    ``DEFAULT_SPINE_JSON_VERSION`` even when asking a legacy/newer codec to translate the
    document to another target family. That established contract remains supported by
    resolving the canonical source marker to the selected target's descriptor default.

    Any other cross-family exact version is rejected. This prevents a custom project
    patch from being silently relabelled as another schema family while preserving the
    canonical cross-target adapter contract.
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

    raw_version = skeleton["spine"]

    # The maintained rewrite historically serializes one canonical 4.2-shaped document
    # through every target codec. Only that exact canonical marker may cross families.
    try:
        canonical_version = validate_spine_json_exact_version_for_target(
            DEFAULT_SPINE_JSON_TARGET,
            raw_version,
        )
    except (TypeError, ValueError):
        canonical_version = None

    if canonical_version == DEFAULT_SPINE_JSON_VERSION:
        return target.exact_version

    return validate_spine_json_exact_version_for_target(target, raw_version)


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
