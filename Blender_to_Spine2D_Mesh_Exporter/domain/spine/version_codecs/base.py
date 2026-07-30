"""Blender-independent contracts for target-specific Spine JSON codecs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..model import SpineDocument
from ..validator import SpineValidator
from ..version_target import SpineJsonTarget


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


__all__ = ["SpineJsonCodecContext", "SpineJsonVersionCodec"]
