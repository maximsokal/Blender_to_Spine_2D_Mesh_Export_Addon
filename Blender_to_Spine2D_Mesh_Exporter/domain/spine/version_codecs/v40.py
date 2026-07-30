"""Spine 4.0 JSON codec for the first production-compatible export scope.

Spine 4.0 uses the same legacy 4.x representation already handled by the Spine 4.1
codec for bones, skin constraint membership, and transform-constraint channel mixes.
The important schema difference for the add-on is that attachment and animation
sequences are not available in the registered 4.0 target and must fail closed instead
of being silently removed.
"""

from __future__ import annotations

import json
from typing import Any

from ..model import SpineDocument
from ..version_target import SpineJsonTarget
from .base import SpineJsonCodecContext
from .v41 import Spine41JsonCodec


def _sequence_paths(value: Any, *, path: str = "document") -> tuple[str, ...]:
    """Return every JSON path containing a Spine ``sequence`` member.

    The scan intentionally covers both setup attachments and animation timelines. A
    custom extra named ``sequence`` is also rejected because emitting an unknown field
    with target-specific semantics would make the 4.0 contract ambiguous.
    """

    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if key == "sequence":
                found.append(child_path)
            found.extend(_sequence_paths(child, path=child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_sequence_paths(child, path=f"{path}[{index}]"))
    return tuple(found)


class Spine40JsonCodec(Spine41JsonCodec):
    """Serialize the proven legacy 4.x schema for exact Spine 4.0.64 output."""

    @property
    def target(self) -> SpineJsonTarget:
        return SpineJsonTarget.SPINE_4_0

    def to_json(
        self,
        document: SpineDocument,
        *,
        context: SpineJsonCodecContext,
        indent: int = 2,
    ) -> str:
        if not isinstance(document, SpineDocument):
            raise TypeError("document must be SpineDocument")
        if not isinstance(context, SpineJsonCodecContext):
            raise TypeError("context must be SpineJsonCodecContext")
        if context.target is not self.target:
            raise ValueError(
                f"Spine40JsonCodec requires {self.target.value}, "
                f"got {context.target.value}"
            )

        encoded = super().to_json(
            document,
            context=context,
            indent=indent,
        )
        payload = json.loads(encoded)
        sequence_paths = _sequence_paths(payload)
        if sequence_paths:
            raise ValueError(
                "Spine 4.0.64 does not support attachment or animation sequences; "
                f"remove sequence data before export: {sequence_paths}"
            )
        return encoded


__all__ = ["Spine40JsonCodec"]
