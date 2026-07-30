"""Spine 4.2 codec preserving the current production serializer byte-for-byte."""

from __future__ import annotations

from ..model import SpineDocument
from ..serializer import SpineSerializer
from ..version_target import SpineJsonTarget
from .base import SpineJsonCodecContext, SpineJsonVersionCodec


class Spine42JsonCodec(SpineJsonVersionCodec):
    """Delegate Spine 4.2.43 output to the proven production serializer."""

    @property
    def target(self) -> SpineJsonTarget:
        return SpineJsonTarget.SPINE_4_2

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
                f"Spine42JsonCodec requires {self.target.value}, "
                f"got {context.target.value}"
            )

        # Do not normalize, copy, strip, or rewrite anything here. This exact delegation
        # is the regression gate that preserves the existing 4.2.43 JSON representation,
        # including current animation and attachment-sequence behavior.
        return SpineSerializer(validator=context.validator).to_json(
            document,
            indent=indent,
        )


__all__ = ["Spine42JsonCodec"]
