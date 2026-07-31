"""Spine 4.2 codec with runtime-safe ordered constraint scheduling."""

from __future__ import annotations

import json
from typing import Any

from ..model import SpineDocument
from ..serializer import SpineSerializer
from ..version_target import SpineJsonTarget
from .base import SpineJsonCodecContext, SpineJsonVersionCodec
from .runtime_constraint_order import normalize_runtime_constraint_orders


def _require_dict(value: Any, *, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{path} must be a JSON object")
    return value


class Spine42JsonCodec(SpineJsonVersionCodec):
    """Serialize exact Spine 4.2.43 JSON with a complete runtime update schedule.

    Canonical builders retain historical dependency numbers because the same typed
    document can target multiple Spine families. The 4.2 runtime, however, scans phases
    ``0..constraint_count-1`` and processes one constraint per phase. This codec therefore
    normalizes only the detached serialized constraint ``order`` fields. Rig topology,
    constraint payloads, animations, skins, attachments, and the canonical document stay
    unchanged.
    """

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

        canonical_json = SpineSerializer(validator=context.validator).to_json(
            document,
            indent=indent,
        )
        output = _require_dict(json.loads(canonical_json), path="document")
        normalize_runtime_constraint_orders(
            output,
            collections=("ik", "transform", "path", "physics"),
        )
        return json.dumps(
            output,
            ensure_ascii=False,
            indent=indent,
            allow_nan=False,
        )


__all__ = ["Spine42JsonCodec"]
