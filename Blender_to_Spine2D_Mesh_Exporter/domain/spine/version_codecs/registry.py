"""Registry and facade for deterministic target-version Spine JSON serialization."""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

from ..model import SpineDocument
from ..validator import SpineValidator
from ..version_target import (
    SpineJsonTarget,
    require_spine_json_target_serializable,
)
from .base import SpineJsonCodecContext, SpineJsonVersionCodec
from .v41 import Spine41JsonCodec
from .v42 import Spine42JsonCodec


_CODECS: Mapping[SpineJsonTarget, SpineJsonVersionCodec] = MappingProxyType(
    {
        SpineJsonTarget.SPINE_4_1: Spine41JsonCodec(),
        SpineJsonTarget.SPINE_4_2: Spine42JsonCodec(),
    }
)


def _validate_registry() -> None:
    for target, codec in _CODECS.items():
        if not isinstance(target, SpineJsonTarget):
            raise RuntimeError("Spine JSON codec registry keys must be SpineJsonTarget")
        if not isinstance(codec, SpineJsonVersionCodec):
            raise RuntimeError(
                f"Codec registered for {target.value} must implement SpineJsonVersionCodec"
            )
        if codec.target is not target:
            raise RuntimeError(
                f"Codec registry key {target.value} does not match codec target "
                f"{codec.target.value}"
            )

    ready_targets = {
        target for target in SpineJsonTarget if target.descriptor.serializer_ready
    }
    registered_targets = set(_CODECS)
    if ready_targets != registered_targets:
        missing = tuple(
            sorted(target.value for target in ready_targets - registered_targets)
        )
        unexpected = tuple(
            sorted(target.value for target in registered_targets - ready_targets)
        )
        raise RuntimeError(
            "Spine JSON codec registry and serializer_ready capabilities disagree: "
            f"missing={missing}, unexpected={unexpected}"
        )


_validate_registry()


def registered_spine_json_codecs() -> Mapping[SpineJsonTarget, SpineJsonVersionCodec]:
    """Return the immutable production codec registry."""

    return _CODECS


def resolve_spine_json_codec(value: object) -> SpineJsonVersionCodec:
    """Resolve a production-ready target and return its registered codec."""

    target = require_spine_json_target_serializable(value)
    codec = _CODECS.get(target)
    if codec is None:
        # Registry validation makes this unreachable during a healthy import, but retain
        # an explicit failure if module state is corrupted by reload or test mutation.
        raise RuntimeError(
            f"No Spine JSON codec is registered for ready target {target.value}"
        )
    return codec


def serialize_spine_document(
    document: SpineDocument,
    target: object,
    *,
    indent: int = 2,
    validator: SpineValidator | None = None,
) -> str:
    """Serialize ``document`` through the only codec registered for ``target``."""

    if not isinstance(document, SpineDocument):
        raise TypeError("document must be SpineDocument")
    if validator is not None and not isinstance(validator, SpineValidator):
        raise TypeError("validator must be SpineValidator or None")

    resolved_target = require_spine_json_target_serializable(target)
    codec = resolve_spine_json_codec(resolved_target)
    return codec.to_json(
        document,
        context=SpineJsonCodecContext(
            target=resolved_target,
            validator=validator,
        ),
        indent=indent,
    )


__all__ = [
    "registered_spine_json_codecs",
    "resolve_spine_json_codec",
    "serialize_spine_document",
]
