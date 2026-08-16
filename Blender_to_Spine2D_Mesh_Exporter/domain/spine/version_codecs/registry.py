"""Registry and facade for deterministic target-version Spine JSON serialization."""

from __future__ import annotations

import json
from types import MappingProxyType
from typing import Mapping

from ..model import SpineDocument
from ..validator import SpineValidator
from ..version_target import (
    SpineJsonTarget,
    require_spine_json_target_serializable,
)
from .base import (
    SpineJsonCodecContext,
    SpineJsonVersionCodec,
    validate_document_spine_version_for_target,
)
from .v38_camera_relative import Spine38CameraRelativeJsonCodec
from .v40 import Spine40JsonCodec
from .v41 import Spine41JsonCodec
from .v42 import Spine42JsonCodec
from .v43 import Spine43JsonCodec


_CODECS: Mapping[SpineJsonTarget, SpineJsonVersionCodec] = MappingProxyType(
    {
        SpineJsonTarget.SPINE_3_8: Spine38CameraRelativeJsonCodec(),
        SpineJsonTarget.SPINE_4_0: Spine40JsonCodec(),
        SpineJsonTarget.SPINE_4_1: Spine41JsonCodec(),
        SpineJsonTarget.SPINE_4_2: Spine42JsonCodec(),
        SpineJsonTarget.SPINE_4_3: Spine43JsonCodec(),
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
    return _CODECS


def resolve_spine_json_codec(value: object) -> SpineJsonVersionCodec:
    target = require_spine_json_target_serializable(value)
    codec = _CODECS.get(target)
    if codec is None:
        raise RuntimeError(
            f"No Spine JSON codec is registered for ready target {target.value}"
        )
    return codec


def _with_resolved_exact_version(
    encoded: str,
    *,
    exact_version: str,
    indent: int,
) -> str:
    """Patch only a legacy canonical cross-target serialization result.

    Same-family production documents already serialize their custom patch correctly, so
    they return byte-for-byte unchanged. Historical canonical 4.2-shaped documents may
    require the facade to replace the source marker with the target descriptor default.
    The replacement happens only on the detached JSON payload and never mutates the
    canonical ``SpineDocument``.
    """

    if not isinstance(encoded, str):
        raise TypeError("encoded must be str")
    if not isinstance(exact_version, str) or not exact_version:
        raise ValueError("exact_version must be a non-empty string")
    if isinstance(indent, bool) or not isinstance(indent, int):
        raise TypeError("indent must be int")

    payload = json.loads(encoded)
    if not isinstance(payload, dict):
        raise TypeError("Serialized Spine document must be a JSON object")
    skeleton = payload.get("skeleton")
    if not isinstance(skeleton, dict):
        raise TypeError("Serialized Spine document.skeleton must be a JSON object")

    if skeleton.get("spine") == exact_version:
        return encoded

    skeleton["spine"] = exact_version
    return json.dumps(
        payload,
        ensure_ascii=False,
        indent=indent,
        allow_nan=False,
    )


def serialize_spine_document(
    document: SpineDocument,
    target: object,
    *,
    indent: int = 2,
    validator: SpineValidator | None = None,
) -> str:
    if not isinstance(document, SpineDocument):
        raise TypeError("document must be SpineDocument")
    if validator is not None and not isinstance(validator, SpineValidator):
        raise TypeError("validator must be SpineValidator or None")

    resolved_target = require_spine_json_target_serializable(target)
    exact_version = validate_document_spine_version_for_target(
        document,
        resolved_target,
    )
    codec = resolve_spine_json_codec(resolved_target)
    encoded = codec.to_json(
        document,
        context=SpineJsonCodecContext(
            target=resolved_target,
            validator=validator,
        ),
        indent=indent,
    )
    return _with_resolved_exact_version(
        encoded,
        exact_version=exact_version,
        indent=indent,
    )


__all__ = [
    "registered_spine_json_codecs",
    "resolve_spine_json_codec",
    "serialize_spine_document",
]
